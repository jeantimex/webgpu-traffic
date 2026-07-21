import { MAX_CARS, NEW_CAR_PARAMS, type GuiState } from '../gui/settings_gui';
import { type Car, type IdmParams, type Obstacle } from '../sim/idm';
import { Road } from '../sim/road';
import { identity, lookAt, multiply, perspective, rotationZ, translationRotationY } from '../utils/mat4';
import { OrbitCamera } from '../utils/orbit';
import { createBufferWithData, resizeCanvasToDisplaySize, type WebGPUState } from '../webgpu/utils';
import { PALETTES, SCENES, pushBox, scene3State, type Palette, type SceneDef, type Vec3 } from './scenes';

const trafficShader = /* wgsl */ `
  struct Camera {
    viewProj: mat4x4f,
    lighting: vec4f, // x: ambient, y: diffuse strength
  };
  struct Draw {
    model: mat4x4f,
    tint: vec4f,
  };

  @group(0) @binding(0) var<uniform> camera: Camera;
  @group(0) @binding(1) var<uniform> draw: Draw;

  struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec3f,
  };

  struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) color: vec3f,
  };

  @vertex
  fn vertexMain(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = camera.viewProj * draw.model * vec4f(input.position, 1);
    output.normal = (draw.model * vec4f(input.normal, 0)).xyz;
    output.color = input.color * draw.tint.rgb;
    return output;
  }

  @fragment
  fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let light = normalize(vec3f(0.4, 0.9, 0.25));
    let diffuse = max(dot(normalize(input.normal), light), 0.0);
    return vec4f(input.color * (camera.lighting.x + camera.lighting.y * diffuse), 1);
  }
`;

/** Initial logical lanes: red/blue inner, yellow/white outer. Cars change lanes on their own. */
const START_LANES = [0, 0, 1, 1];
/** Start positions as fractions of the loop: each lane's pair begins half a lap apart. */
const START_FRACTIONS = [0, 0.5, 0.25, 0.75];
const SIM_STEP = 1 / 60;
/** Byte stride between per-draw uniform slots (WebGPU dynamic-offset alignment). */
const DRAW_STRIDE = 256;
const FLOATS_PER_DRAW = DRAW_STRIDE / Float32Array.BYTES_PER_ELEMENT;
/** Uniform slot of the first signal lamp: after 1 track slot + all car slots. */
const LAMP_SLOT = 1 + MAX_CARS;
/** Total uniform slots: track + cars + up to 4 lights × 3 lamps. */
const TOTAL_SLOTS = 1 + MAX_CARS + 12;
const NO_OBSTACLES: Obstacle[] = [];

/** Red on top, yellow in the middle, green at the bottom. */
const LAMP_HEIGHTS = { red: 3.2, yellow: 2.4, green: 1.6 } as const;
const LAMP_COLORS: Record<LightPhase, Vec3> = {
  red: [0.95, 0.15, 0.15],
  yellow: [0.95, 0.75, 0.1],
  green: [0.1, 0.85, 0.3],
};
const INACTIVE_LAMP_DIM = 0.25;

type LightPhase = 'red' | 'yellow' | 'green';

function lightPhase(clock: number, green: number, yellow: number, red: number): LightPhase {
  const t = clock % (green + yellow + red);
  return t < green ? 'green' : t < green + yellow ? 'yellow' : 'red';
}

/** A car-shaped box, white so the per-draw tint shows through. */
function buildCarMesh(carLength: number): number[] {
  const verts: number[] = [];
  pushBox(verts, [-carLength / 2, 0, -1], [carLength / 2, 1.5, 1], [1, 1, 1]);
  return verts;
}

/** The signal lamp box, white so the per-draw red/green tint shows through. */
function buildLampMesh(): number[] {
  const verts: number[] = [];
  pushBox(verts, [-0.35, 0, -0.35], [0.35, 0.7, 0.35], [1, 1, 1]);
  return verts;
}

export class Renderer {
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private vertexBuffer: GPUBuffer;
  private readonly cameraBuffer: GPUBuffer;
  private readonly drawBuffer: GPUBuffer;
  private readonly pipeline: GPURenderPipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly carVertexCount: number;
  private readonly lampFirstVertex: number;
  private readonly lampVertexCount: number;
  private readonly staticFirstVertex: number;
  private staticVertexCount: number;
  private carVerts: number[];
  private readonly lampVerts: number[];
  private cars: Car[] = [];
  private carParams: IdmParams[] = [];
  private def: SceneDef;
  private scene: number;
  private roadKey = '';
  private builtCarLength: number;
  private builtDayMode: boolean;
  private lightClock = 0;
  private readonly orbit: OrbitCamera;
  private depthTexture?: GPUTexture;
  private configured = false;
  private animationFrame?: number;
  private lastTime?: number;
  private accumulator = 0;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    gpu: WebGPUState,
    private readonly gui: GuiState,
  ) {
    this.device = gpu.device;
    this.context = gpu.context;
    this.format = gpu.format;

    this.scene = gui.settings.scene;
    if (this.scene === 3) {
      scene3State.road = new Road(this.gui.settings.scene3);
      this.roadKey = JSON.stringify(this.gui.settings.scene3);
    }
    this.def = SCENES[this.scene - 1];

    // Vertex layout: [car mesh][lamp mesh][static scene mesh] — car/lamp offsets are
    // fixed so the static part can be rebuilt (palette or scene switch) independently.
    this.carVerts = buildCarMesh(this.gui.settings.carLength);
    this.lampVerts = buildLampMesh();
    this.carVertexCount = this.carVerts.length / 9;
    this.lampFirstVertex = this.carVertexCount;
    this.lampVertexCount = this.lampVerts.length / 9;
    this.staticFirstVertex = this.lampFirstVertex + this.lampVertexCount;
    const staticVerts = this.def.buildStatic(this.palette());
    this.staticVertexCount = staticVerts.length / 9;
    this.vertexBuffer = createBufferWithData(
      this.device,
      'scene vertices',
      new Float32Array([...this.carVerts, ...this.lampVerts, ...staticVerts]),
      GPUBufferUsage.VERTEX,
    );

    this.cameraBuffer = this.device.createBuffer({
      label: 'camera uniforms',
      size: 80, // mat4x4 viewProj + vec4 lighting
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.drawBuffer = this.device.createBuffer({
      label: 'per-draw uniforms',
      size: DRAW_STRIDE * TOTAL_SLOTS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 80 },
        },
      ],
    });

    const shaderModule = this.device.createShaderModule({ code: trafficShader });
    this.pipeline = this.device.createRenderPipeline({
      label: 'traffic pipeline',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: shaderModule,
        entryPoint: 'vertexMain',
        buffers: [
          {
            arrayStride: 9 * Float32Array.BYTES_PER_ELEMENT,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 12, format: 'float32x3' },
              { shaderLocation: 2, offset: 24, format: 'float32x3' },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fragmentMain',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: this.drawBuffer, offset: 0, size: 80 } },
      ],
    });

    this.resetCars(false);
    this.builtCarLength = this.gui.settings.carLength;
    this.builtDayMode = this.gui.settings.dayMode;
    this.orbit = new OrbitCamera(canvas);
  }

  start(): void {
    if (this.animationFrame !== undefined) return;
    this.animationFrame = requestAnimationFrame(this.render);
  }

  stop(): void {
    if (this.animationFrame === undefined) return;
    cancelAnimationFrame(this.animationFrame);
    this.animationFrame = undefined;
  }

  destroy(): void {
    this.stop();
    this.depthTexture?.destroy();
    this.vertexBuffer.destroy();
    this.cameraBuffer.destroy();
    this.drawBuffer.destroy();
  }

  private palette(): Palette {
    return PALETTES[this.gui.settings.dayMode ? 'day' : 'night'];
  }

  /** Light phase: the GUI override wins; 'auto' runs the timed cycle. */
  private currentPhase(): LightPhase {
    const { green, yellow, red, override } = this.gui.settings.light;
    return override === 'auto' ? lightPhase(this.lightClock, green, yellow, red) : override;
  }

  /** Switches the active scene: rebuilds the static mesh and restarts traffic. */
  private applyScene(scene: number, preserveCars = false): void {
    this.scene = scene;
    if (scene === 3) {
      scene3State.road = new Road(this.gui.settings.scene3);
      this.roadKey = JSON.stringify(this.gui.settings.scene3);
    }
    this.def = SCENES[scene - 1];
    const staticVerts = this.def.buildStatic(this.palette());
    this.staticVertexCount = staticVerts.length / 9;
    this.vertexBuffer.destroy();
    this.vertexBuffer = createBufferWithData(
      this.device,
      'scene vertices',
      new Float32Array([...this.carVerts, ...this.lampVerts, ...staticVerts]),
      GPUBufferUsage.VERTEX,
    );
    this.resetCars(preserveCars);
    this.lightClock = 0;
  }

  /**
   * Places all cars for the current scene. Fresh scenes deal cars round-robin; a scene-3
   * config rebuild preserves each car's lane/position/speed instead (clamped to what
   * still exists), so adding a lane doesn't scatter traffic.
   */
  private resetCars(preserve: boolean): void {
    const c = this.def.c;
    const numLanes = this.scene === 3 ? (scene3State.road?.lanes.length ?? 1) : 2;
    const laneFor = (i: number): number =>
      this.scene === 3 ? i % numLanes : START_LANES[i % START_LANES.length] % numLanes;
    this.cars = this.gui.settings.cars.map((params, i) => {
      const prev = preserve && this.carParams[i] === params ? this.cars[i] : undefined;
      if (prev) {
        const lane = Math.min(prev.lane, numLanes - 1);
        return {
          ...prev,
          s: Math.min(prev.s, c),
          lane,
          lateral: lane,
          lateralVel: 0,
          laneFrom: lane,
          laneProgress: 1,
        };
      }
      return {
        s: START_FRACTIONS[i % START_FRACTIONS.length] * c,
        v: params.v0,
        a: 0,
        lane: laneFor(i),
        lateral: laneFor(i),
        lateralVel: 0,
        laneFrom: laneFor(i),
        laneProgress: 1,
        cooldown: 0,
      };
    });
    this.carParams = [...this.gui.settings.cars];
  }

  /** Spawns a car in the scene's best safe slot (falls back to the start; the Add button prevents this). */
  private spawnCar(params: IdmParams): Car {
    const slot =
      this.def.findSpawnSlot(this.cars, this.carParams, params, this.gui.settings.carLength) ?? {
        s: 0,
        lane: 0,
      };
    return {
      s: slot.s,
      v: params.v0,
      a: 0,
      lane: slot.lane,
      lateral: slot.lane,
      lateralVel: 0,
      laneFrom: slot.lane,
      laneProgress: 1,
      cooldown: 0,
    };
  }

  /** Reconciles the sim cars with the GUI's param list after an add/delete (matched by object identity). */
  private syncCars(): void {
    const paramsList = this.gui.settings.cars.slice(0, MAX_CARS);
    if (paramsList.length === this.cars.length) return;
    const used = new Set<number>();
    this.cars = paramsList.map((params) => {
      const idx = this.carParams.findIndex((p, i) => p === params && !used.has(i));
      if (idx >= 0) {
        used.add(idx);
        return this.cars[idx];
      }
      return this.spawnCar(params);
    });
    this.carParams = [...paramsList];
  }

  private readonly render = (now: number): void => {
    if (this.gui.settings.scene !== this.scene) this.applyScene(this.gui.settings.scene);
    // Scene 3's road is user-configurable: rebuild it on any change, keeping car state.
    if (this.scene === 3) {
      const key = JSON.stringify(this.gui.settings.scene3);
      if (key !== this.roadKey) this.applyScene(3, true);
    }

    const resized = resizeCanvasToDisplaySize(this.canvas, this.device.limits.maxTextureDimension2D);
    if (resized || !this.configured) {
      this.context.configure({
        device: this.device,
        format: this.format,
        alphaMode: 'opaque',
      });
      this.configured = true;
    }
    if (resized || !this.depthTexture) {
      this.depthTexture?.destroy();
      this.depthTexture = this.device.createTexture({
        label: 'depth',
        size: [this.canvas.width, this.canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    // Fixed-step simulation so IDM behavior is frame-rate independent.
    const dt = Math.min((now - (this.lastTime ?? now)) / 1000, 0.25);
    this.lastTime = now;
    this.syncCars();
    this.gui.canSpawn =
      this.def.findSpawnSlot(this.cars, this.carParams, NEW_CAR_PARAMS, this.gui.settings.carLength) !==
      null;
    this.accumulator = Math.min(this.accumulator + dt * this.gui.settings.timeScale, 1);
    while (this.accumulator >= SIM_STEP) {
      this.lightClock += SIM_STEP;
      // Yellow brakes like red: stop if you can.
      const clear = this.currentPhase() === 'green';
      this.def.step(
        this.cars,
        this.gui.settings.cars,
        this.gui.settings.carLength,
        SIM_STEP,
        clear ? NO_OBSTACLES : this.def.obstacles,
      );
      this.accumulator -= SIM_STEP;
    }
    const phase = this.currentPhase();
    this.gui.telemetry.light = phase;

    // Vehicle length is baked into the car mesh; rebuild it in place when the slider moves.
    if (this.gui.settings.carLength !== this.builtCarLength) {
      this.builtCarLength = this.gui.settings.carLength;
      this.carVerts = buildCarMesh(this.builtCarLength);
      this.device.queue.writeBuffer(this.vertexBuffer, 0, new Float32Array(this.carVerts));
    }

    // Ground/asphalt colors are baked into the static mesh; rebuild it on a day/night switch.
    const palette = this.palette();
    if (this.gui.settings.dayMode !== this.builtDayMode) {
      this.builtDayMode = this.gui.settings.dayMode;
      this.device.queue.writeBuffer(
        this.vertexBuffer,
        this.staticFirstVertex * 9 * Float32Array.BYTES_PER_ELEMENT,
        new Float32Array(this.def.buildStatic(palette)),
      );
    }
    this.cars.forEach((car, i) => {
      const gap = this.def.leaderGap(this.cars, i, this.gui.settings.carLength);
      this.gui.telemetry.speeds[String(i)] =
        gap === null
          ? `${car.v.toFixed(1)} m/s, free road`
          : `${car.v.toFixed(1)} m/s, gap ${gap.toFixed(1)} m`;
    });

    const viewProj = multiply(
      perspective((42 * Math.PI) / 180, this.canvas.width / this.canvas.height, 0.5, 600),
      lookAt(this.orbit.eye(), [0, 0, 0], [0, 1, 0]),
    );
    this.device.queue.writeBuffer(this.cameraBuffer, 0, viewProj);
    this.device.queue.writeBuffer(
      this.cameraBuffer,
      64,
      new Float32Array([palette.ambient, palette.diffuse, 0, 0]),
    );

    // Uniform slot 0: static scene. Slots 1..N: the cars. Slots LAMP_SLOT..: the signal lamps.
    const drawData = new Float32Array(FLOATS_PER_DRAW * TOTAL_SLOTS);
    drawData.set(identity(), 0);
    drawData.set([1, 1, 1, 1], 16);
    this.cars.forEach((car, i) => {
      const pose = this.def.carPose(car);
      const offset = FLOATS_PER_DRAW * (i + 1);
      drawData.set(
        multiply(
          translationRotationY(pose.x, pose.y, pose.z, pose.angle),
          rotationZ(pose.pitch),
        ),
        offset,
      );
      // White cars; yellow while changing lanes; the selected car is red.
      const rgb: Vec3 =
        i === this.gui.selectedCar
          ? [0.85, 0.27, 0.3]
          : car.laneProgress < 1
            ? [0.95, 0.8, 0.2]
            : [0.9, 0.9, 0.9];
      drawData.set([...rgb, 1], offset + 16);
    });
    this.def.lamps.forEach((pos, li) => {
      (['red', 'yellow', 'green'] as const).forEach((lamp, ci) => {
        const offset = FLOATS_PER_DRAW * (LAMP_SLOT + li * 3 + ci);
        const scale = phase === lamp ? 1 : INACTIVE_LAMP_DIM;
        drawData.set(translationRotationY(pos.x, LAMP_HEIGHTS[lamp], pos.z, 0), offset);
        drawData.set([...LAMP_COLORS[lamp].map((c) => c * scale), 1] as number[], offset + 16);
      });
    });
    this.device.queue.writeBuffer(this.drawBuffer, 0, drawData);

    const encoder = this.device.createCommandEncoder({ label: 'frame encoder' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: palette.sky[0], g: palette.sky[1], b: palette.sky[2], a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });
    pass.setPipeline(this.pipeline);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setBindGroup(0, this.bindGroup, [0]);
    pass.draw(this.staticVertexCount, 1, this.staticFirstVertex);
    this.cars.forEach((_, i) => {
      pass.setBindGroup(0, this.bindGroup, [DRAW_STRIDE * (i + 1)]);
      pass.draw(this.carVertexCount, 1, 0);
    });
    for (let i = 0; i < this.def.lamps.length * 3; i++) {
      pass.setBindGroup(0, this.bindGroup, [DRAW_STRIDE * (LAMP_SLOT + i)]);
      pass.draw(this.lampVertexCount, 1, this.lampFirstVertex);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    this.animationFrame = requestAnimationFrame(this.render);
  };
}
