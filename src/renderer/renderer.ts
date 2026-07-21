import { MAX_CARS, NEW_CAR_PARAMS, type GuiState } from '../gui/settings_gui';
import { type Car, type IdmParams } from '../sim/idm';
import { locate } from '../sim/network';
import { identity, lookAt, multiply, perspective, rotationZ, translationRotationY } from '../utils/mat4';
import { OrbitCamera } from '../utils/orbit';
import { createBufferWithData, resizeCanvasToDisplaySize, type WebGPUState } from '../webgpu/utils';
import {
  buildScene3,
  buildScene4,
  PALETTES,
  SCENES,
  pushBox,
  scene3State,
  scene4State,
  type Palette,
  type SceneDef,
  type Vec3,
} from './scenes';

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

/** Red on top, yellow in the middle, green at the bottom. */
const LAMP_HEIGHTS = { red: 3.2, yellow: 2.4, green: 1.6 } as const;
const LAMP_COLORS: Record<LightPhase, Vec3> = {
  red: [0.95, 0.15, 0.15],
  yellow: [0.95, 0.75, 0.1],
  green: [0.1, 0.85, 0.3],
};
const INACTIVE_LAMP_DIM = 0.25;

type LightPhase = 'red' | 'yellow' | 'green';

/** A car-shaped box, white so the per-draw tint shows through. */
function buildCarMesh(carLength: number): number[] {
  const verts: number[] = [];
  pushBox(verts, [-carLength / 2, 0, -1], [carLength / 2, 1.5, 1], [1, 1, 1]);
  return verts;
}

/** Random exit choice among the routes a lane actually has: mostly straight, sometimes a turn. */
function randomRoute(available: number[]): number {
  if (available.includes(0) && Math.random() < 0.6) return 0;
  const turns = available.filter((r) => r !== 0);
  return turns.length > 0 ? turns[Math.floor(Math.random() * turns.length)] : 0;
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
      buildScene3(this.gui.settings.scene3, this.gui.settings.scene3B, this.handed());
      this.roadKey = JSON.stringify([this.gui.settings.scene3, this.gui.settings.scene3B, this.gui.settings.trafficSide]);
    } else if (this.scene === 4) {
      buildScene4(this.gui.settings.scene4, this.handed());
      this.roadKey = JSON.stringify([this.gui.settings.scene4, this.gui.settings.trafficSide]);
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

    this.resetCars();
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

  private handed(): number {
    return this.gui.settings.trafficSide === 'right' ? 1 : -1;
  }

  /** Switches the active scene: rebuilds the static mesh and restarts traffic. */
  private applyScene(scene: number): void {
    this.scene = scene;
    if (scene === 3) {
      buildScene3(this.gui.settings.scene3, this.gui.settings.scene3B, this.handed());
      this.roadKey = JSON.stringify([this.gui.settings.scene3, this.gui.settings.scene3B, this.gui.settings.trafficSide]);
    } else if (scene === 4) {
      buildScene4(this.gui.settings.scene4, this.handed());
      this.roadKey = JSON.stringify([this.gui.settings.scene4, this.gui.settings.trafficSide]);
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
    this.resetCars();
    this.lightClock = 0;
  }

  /** Last-resort spawn slot: the first lane entrance that is not closed. */
  private fallbackSlot(): { s: number; lane: number } {
    const net = this.scene === 3 ? scene3State.net : this.scene === 4 ? (scene4State.state?.net ?? null) : null;
    if (net) {
      for (let g = 0; g < net.numLanes; g++) {
        if (net.closedLanes.has(g)) continue;
        const { road: ri, lane: li } = locate(net, g);
        const road = net.roads[ri];
        return { s: road.lanes[li].direction > 0 ? 0 : road.length, lane: g };
      }
    }
    return { s: 0, lane: 0 };
  }

  /**
   * Places all cars for the current scene. Ring scenes deal fixed pairs per lane;
   * scene 3 deals round-robin; scene 4 places each car through the safe-slot search
   * so nothing ever spawns on a closed lane.
   */
  private resetCars(): void {
    const c = this.def.c;
    const net = this.scene === 3 ? scene3State.net : this.scene === 4 ? (scene4State.state?.net ?? null) : null;
    const numLanes = net ? net.numLanes : 2;
    const laneFor = (i: number): number =>
      net ? i % numLanes : START_LANES[i % START_LANES.length] % numLanes;
    if (this.scene === 4 && net) {
      const cars: Car[] = [];
      const carParams: IdmParams[] = [];
      this.gui.settings.cars.forEach((params) => {
        const slot =
          this.def.findSpawnSlot(cars, carParams, params, this.gui.settings.carLength) ??
          this.fallbackSlot();
        carParams.push(params);
        cars.push({
          s: slot.s,
          v: params.v0,
          a: 0,
          lane: slot.lane,
          route: randomRoute(this.availableRoutes(slot.lane)),
          lateral: slot.lane,
          lateralVel: 0,
          laneFrom: slot.lane,
          laneProgress: 1,
          cooldown: 0,
        });
      });
      this.cars = cars;
      this.carParams = carParams;
      return;
    }
    this.cars = this.gui.settings.cars.map((params, i) => {
      const lane = laneFor(i);
      const span = net ? net.roads[locate(net, lane).road].length : c;
      return {
        s: START_FRACTIONS[i % START_FRACTIONS.length] * span,
        v: params.v0,
        a: 0,
        lane,
        route: 0,
        lateral: lane,
        lateralVel: 0,
        laneFrom: lane,
        laneProgress: 1,
        cooldown: 0,
      };
    });
    this.carParams = [...this.gui.settings.cars];
  }

  /** Spawns a car in the scene's best safe slot (falls back to the first open lane entrance). */
  private spawnCar(params: IdmParams): Car {
    const slot =
      this.def.findSpawnSlot(this.cars, this.carParams, params, this.gui.settings.carLength) ??
      this.fallbackSlot();
    return {
      s: slot.s,
      v: params.v0,
      a: 0,
      lane: slot.lane,
      route: this.scene === 4 ? randomRoute(this.availableRoutes(slot.lane)) : 0,
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

  /** Route indices available on a lane (scene 4); [0] elsewhere. */
  private availableRoutes(lane: number): number[] {
    if (this.scene !== 4 || !scene4State.state) return [0];
    const net = scene4State.state.net;
    const { road: ri, lane: li } = locate(net, lane);
    return net.exit[ri][li].flatMap((conn, i) => (conn ? [i] : []));
  }

  /** Cars stopped at an unconnected end "left the map": respawn them at a safe entrance. */
  private recycleCars(): void {
    const net = this.scene === 3 ? scene3State.net : this.scene === 4 ? (scene4State.state?.net ?? null) : null;
    if (!net) return;
    this.cars.forEach((car, i) => {
      const { road: ri, lane: li } = locate(net, car.lane);
      if (net.exit[ri][li].length > 0) return; // connected end: keeps flowing
      const road = net.roads[ri];
      const dir = road.lanes[li].direction;
      const distToEnd = dir > 0 ? road.length - car.s : car.s;
      if (distToEnd > 3 || car.v > 0.5) return;
      const slot = this.def.findSpawnSlot(
        this.cars,
        this.carParams,
        this.carParams[i],
        this.gui.settings.carLength,
      );
      if (!slot) return;
      car.s = slot.s;
      car.lane = slot.lane;
      car.lateral = slot.lane;
      car.laneFrom = slot.lane;
      car.laneProgress = 1;
      car.lateralVel = 0;
      car.v = this.carParams[i].v0;
      car.a = 0;
      car.cooldown = 0;
      car.route = this.scene === 4 ? randomRoute(this.availableRoutes(slot.lane)) : 0;
    });
  }

  private readonly render = (now: number): void => {
    if (this.gui.settings.scene !== this.scene) this.applyScene(this.gui.settings.scene);
    // Network scenes are user-configurable: any change clears traffic and re-deals fresh cars.
    if (this.scene === 3 || this.scene === 4) {
      const key =
        this.scene === 3
          ? JSON.stringify([this.gui.settings.scene3, this.gui.settings.scene3B, this.gui.settings.trafficSide])
          : JSON.stringify([this.gui.settings.scene4, this.gui.settings.trafficSide]);
      if (key !== this.roadKey) this.applyScene(this.scene);
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
    let phase = '';
    while (this.accumulator >= SIM_STEP) {
      this.lightClock += SIM_STEP;
      phase = this.def.phaseAt(this.lightClock, this.gui.settings.light);
      this.def.step(
        this.cars,
        this.gui.settings.cars,
        this.gui.settings.carLength,
        SIM_STEP,
        this.def.obstaclesFor(phase),
      );
      this.accumulator -= SIM_STEP;
    }
    phase = this.def.phaseAt(this.lightClock, this.gui.settings.light);
    this.gui.telemetry.light = phase;
    this.recycleCars();

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
      const route = this.scene === 4 ? ` ${['S', 'R', 'L'][car.route]}` : '';
      this.gui.telemetry.speeds[String(i)] =
        gap === null
          ? `${car.v.toFixed(1)} m/s${route}, free road`
          : `${car.v.toFixed(1)} m/s${route}, gap ${gap.toFixed(1)} m`;
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
      const lit = this.def.lampColor(li, phase);
      (['red', 'yellow', 'green'] as const).forEach((lamp, ci) => {
        const offset = FLOATS_PER_DRAW * (LAMP_SLOT + li * 3 + ci);
        const scale = lit === lamp ? 1 : INACTIVE_LAMP_DIM;
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
