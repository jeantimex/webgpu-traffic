import type { GuiState } from '../gui/settings_gui';
import { stepRing, type Car, type Obstacle } from '../sim/idm';
import { identity, lookAt, multiply, perspective, rotationZ, translationRotationY } from '../utils/mat4';
import { OrbitCamera } from '../utils/orbit';
import { createBufferWithData, resizeCanvasToDisplaySize, type WebGPUState } from '../webgpu/utils';

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

const TRACK_RADIUS = 40;
const ROAD_HALF_WIDTH = 4;
const CIRCUMFERENCE = 2 * Math.PI * TRACK_RADIUS;
/** Lane center radii: lane 0 = inner, lane 1 = outer (two 4 m lanes share the road). */
const LANE_RADIUS = [TRACK_RADIUS - 2, TRACK_RADIUS + 2];
const SIM_STEP = 1 / 60;
/** Byte stride between per-draw uniform slots (WebGPU dynamic-offset alignment). */
const DRAW_STRIDE = 256;
const FLOATS_PER_DRAW = DRAW_STRIDE / Float32Array.BYTES_PER_ELEMENT;
const CAR_TINTS: [number, number, number][] = [
  [0.85, 0.27, 0.3], // red
  [0.3, 0.5, 0.95], // blue
  [0.95, 0.8, 0.2], // yellow
  [0.9, 0.9, 0.9], // white
];
/** Start arc positions: each lane's pair begins half a lap apart. */
const START_S = [0, CIRCUMFERENCE / 2, CIRCUMFERENCE / 4, (3 * CIRCUMFERENCE) / 4];

interface Palette {
  sky: Vec3;
  ground: Vec3;
  asphalt: Vec3;
  wall: Vec3;
  ambient: number;
  diffuse: number;
}

const PALETTES: Record<'day' | 'night', Palette> = {
  day: {
    sky: [0.53, 0.75, 0.95],
    ground: [0.32, 0.47, 0.25],
    asphalt: [0.38, 0.39, 0.41],
    wall: [0.3, 0.3, 0.32],
    ambient: 0.55,
    diffuse: 0.7,
  },
  night: {
    sky: [0.05, 0.06, 0.09],
    ground: [0.1, 0.12, 0.1],
    asphalt: [0.24, 0.25, 0.27],
    wall: [0.17, 0.17, 0.19],
    ambient: 0.35,
    diffuse: 0.65,
  },
};
/** Arc position of the stop line / pedestrian crossing (quarter lap, nearest the camera). */
const STOP_S = CIRCUMFERENCE / 4;
const RED_LIGHT: Obstacle[] = [{ s: STOP_S }];
const NO_OBSTACLES: Obstacle[] = [];
/** The signal hangs just inside the inner road edge at the stop line. */
const LAMP_X = (TRACK_RADIUS - ROAD_HALF_WIDTH - 1.2) * Math.cos(STOP_S / TRACK_RADIUS);
const LAMP_Z = (TRACK_RADIUS - ROAD_HALF_WIDTH - 1.2) * Math.sin(STOP_S / TRACK_RADIUS);
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

/** The bridge is a raised-cosine bump on the far side of the ring (clear of the cars' start and the crossing). */
const BRIDGE_LENGTH = 60; // m along the arc
const BRIDGE_HEIGHT = 4; // m
const BRIDGE_START = (3 * CIRCUMFERENCE) / 4 - BRIDGE_LENGTH / 2;

/** Road surface height above the ground at arc position s (0 off the bridge). */
function roadHeight(s: number): number {
  if (s < BRIDGE_START || s > BRIDGE_START + BRIDGE_LENGTH) return 0;
  const u = (s - BRIDGE_START) / BRIDGE_LENGTH;
  return (BRIDGE_HEIGHT / 2) * (1 - Math.cos(2 * Math.PI * u));
}

/** Slope (dh/ds) of the road at arc position s. */
function roadGrade(s: number): number {
  if (s < BRIDGE_START || s > BRIDGE_START + BRIDGE_LENGTH) return 0;
  const u = (s - BRIDGE_START) / BRIDGE_LENGTH;
  return ((BRIDGE_HEIGHT * Math.PI) / BRIDGE_LENGTH) * Math.sin(2 * Math.PI * u);
}

type Vec3 = [number, number, number];

/** Appends a quad (6 vertices, interleaved position/normal/color). Corners must be CCW seen from outside. */
function pushQuad(out: number[], corners: [Vec3, Vec3, Vec3, Vec3], color: Vec3): void {
  const [a, b, c, d] = corners;
  const ux = b[0] - a[0];
  const uy = b[1] - a[1];
  const uz = b[2] - a[2];
  const vx = c[0] - a[0];
  const vy = c[1] - a[1];
  const vz = c[2] - a[2];
  const nx = uy * vz - uz * vy;
  const ny = uz * vx - ux * vz;
  const nz = ux * vy - uy * vx;
  const len = Math.hypot(nx, ny, nz);
  for (const p of [a, b, c, a, c, d]) {
    out.push(p[0], p[1], p[2], nx / len, ny / len, nz / len, color[0], color[1], color[2]);
  }
}

/** Appends a box (5 faces, bottom omitted: the camera stays above) to a vertex list. */
function pushBox(out: number[], min: Vec3, max: Vec3, color: Vec3): void {
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  pushQuad(out, [[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]], color); // top
  pushQuad(out, [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], color); // +x
  pushQuad(out, [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], color); // −x
  pushQuad(out, [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], color); // +z
  pushQuad(out, [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], color); // −z
}

/** Appends a rectangular road patch covering arc [s0, s1] and radius [r0, r1]. */
function pushRoadPatch(out: number[], s0: number, s1: number, r0: number, r1: number, color: Vec3): void {
  // ponytail: a single straight quad per patch; over a few meters of arc it sags ~4 cm off the
  // circle, invisible at this zoom. Segment along the arc if patches get much longer.
  const at = (s: number, r: number): Vec3 => {
    const theta = s / TRACK_RADIUS;
    return [r * Math.cos(theta), 0.03, r * Math.sin(theta)];
  };
  pushQuad(out, [at(s0, r0), at(s1, r0), at(s1, r1), at(s0, r1)], color);
}

/** Appends a lane-divider dash centered between the two lanes, following the road height. */
function pushLaneDash(out: number[], s0: number, s1: number, color: Vec3): void {
  const r0 = TRACK_RADIUS - 0.075;
  const r1 = TRACK_RADIUS + 0.075;
  const at = (s: number, r: number): Vec3 => {
    const theta = s / TRACK_RADIUS;
    return [r * Math.cos(theta), 0.03 + roadHeight(s), r * Math.sin(theta)];
  };
  pushQuad(out, [at(s0, r0), at(s1, r0), at(s1, r1), at(s0, r1)], color);
}

/** Ground plane + ring road + crossing paint + light pole, vertex colors baked in. */
function buildStaticMesh(palette: Palette): number[] {
  const verts: number[] = [];
  const G = 300;
  pushQuad(verts, [[-G, 0, -G], [-G, 0, G], [G, 0, G], [G, 0, -G]], palette.ground);

  const asphalt = palette.asphalt;
  const wall = palette.wall;
  const inner = TRACK_RADIUS - ROAD_HALF_WIDTH;
  const outer = TRACK_RADIUS + ROAD_HALF_WIDTH;
  const SEGMENTS = 128;
  for (let i = 0; i < SEGMENTS; i++) {
    const t0 = (i / SEGMENTS) * 2 * Math.PI;
    const t1 = ((i + 1) / SEGMENTS) * 2 * Math.PI;
    const h0 = 0.02 + roadHeight(t0 * TRACK_RADIUS);
    const h1 = 0.02 + roadHeight(t1 * TRACK_RADIUS);
    pushQuad(
      verts,
      [
        [inner * Math.cos(t0), h0, inner * Math.sin(t0)],
        [inner * Math.cos(t1), h1, inner * Math.sin(t1)],
        [outer * Math.cos(t1), h1, outer * Math.sin(t1)],
        [outer * Math.cos(t0), h0, outer * Math.sin(t0)],
      ],
      asphalt,
    );
    // Side walls under the elevated section so the bridge reads as solid.
    if (h0 > 0.05 || h1 > 0.05) {
      pushQuad(
        verts,
        [
          [outer * Math.cos(t0), 0, outer * Math.sin(t0)],
          [outer * Math.cos(t0), h0, outer * Math.sin(t0)],
          [outer * Math.cos(t1), h1, outer * Math.sin(t1)],
          [outer * Math.cos(t1), 0, outer * Math.sin(t1)],
        ],
        wall,
      );
      pushQuad(
        verts,
        [
          [inner * Math.cos(t0), h0, inner * Math.sin(t0)],
          [inner * Math.cos(t0), 0, inner * Math.sin(t0)],
          [inner * Math.cos(t1), 0, inner * Math.sin(t1)],
          [inner * Math.cos(t1), h1, inner * Math.sin(t1)],
        ],
        wall,
      );
    }
  }

  // Solid stop line across the lane, then a zebra crossing after it: stripes run
  // parallel to travel, packed across the lane width.
  const paint: Vec3 = [0.9, 0.9, 0.9];
  pushRoadPatch(verts, STOP_S - 0.125, STOP_S + 0.125, inner, outer, paint);
  for (let i = 0; i < 7; i++) {
    const r0 = inner + 0.6 + i * 1.0;
    pushRoadPatch(verts, STOP_S + 0.8, STOP_S + 4.3, r0, r0 + 0.5, paint);
  }

  // Dashed divider between the lanes, skipping the crossing.
  for (let s = 0; s < CIRCUMFERENCE; s += 6) {
    if (s > STOP_S - 2 && s < STOP_S + 6) continue;
    pushLaneDash(verts, s, s + 2, paint);
  }

  // Traffic-light pole beside the road, tall enough for the three lamps.
  pushBox(
    verts,
    [LAMP_X - 0.1, 0, LAMP_Z - 0.1],
    [LAMP_X + 0.1, 4, LAMP_Z + 0.1],
    [0.4, 0.4, 0.42],
  );
  return verts;
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
  private readonly vertexBuffer: GPUBuffer;
  private readonly cameraBuffer: GPUBuffer;
  private readonly drawBuffer: GPUBuffer;
  private readonly pipeline: GPURenderPipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly staticVertexCount: number;
  private readonly carFirstVertex: number;
  private readonly carVertexCount: number;
  private readonly lampFirstVertex: number;
  private readonly lampVertexCount: number;
  private readonly cars: Car[];
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

    const staticVerts = buildStaticMesh(this.palette());
    const carVerts = buildCarMesh(this.gui.settings.carLength);
    const lampVerts = buildLampMesh();
    this.staticVertexCount = staticVerts.length / 9;
    this.carFirstVertex = this.staticVertexCount;
    this.carVertexCount = carVerts.length / 9;
    this.lampFirstVertex = this.carFirstVertex + this.carVertexCount;
    this.lampVertexCount = lampVerts.length / 9;
    this.vertexBuffer = createBufferWithData(
      this.device,
      'scene vertices',
      new Float32Array([...staticVerts, ...carVerts, ...lampVerts]),
      GPUBufferUsage.VERTEX,
    );

    this.cameraBuffer = this.device.createBuffer({
      label: 'camera uniforms',
      size: 80, // mat4x4 viewProj + vec4 lighting
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.drawBuffer = this.device.createBuffer({
      label: 'per-draw uniforms',
      // 1 track + N cars + 3 lamps
      size: DRAW_STRIDE * (4 + this.gui.settings.cars.length),
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

    // Cars start at their desired speed, each lane's pair half a lap apart.
    this.cars = this.gui.settings.cars.map((params, i) => ({
      s: START_S[i % START_S.length],
      v: params.v0,
      a: 0,
      lane: this.gui.settings.carLanes[i].lane,
    }));
    this.builtCarLength = this.gui.settings.carLength;
    this.builtDayMode = this.gui.settings.dayMode;
    this.orbit = new OrbitCamera(canvas);
  }

  start(): void {
    if (this.animationFrame !== undefined) return;
    this.animationFrame = requestAnimationFrame(this.render);
  }

  private palette(): Palette {
    return PALETTES[this.gui.settings.dayMode ? 'day' : 'night'];
  }

  /** Bumper gap to the nearest car ahead in the same lane, or null when alone in the lane. */
  private leaderGap(i: number): number | null {
    const car = this.cars[i];
    let gap = Infinity;
    for (let j = 0; j < this.cars.length; j++) {
      if (j === i || this.cars[j].lane !== car.lane) continue;
      gap = Math.min(
        gap,
        (((this.cars[j].s - car.s) % CIRCUMFERENCE) + CIRCUMFERENCE) % CIRCUMFERENCE,
      );
    }
    return Number.isFinite(gap) ? gap - this.gui.settings.carLength : null;
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

  private readonly render = (now: number): void => {
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
    const { green, yellow, red } = this.gui.settings.light;
    this.accumulator = Math.min(this.accumulator + dt * this.gui.settings.timeScale, 1);
    this.cars.forEach((car, i) => {
      car.lane = this.gui.settings.carLanes[i].lane;
    });
    while (this.accumulator >= SIM_STEP) {
      this.lightClock += SIM_STEP;
      // Yellow brakes like red: stop if you can.
      const clear = lightPhase(this.lightClock, green, yellow, red) === 'green';
      stepRing(
        this.cars,
        this.gui.settings.cars,
        CIRCUMFERENCE,
        this.gui.settings.carLength,
        SIM_STEP,
        clear ? NO_OBSTACLES : RED_LIGHT,
      );
      this.accumulator -= SIM_STEP;
    }
    const phase = lightPhase(this.lightClock, green, yellow, red);
    this.gui.telemetry.light = phase;

    // Vehicle length is baked into the car mesh; rebuild it in place when the slider moves.
    if (this.gui.settings.carLength !== this.builtCarLength) {
      this.builtCarLength = this.gui.settings.carLength;
      this.device.queue.writeBuffer(
        this.vertexBuffer,
        this.carFirstVertex * 9 * Float32Array.BYTES_PER_ELEMENT,
        new Float32Array(buildCarMesh(this.builtCarLength)),
      );
    }

    // Ground/asphalt colors are baked into the static mesh; rebuild it on a day/night switch.
    const palette = this.palette();
    if (this.gui.settings.dayMode !== this.builtDayMode) {
      this.builtDayMode = this.gui.settings.dayMode;
      this.device.queue.writeBuffer(this.vertexBuffer, 0, new Float32Array(buildStaticMesh(palette)));
    }
    const keys = ['carA', 'carB', 'carC', 'carD'] as const;
    this.cars.forEach((car, i) => {
      const gap = this.leaderGap(i);
      this.gui.telemetry[keys[i]] =
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

    // Uniform slot 0: static track. Slots 1..N: the cars. Then 3 slots: red/yellow/green lamps.
    const drawData = new Float32Array(FLOATS_PER_DRAW * (4 + this.cars.length));
    drawData.set(identity(), 0);
    drawData.set([1, 1, 1, 1], 16);
    this.cars.forEach((car, i) => {
      const theta = car.s / TRACK_RADIUS;
      const r = LANE_RADIUS[car.lane];
      const offset = FLOATS_PER_DRAW * (i + 1);
      drawData.set(
        multiply(
          translationRotationY(
            r * Math.cos(theta),
            0.02 + roadHeight(car.s),
            r * Math.sin(theta),
            -theta - Math.PI / 2,
          ),
          rotationZ(Math.atan(roadGrade(car.s))),
        ),
        offset,
      );
      drawData.set([...CAR_TINTS[i], 1], offset + 16);
    });
    (['red', 'yellow', 'green'] as const).forEach((lamp, i) => {
      const offset = FLOATS_PER_DRAW * (i + 1 + this.cars.length);
      const scale = phase === lamp ? 1 : INACTIVE_LAMP_DIM;
      drawData.set(translationRotationY(LAMP_X, LAMP_HEIGHTS[lamp], LAMP_Z, 0), offset);
      drawData.set([...LAMP_COLORS[lamp].map((c) => c * scale), 1] as number[], offset + 16);
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
    pass.draw(this.staticVertexCount);
    this.cars.forEach((_, i) => {
      pass.setBindGroup(0, this.bindGroup, [DRAW_STRIDE * (i + 1)]);
      pass.draw(this.carVertexCount, 1, this.carFirstVertex);
    });
    for (let i = 0; i < 3; i++) {
      pass.setBindGroup(0, this.bindGroup, [DRAW_STRIDE * (1 + this.cars.length + i)]);
      pass.draw(this.lampVertexCount, 1, this.lampFirstVertex);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    this.animationFrame = requestAnimationFrame(this.render);
  };
}
