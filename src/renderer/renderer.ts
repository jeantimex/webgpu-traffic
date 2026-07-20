import type { GuiState } from '../gui/settings_gui';
import { stepRing, type Car } from '../sim/idm';
import { identity, lookAt, multiply, perspective, translationRotationY } from '../utils/mat4';
import { createBufferWithData, resizeCanvasToDisplaySize, type WebGPUState } from '../webgpu/utils';

const trafficShader = /* wgsl */ `
  struct Camera {
    viewProj: mat4x4f,
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
    return vec4f(input.color * (0.35 + 0.65 * diffuse), 1);
  }
`;

const TRACK_RADIUS = 40;
const ROAD_HALF_WIDTH = 4;
const CIRCUMFERENCE = 2 * Math.PI * TRACK_RADIUS;
const SIM_STEP = 1 / 60;
/** Byte stride between per-draw uniform slots (WebGPU dynamic-offset alignment). */
const DRAW_STRIDE = 256;
const FLOATS_PER_DRAW = DRAW_STRIDE / Float32Array.BYTES_PER_ELEMENT;
const CAR_TINTS: [number, number, number][] = [
  [0.85, 0.27, 0.3], // red
  [0.3, 0.5, 0.95], // blue
];

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

/** Ground plane + ring road, vertex colors baked in. */
function buildStaticMesh(): number[] {
  const verts: number[] = [];
  const G = 300;
  pushQuad(verts, [[-G, 0, -G], [-G, 0, G], [G, 0, G], [G, 0, -G]], [0.1, 0.12, 0.1]);

  const asphalt: Vec3 = [0.24, 0.25, 0.27];
  const inner = TRACK_RADIUS - ROAD_HALF_WIDTH;
  const outer = TRACK_RADIUS + ROAD_HALF_WIDTH;
  const SEGMENTS = 128;
  for (let i = 0; i < SEGMENTS; i++) {
    const t0 = (i / SEGMENTS) * 2 * Math.PI;
    const t1 = ((i + 1) / SEGMENTS) * 2 * Math.PI;
    pushQuad(
      verts,
      [
        [inner * Math.cos(t0), 0.02, inner * Math.sin(t0)],
        [inner * Math.cos(t1), 0.02, inner * Math.sin(t1)],
        [outer * Math.cos(t1), 0.02, outer * Math.sin(t1)],
        [outer * Math.cos(t0), 0.02, outer * Math.sin(t0)],
      ],
      asphalt,
    );
  }
  return verts;
}

/** A car-shaped box, white so the per-draw tint shows through. Bottom omitted: the camera stays above. */
function buildCarMesh(carLength: number): number[] {
  const verts: number[] = [];
  const white: Vec3 = [1, 1, 1];
  const x = carLength / 2;
  const z = 1;
  const y = 1.5;
  pushQuad(verts, [[-x, y, -z], [-x, y, z], [x, y, z], [x, y, -z]], white); // top
  pushQuad(verts, [[x, 0, z], [x, 0, -z], [x, y, -z], [x, y, z]], white); // front
  pushQuad(verts, [[-x, 0, -z], [-x, 0, z], [-x, y, z], [-x, y, -z]], white); // back
  pushQuad(verts, [[-x, 0, z], [x, 0, z], [x, y, z], [-x, y, z]], white); // left
  pushQuad(verts, [[x, 0, -z], [-x, 0, -z], [-x, y, -z], [x, y, -z]], white); // right
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
  private readonly cars: Car[];
  private builtCarLength: number;
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

    const staticVerts = buildStaticMesh();
    const carVerts = buildCarMesh(this.gui.settings.carLength);
    this.staticVertexCount = staticVerts.length / 9;
    this.carFirstVertex = this.staticVertexCount;
    this.carVertexCount = carVerts.length / 9;
    this.vertexBuffer = createBufferWithData(
      this.device,
      'scene vertices',
      new Float32Array([...staticVerts, ...carVerts]),
      GPUBufferUsage.VERTEX,
    );

    this.cameraBuffer = this.device.createBuffer({
      label: 'camera uniforms',
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.drawBuffer = this.device.createBuffer({
      label: 'per-draw uniforms',
      size: DRAW_STRIDE * 3,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
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

    // Cars start half a lap apart, each at its desired speed.
    this.cars = [
      { s: 0, v: this.gui.settings.cars[0].v0, a: 0 },
      { s: CIRCUMFERENCE / 2, v: this.gui.settings.cars[1].v0, a: 0 },
    ];
    this.builtCarLength = this.gui.settings.carLength;
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
    this.accumulator = Math.min(this.accumulator + dt * this.gui.settings.timeScale, 1);
    while (this.accumulator >= SIM_STEP) {
      stepRing(this.cars, this.gui.settings.cars, CIRCUMFERENCE, this.gui.settings.carLength, SIM_STEP);
      this.accumulator -= SIM_STEP;
    }

    // Vehicle length is baked into the car mesh; rebuild it in place when the slider moves.
    if (this.gui.settings.carLength !== this.builtCarLength) {
      this.builtCarLength = this.gui.settings.carLength;
      this.device.queue.writeBuffer(
        this.vertexBuffer,
        this.carFirstVertex * 9 * Float32Array.BYTES_PER_ELEMENT,
        new Float32Array(buildCarMesh(this.builtCarLength)),
      );
    }
    const carLength = this.gui.settings.carLength;
    const gapA =
      ((((this.cars[1].s - this.cars[0].s) % CIRCUMFERENCE) + CIRCUMFERENCE) % CIRCUMFERENCE) -
      carLength;
    this.gui.telemetry.carA = `${this.cars[0].v.toFixed(1)} m/s, gap ${gapA.toFixed(1)} m`;
    this.gui.telemetry.carB = `${this.cars[1].v.toFixed(1)} m/s, gap ${(CIRCUMFERENCE - gapA - 2 * carLength).toFixed(1)} m`;

    const viewProj = multiply(
      perspective((42 * Math.PI) / 180, this.canvas.width / this.canvas.height, 0.5, 600),
      lookAt([0, 110, 110], [0, 0, 0], [0, 1, 0]),
    );
    this.device.queue.writeBuffer(this.cameraBuffer, 0, viewProj);

    // Uniform slot 0: static track. Slots 1-2: the two cars.
    const drawData = new Float32Array(FLOATS_PER_DRAW * 3);
    drawData.set(identity(), 0);
    drawData.set([1, 1, 1, 1], 16);
    this.cars.forEach((car, i) => {
      const theta = car.s / TRACK_RADIUS;
      const offset = FLOATS_PER_DRAW * (i + 1);
      drawData.set(
        translationRotationY(
          TRACK_RADIUS * Math.cos(theta),
          0.02,
          TRACK_RADIUS * Math.sin(theta),
          -theta - Math.PI / 2,
        ),
        offset,
      );
      drawData.set([...CAR_TINTS[i], 1], offset + 16);
    });
    this.device.queue.writeBuffer(this.drawBuffer, 0, drawData);

    const encoder = this.device.createCommandEncoder({ label: 'frame encoder' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.06, b: 0.09, a: 1 },
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
    pass.setBindGroup(0, this.bindGroup, [DRAW_STRIDE]);
    pass.draw(this.carVertexCount, 1, this.carFirstVertex);
    pass.setBindGroup(0, this.bindGroup, [DRAW_STRIDE * 2]);
    pass.draw(this.carVertexCount, 1, this.carFirstVertex);
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    this.animationFrame = requestAnimationFrame(this.render);
  };
}
