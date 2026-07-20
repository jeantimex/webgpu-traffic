import type { Vec3 } from './mat4';

function clamp(x: number, min: number, max: number): number {
  return Math.min(Math.max(x, min), max);
}

/** Minimal orbit camera around the origin: drag to orbit, wheel to zoom. */
export class OrbitCamera {
  private azimuth = 0; // angle around the Y axis
  private polar = Math.PI / 4; // angle down from +Y
  private radius = 156;

  constructor(canvas: HTMLCanvasElement) {
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    canvas.addEventListener('pointerdown', (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointerup', (e) => {
      dragging = false;
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      this.azimuth -= (e.clientX - lastX) * 0.005;
      this.polar = clamp(this.polar - (e.clientY - lastY) * 0.005, 0.05, Math.PI / 2 - 0.05);
      lastX = e.clientX;
      lastY = e.clientY;
    });
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        this.radius = clamp(this.radius * (1 + e.deltaY * 0.001), 30, 500);
      },
      { passive: false },
    );
  }

  eye(): Vec3 {
    const horizontal = Math.sin(this.polar) * this.radius;
    return [
      horizontal * Math.sin(this.azimuth),
      Math.cos(this.polar) * this.radius,
      horizontal * Math.cos(this.azimuth),
    ];
  }
}
