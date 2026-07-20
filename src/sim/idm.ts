/**
 * Intelligent Driver Model (IDM) on a circular single-lane track.
 * https://en.wikipedia.org/wiki/Intelligent_driver_model
 */

/** IDM parameters for one car. */
export interface IdmParams {
  v0: number; // desired velocity (m/s)
  T: number; // desired time headway to the leader (s)
  a: number; // maximum acceleration (m/s²)
  b: number; // comfortable deceleration (m/s², positive value)
  s0: number; // minimum bumper-to-bumper gap in a jam (m)
  delta: number; // acceleration exponent (4 is the standard value)
}

export interface Car {
  s: number; // arc position along the ring (m), in [0, circumference)
  v: number; // current speed (m/s)
  a: number; // last computed acceleration (m/s²)
}

const MIN_GAP = 0.1; // floor for the gap so the interaction term cannot divide by zero

/** IDM acceleration for a car at speed `v`, `gap` behind its leader, closing at `closingSpeed` (v - vLeader). */
export function idmAcceleration(
  v: number,
  gap: number,
  closingSpeed: number,
  p: IdmParams,
): number {
  const s = Math.max(gap, MIN_GAP);
  const sStar = Math.max(p.s0, p.s0 + v * p.T + (v * closingSpeed) / (2 * Math.sqrt(p.a * p.b)));
  return p.a * (1 - (v / p.v0) ** p.delta - (sStar / s) ** 2);
}

/** A virtual standing vehicle on the ring, e.g. the stop line of a red light. */
export interface Obstacle {
  s: number; // arc position (m)
}

/**
 * Advances every car on the ring by one fixed step (semi-implicit Euler).
 * Each car follows the nearest car ahead of it and brakes for any obstacles,
 * whichever constraint is strongest.
 */
export function stepRing(
  cars: Car[],
  params: IdmParams[],
  circumference: number,
  carLength: number,
  dt: number,
  obstacles: Obstacle[] = [],
): void {
  // ponytail: O(n²) leader search, trivial for a handful of cars; sort by arc position if the car count grows large.
  const accels = cars.map((car, i) => {
    let accel = idmAcceleration(car.v, 1e6, 0, params[i]); // free road
    let gap = Infinity;
    let vLeader = 0;
    for (let j = 0; j < cars.length; j++) {
      if (j === i) continue;
      const d = (((cars[j].s - car.s) % circumference) + circumference) % circumference;
      if (d < gap) {
        gap = d;
        vLeader = cars[j].v;
      }
    }
    if (Number.isFinite(gap)) {
      accel = Math.min(accel, idmAcceleration(car.v, gap - carLength, car.v - vLeader, params[i]));
    }
    for (const obstacle of obstacles) {
      const d = (((obstacle.s - car.s) % circumference) + circumference) % circumference;
      // The car's front bumper stops at the obstacle: subtract its own half length.
      accel = Math.min(accel, idmAcceleration(car.v, d - carLength / 2, car.v, params[i]));
    }
    return accel;
  });

  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    car.a = accels[i];
    car.v = Math.max(0, car.v + car.a * dt);
    car.s = (car.s + car.v * dt) % circumference;
  }
}
