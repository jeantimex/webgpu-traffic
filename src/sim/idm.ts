/**
 * Intelligent Driver Model (IDM) primitives.
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
  lane: number; // logical lane: 0 = inner, 1 = outer (global lane index in networks)
  route: number; // exit choice at connections: 0 = straight, 1 = right, 2 = left
  lateral: number; // visual lane position in lane units (0 = inner, 1 = outer), eases toward `lane`
  lateralVel: number; // lateral velocity (lane units/s) during a lane change, 0 otherwise
  laneFrom: number; // lateral position where the current lane change started
  laneProgress: number; // lane-change progress in [0, 1]; 1 = settled in lane
  cooldown: number; // seconds before this car may change lanes again
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

// Lane-change tuning (MOBIL-lite).
const LANE_CHANGE_TIME = 2; // s for the lateral slide
export const LANE_CHANGE_COOLDOWN = 4; // s between one car's lane changes (prevents weaving)
export const B_SAFE = 4; // m/s², the most braking a lane change (or spawn) may impose on anyone
export const DELTA_A = 0.2; // m/s², minimum advantage that makes a change worthwhile

/** Advances one car's cooldown and cosine-eased lane-change slide by dt (S-curve, zero jerk at both ends). */
export function advanceLateral(car: Car, dt: number): void {
  car.cooldown = Math.max(0, car.cooldown - dt);
  if (car.laneProgress < 1) {
    car.laneProgress = Math.min(1, car.laneProgress + dt / LANE_CHANGE_TIME);
    const p = car.laneProgress;
    car.lateral = car.laneFrom + (car.lane - car.laneFrom) * (0.5 - 0.5 * Math.cos(Math.PI * p));
    car.lateralVel =
      ((car.lane - car.laneFrom) * 0.5 * Math.PI * Math.sin(Math.PI * p)) / LANE_CHANGE_TIME;
    if (p === 1) car.lateralVel = 0;
  } else {
    car.lateralVel = 0;
  }
}
