/**
 * Intelligent Driver Model (IDM) on a circular two-lane track.
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
  lane: number; // logical lane: 0 = inner, 1 = outer
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
const LANE_CHANGE_COOLDOWN = 4; // s between one car's lane changes (prevents weaving)
export const B_SAFE = 4; // m/s², the most braking a lane change (or spawn) may impose on anyone
const DELTA_A = 0.2; // m/s², minimum advantage that makes a change worthwhile
const KEEP_RIGHT_GAP = 60; // m, "inner lane is free ahead" threshold for drifting back

interface LaneNeighbor {
  index: number;
  gap: number; // bumper-to-lead-position distance (m)
  v: number;
}

/** Nearest car in `lane` ahead of (or behind) car `me`, measured along the ring. */
function nearestInLane(
  cars: Car[],
  me: number,
  lane: number,
  circumference: number,
  ahead: boolean,
): LaneNeighbor | null {
  let best: LaneNeighbor | null = null;
  for (let j = 0; j < cars.length; j++) {
    if (j === me || cars[j].lane !== lane) continue;
    const d = (((cars[j].s - cars[me].s) % circumference) + circumference) % circumference;
    const gap = ahead ? d : (circumference - d) % circumference;
    if (best === null || gap < best.gap) best = { index: j, gap, v: cars[j].v };
  }
  return best;
}

function accelToward(car: Car, leader: LaneNeighbor | null, carLength: number, p: IdmParams): number {
  return leader
    ? idmAcceleration(car.v, leader.gap - carLength, car.v - leader.v, p)
    : idmAcceleration(car.v, 1e6, 0, p);
}

/**
 * MOBIL-lite: a car changes lanes when the target lane is clearly better (overtake /
 * avoid a braking leader) — but only if neither it nor the target-lane follower has
 * to brake harder than B_SAFE. Cars also drift back to the inner lane when it's free
 * ahead. Obstacles (the red light) never trigger lane changes.
 */
function updateLanes(cars: Car[], params: IdmParams[], circumference: number, carLength: number): void {
  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    if (car.cooldown > 0) continue;
    const target = 1 - car.lane;
    const accelHere = accelToward(car, nearestInLane(cars, i, car.lane, circumference, true), carLength, params[i]);
    const accelThere = accelToward(car, nearestInLane(cars, i, target, circumference, true), carLength, params[i]);
    const keepRight = target === 0 && accelThere >= accelHere - DELTA_A &&
      (nearestInLane(cars, i, 0, circumference, true)?.gap ?? Infinity) > KEEP_RIGHT_GAP;
    if (accelThere - accelHere < DELTA_A && !keepRight) continue;
    // Safety first: no hard braking for me or for the car behind me in the target lane.
    if (accelThere < -B_SAFE) continue;
    const follower = nearestInLane(cars, i, target, circumference, false);
    if (follower) {
      const followerAccel = idmAcceleration(
        follower.v,
        follower.gap - carLength,
        follower.v - car.v,
        params[follower.index],
      );
      if (followerAccel < -B_SAFE) continue;
    }
    car.lane = target;
    car.cooldown = LANE_CHANGE_COOLDOWN;
    car.laneFrom = car.lateral;
    car.laneProgress = 0;
  }
}

/**
 * Advances every car on the ring by one fixed step (semi-implicit Euler).
 * Each car follows the nearest car ahead of it in its lane and brakes for any
 * obstacles, whichever constraint is strongest. Includes MOBIL-lite lane changes.
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
      if (j === i || cars[j].lane !== car.lane) continue;
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

  updateLanes(cars, params, circumference, carLength);

  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    car.a = accels[i];
    car.v = Math.max(0, car.v + car.a * dt);
    car.s = (car.s + car.v * dt) % circumference;
    car.cooldown = Math.max(0, car.cooldown - dt);
    // Cosine-eased lateral slide: zero lateral velocity at both ends, so the car's
    // path is an S-curve that joins the target lane without a heading jerk.
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
}
