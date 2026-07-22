import {
  advanceLateral,
  B_SAFE,
  DELTA_A,
  idmAcceleration,
  LANE_CHANGE_COOLDOWN,
  type Car,
  type IdmParams,
  type Obstacle,
} from './idm';
import { laneNode, type Network } from './network';

const KEEP_RIGHT_GAP = 60;

interface LoopNeighbor {
  index: number;
  gap: number;
  v: number;
}

function nearestLoopLane(
  cars: Car[],
  me: number,
  lane: number,
  circumference: number,
  ahead: boolean,
): LoopNeighbor | null {
  let best: LoopNeighbor | null = null;
  for (let j = 0; j < cars.length; j++) {
    if (j === me || cars[j].lane !== lane) continue;
    const d = (((cars[j].s - cars[me].s) % circumference) + circumference) % circumference;
    const gap = ahead ? d : (circumference - d) % circumference;
    if (best === null || gap < best.gap) best = { index: j, gap, v: cars[j].v };
  }
  return best;
}

function accelToward(car: Car, leader: LoopNeighbor | null, carLength: number, p: IdmParams): number {
  return leader
    ? idmAcceleration(car.v, leader.gap - carLength, car.v - leader.v, p)
    : idmAcceleration(car.v, 1e6, 0, p);
}

function updateLoopLanes(net: Network, cars: Car[], params: IdmParams[], circumference: number, carLength: number): void {
  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    if (car.cooldown > 0) continue;
    const accelHere = accelToward(
      car,
      nearestLoopLane(cars, i, car.lane, circumference, true),
      carLength,
      params[i],
    );
    let bestTarget = -1;
    for (const target of [laneNode(net, car.lane).leftNeighbor, laneNode(net, car.lane).rightNeighbor]) {
      if (target === null) continue;
      const accelThere = accelToward(
        car,
        nearestLoopLane(cars, i, target, circumference, true),
        carLength,
        params[i],
      );
      const keepRight =
        target === 0 &&
        accelThere >= accelHere - DELTA_A &&
        (nearestLoopLane(cars, i, 0, circumference, true)?.gap ?? Infinity) > KEEP_RIGHT_GAP;
      if (accelThere - accelHere < DELTA_A && !keepRight) continue;
      if (accelThere < -B_SAFE) continue;
      const follower = nearestLoopLane(cars, i, target, circumference, false);
      if (follower) {
        const followerAccel = idmAcceleration(
          follower.v,
          follower.gap - carLength,
          follower.v - car.v,
          params[follower.index],
        );
        if (followerAccel < -B_SAFE) continue;
      }
      bestTarget = target;
    }
    if (bestTarget >= 0) {
      car.lane = bestTarget;
      car.cooldown = LANE_CHANGE_COOLDOWN;
      car.laneFrom = car.lateral;
      car.laneProgress = 0;
    }
  }
}

export function stepLaneLoop(
  net: Network,
  cars: Car[],
  params: IdmParams[],
  circumference: number,
  carLength: number,
  dt: number,
  obstacles: Obstacle[] = [],
): void {
  const accels = cars.map((car, i) => {
    let accel = idmAcceleration(car.v, 1e6, 0, params[i]);
    const leader = nearestLoopLane(cars, i, car.lane, circumference, true);
    if (leader) accel = Math.min(accel, idmAcceleration(car.v, leader.gap - carLength, car.v - leader.v, params[i]));
    for (const obstacle of obstacles) {
      const d = (((obstacle.s - car.s) % circumference) + circumference) % circumference;
      accel = Math.min(accel, idmAcceleration(car.v, d - carLength / 2, car.v, params[i]));
    }
    return accel;
  });

  updateLoopLanes(net, cars, params, circumference, carLength);

  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    car.a = accels[i];
    car.v = Math.max(0, car.v + car.a * dt);
    car.s = (car.s + car.v * dt) % circumference;
    advanceLateral(car, dt);
  }
}

export function laneLoopGapAhead(
  cars: Car[],
  i: number,
  circumference: number,
  carLength: number,
): number | null {
  const leader = nearestLoopLane(cars, i, cars[i].lane, circumference, true);
  return leader ? leader.gap - carLength : null;
}

export function laneLoopSpawnSlot(
  net: Network,
  cars: Car[],
  carParams: IdmParams[],
  params: IdmParams,
  circumference: number,
  carLength: number,
): { s: number; lane: number } | null {
  let best: { s: number; lane: number } | null = null;
  let bestScore = -Infinity;
  for (const lane of net.lanes) {
    for (let k = 0; k < 32; k++) {
      const s = (k * circumference) / 32;
      let leaderGap = Infinity;
      let leaderV = params.v0;
      let followerGap = Infinity;
      let followerV = params.v0;
      let followerParams: IdmParams | null = null;
      cars.forEach((car, j) => {
        if (car.lane !== lane.global && car.laneProgress >= 1) return;
        const fwd = (((car.s - s) % circumference) + circumference) % circumference;
        if (fwd < leaderGap) {
          leaderGap = fwd;
          leaderV = car.v;
        }
        const back = (circumference - fwd) % circumference;
        if (back < followerGap) {
          followerGap = back;
          followerV = car.v;
          followerParams = carParams[j];
        }
      });
      if (idmAcceleration(params.v0, leaderGap - carLength, params.v0 - leaderV, params) < -B_SAFE)
        continue;
      if (
        followerParams !== null &&
        idmAcceleration(followerV, followerGap - carLength, followerV - params.v0, followerParams) < -B_SAFE
      )
        continue;
      const score = Math.min(leaderGap, followerGap);
      if (score > bestScore) {
        bestScore = score;
        best = { s, lane: lane.global };
      }
    }
  }
  return best;
}
