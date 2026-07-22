/**
 * Network: connects roads at their ends. A link joins road A's end to road B's
 * start; lane connections are derived automatically — A's forward lanes feed B's
 * forward lanes, B's backward lanes feed A's backward lanes, clamped by index when
 * lane counts differ.
 *
 * Sim model: a car's constraint is its lane leader, plus a "downstream" constraint
 * at the road end — either the stop sign (unconnected end) or a virtual leader:
 * the tail car on the connected road. Crossing the seam remaps the car to the next
 * road's lane, preserving speed.
 */
import {
  advanceLateral,
  B_SAFE,
  DELTA_A,
  idmAcceleration,
  LANE_CHANGE_COOLDOWN,
  type Car,
  type IdmParams,
} from './idm';
import type { PathPoint, Road } from './road';

/** A signal stop line on one lane: cars on `lane` brake for a standing obstacle at `s`. */
export interface NetObstacle {
  lane: number; // global lane index
  s: number;
}

export enum PriorityType {
  DIRECT = 'DIRECT',
  MERGE = 'MERGE',
  DIVERGE = 'DIVERGE',
  YIELD = 'YIELD',
}

/** A lane exit connected to another road's lane entrance. */
export interface LaneConnection {
  toRoad: number;
  toLane: number; // local lane index on the target road
  entranceS: number; // arc position where traffic enters the target road
  priority: PriorityType;
}

/** Lane-level graph node derived from the road/lane arrays and route table. */
export interface LaneNode {
  id: string;
  road: number;
  lane: number; // local lane index on `road`
  global: number;
  length: number;
  direction: 1 | -1;
  offset: number;
  leftNeighbor: number | null; // global lane index
  rightNeighbor: number | null; // global lane index
  outgoingConnections: LaneConnection[];
  incomingConnections: LaneConnection[];
  point(s: number): PathPoint;
}

/** Route indices into a lane's connection list. */
export const ROUTE_STRAIGHT = 0;
export const ROUTE_RIGHT = 1;
export const ROUTE_LEFT = 2;

export interface Network {
  roads: Road[];
  /** Prefix sums of lane counts: global lane index = laneOffsets[road] + localLane. */
  laneOffsets: number[];
  numLanes: number;
  /** Lane-level topology view. Kept additive for now; `exit` remains the source of routing behavior. */
  lanes: LaneNode[];
  /** exit[road][lane] = route-indexed connections at that lane's travel end (empty = stop sign; null entry = route unavailable). */
  exit: (LaneConnection | null)[][][];
  /** Global lanes closed to spawning (e.g. an entry-closed approach at an intersection). */
  closedLanes: Set<number>;
}

export function globalLane(net: Network, road: number, lane: number): number {
  return net.laneOffsets[road] + lane;
}

/** Resolves a global lane index to (road, localLane). Linear scan: networks are tiny. */
export function locate(net: Network, global: number): { road: number; lane: number } {
  for (let r = net.roads.length - 1; r >= 0; r--) {
    if (global >= net.laneOffsets[r]) return { road: r, lane: global - net.laneOffsets[r] };
  }
  return { road: 0, lane: 0 };
}

export function laneNode(net: Network, global: number): LaneNode {
  return net.lanes[global];
}

export function laneConnectionFor(net: Network, global: number, route: number): LaneConnection | null {
  const lane = laneNode(net, global);
  const conns = net.exit[lane.road][lane.lane];
  return conns.length === 0 ? null : conns[Math.min(route, conns.length - 1)];
}

export function connectionTargetLane(net: Network, conn: LaneConnection): number {
  return globalLane(net, conn.toRoad, conn.toLane);
}

export function lateralNeighbors(net: Network, global: number): number[] {
  const lane = laneNode(net, global);
  return [lane.leftNeighbor, lane.rightNeighbor].filter((n): n is number => n !== null);
}

export function lanePathPoint(net: Network, global: number, s: number): PathPoint {
  return laneNode(net, global).point(s);
}

/** A link between two road ends. end: 1 = road end (s = length), 0 = road start (s = 0). */
export type RoadLink = [a: number, b: number, aEnd?: number, bEnd?: number];

function uniqueConnections(conns: (LaneConnection | null)[]): LaneConnection[] {
  return [...new Set(conns.filter((conn): conn is LaneConnection => conn !== null))];
}

/** Rebuilds the additive lane graph view from `roads` and the route-indexed `exit` table. */
export function rebuildLaneGraph(net: Network): void {
  const lanes: LaneNode[] = [];
  net.roads.forEach((road, ri) => {
    road.lanes.forEach((lane, li) => {
      const global = globalLane(net, ri, li);
      lanes[global] = {
        id: `r${ri}:l${li}`,
        road: ri,
        lane: li,
        global,
        length: road.length,
        direction: lane.direction,
        offset: lane.offset,
        leftNeighbor: lane.leftNeighbor === null ? null : globalLane(net, ri, lane.leftNeighbor),
        rightNeighbor: lane.rightNeighbor === null ? null : globalLane(net, ri, lane.rightNeighbor),
        outgoingConnections: uniqueConnections(net.exit[ri][li]),
        incomingConnections: [],
        point: lane.point,
      };
    });
  });

  lanes.forEach((lane) => {
    lane.outgoingConnections.forEach((conn) => {
      lanes[globalLane(net, conn.toRoad, conn.toLane)].incomingConnections.push(conn);
    });
  });
  net.lanes = lanes;
}

/**
 * Builds a network from roads and links. Lane mapping is automatic: lanes exiting
 * at one side feed the lanes entering at the other side, clamped by index.
 */
export function buildNetwork(roads: Road[], links: RoadLink[]): Network {
  const laneOffsets: number[] = [];
  let total = 0;
  for (const road of roads) {
    laneOffsets.push(total);
    total += road.lanes.length;
  }
  const exit: LaneConnection[][][] = roads.map((r) => r.lanes.map(() => []));

  // Lanes exiting (leaving) or entering at one side of a road, in offset order.
  const lanesAt = (roadIdx: number, end: number, exiting: boolean): number[] => {
    const dir = exiting ? (end === 1 ? 1 : -1) : end === 1 ? -1 : 1;
    return roads[roadIdx].lanes.flatMap((lane, i) => (lane.direction === dir ? [i] : []));
  };

  for (const [a, b, aEnd = 1, bEnd = 0] of links) {
    const exitA = lanesAt(a, aEnd, true);
    const enterB = lanesAt(b, bEnd, false);
    const exitB = lanesAt(b, bEnd, true);
    const enterA = lanesAt(a, aEnd, false);
    if (enterB.length > 0) {
      exitA.forEach((lane, i) => {
        exit[a][lane].push({
          toRoad: b,
          toLane: enterB[Math.min(i, enterB.length - 1)],
          entranceS: bEnd === 0 ? 0 : roads[b].length,
          priority: PriorityType.DIRECT,
        });
      });
    }
    if (enterA.length > 0) {
      exitB.forEach((lane, j) => {
        exit[b][lane].push({
          toRoad: a,
          toLane: enterA[Math.min(j, enterA.length - 1)],
          entranceS: aEnd === 0 ? 0 : roads[a].length,
          priority: PriorityType.DIRECT,
        });
      });
    }
  }
  const net: Network = { roads, laneOffsets, numLanes: total, lanes: [], exit, closedLanes: new Set<number>() };
  rebuildLaneGraph(net);
  return net;
}

/** The connection a car follows at its lane's end, by route (clamped to what exists). */
function connectionFor(net: Network, road: number, lane: number, route: number): LaneConnection | null {
  return laneConnectionFor(net, globalLane(net, road, lane), route);
}

/** Nearest car in `laneIndex` (local, same road) ahead of (or behind) car `me`, in travel direction. */
function nearestInLane(
  net: Network,
  cars: Car[],
  me: number,
  road: number,
  lane: number,
  direction: number,
  ahead: boolean,
): { index: number; gap: number; v: number } | null {
  const g = globalLane(net, road, lane);
  let best: { index: number; gap: number; v: number } | null = null;
  for (let j = 0; j < cars.length; j++) {
    if (j === me || cars[j].lane !== g) continue;
    const d = (cars[j].s - cars[me].s) * direction;
    const gap = ahead ? d : -d;
    if (gap > 0 && (best === null || gap < best.gap)) best = { index: j, gap, v: cars[j].v };
  }
  return best;
}

interface Constraint {
  gap: number;
  vLead: number;
}

/**
 * What a car must brake for at its lane's end: the stop sign (unconnected), or the
 * tail car on the connected road — walking the chain while lanes are empty, so a
 * stop at the far end of the network propagates upstream.
 */
function downstream(
  net: Network,
  cars: Car[],
  me: number,
  road: number,
  lane: number,
  distToExit: number,
  carLength: number,
): Constraint {
  let gap = distToExit;
  let r = road;
  let l = lane;
  for (let hop = 0; hop < 8; hop++) {
    const conn = connectionFor(net, r, l, cars[me].route);
    if (!conn) return { gap: gap - carLength / 2, vLead: 0 }; // stop sign at this end
    const tGlobal = connectionTargetLane(net, conn);
    const target = laneNode(net, tGlobal);
    let best: { dist: number; v: number } | null = null;
    for (let j = 0; j < cars.length; j++) {
      if (j === me || cars[j].lane !== tGlobal) continue;
      const dist = target.direction > 0 ? cars[j].s : target.length - cars[j].s;
      if (best === null || dist < best.dist) best = { dist, v: cars[j].v };
    }
    if (best) return { gap: gap + best.dist - carLength, vLead: best.v };
    gap += target.length;
    r = conn.toRoad;
    l = conn.toLane;
  }
  return { gap: 1e6, vLead: 0 }; // closed loop with nobody downstream: free road
}

function accelToward(
  car: Car,
  leader: { gap: number; v: number } | null,
  carLength: number,
  p: IdmParams,
): number {
  return leader
    ? idmAcceleration(car.v, leader.gap - carLength, car.v - leader.v, p)
    : idmAcceleration(car.v, 1e6, 0, p);
}

const LANE_CHANGE_MIN_DIST = 40; // m: no lane changes this close to the lane end

/** MOBIL-lite lane changes, scoped to adjacent same-direction lanes on the same road. */
function updateNetworkLanes(net: Network, cars: Car[], params: IdmParams[], carLength: number): void {
  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    if (car.cooldown > 0) continue;
    const current = laneNode(net, car.lane);
    const { road: ri, lane: li, direction: dir } = current;
    // A route's meaning is lane-indexed, so changing lanes near an exit can strand a
    // car on a lane where its route doesn't exist (e.g. a turning car stuck going
    // straight). Like real drivers, cars commit to their lane before the intersection.
    const distToExit = dir > 0 ? current.length - car.s : car.s;
    if (distToExit < LANE_CHANGE_MIN_DIST) continue;
    const accelHere = accelToward(car, nearestInLane(net, cars, i, ri, li, dir, true), carLength, params[i]);
    let bestTarget = -1;
    let bestAccel = accelHere + DELTA_A;
    lateralNeighbors(net, car.lane).forEach((targetGlobal) => {
      const targetLane = laneNode(net, targetGlobal);
      const accelThere = accelToward(car, nearestInLane(net, cars, i, ri, targetLane.lane, dir, true), carLength, params[i]);
      if (accelThere <= bestAccel || accelThere < -B_SAFE) return;
      const follower = nearestInLane(net, cars, i, ri, targetLane.lane, dir, false);
      if (follower) {
        const followerAccel = idmAcceleration(
          follower.v,
          follower.gap - carLength,
          follower.v - car.v,
          params[follower.index],
        );
        if (followerAccel < -B_SAFE) return;
      }
      bestTarget = targetLane.lane;
      bestAccel = accelThere;
    });
    if (bestTarget >= 0) {
      // If the car's route doesn't exist on the new lane, take what the lane offers
      // (wrong lane for the turn → go wherever the lane goes).
      const conns = net.exit[ri][bestTarget];
      const usable = conns.length > 0 && conns[Math.min(car.route, conns.length - 1)];
      if (!usable) {
        const fallback = conns.findIndex((c) => c !== null);
        car.route = fallback >= 0 ? fallback : 0;
      }
      car.lane = globalLane(net, ri, bestTarget);
      car.cooldown = LANE_CHANGE_COOLDOWN;
      car.laneFrom = car.lateral;
      car.laneProgress = 0;
    }
  }
}

/** Advances every car in the network by one fixed step, crossing seams where connected. */
export function stepNetwork(
  net: Network,
  cars: Car[],
  params: IdmParams[],
  carLength: number,
  dt: number,
  obstacles: NetObstacle[] = [],
): void {
  const accels = cars.map((car, i) => {
    const { road: ri, lane: li } = locate(net, car.lane);
    const road = net.roads[ri];
    const dir = road.lanes[li].direction;
    let accel = idmAcceleration(car.v, 1e6, 0, params[i]); // free road
    const leader = nearestInLane(net, cars, i, ri, li, dir, true);
    if (leader) accel = Math.min(accel, accelToward(car, leader, carLength, params[i]));
    const distToExit = dir > 0 ? road.length - car.s : car.s;
    const down = downstream(net, cars, i, ri, li, distToExit, carLength);
    accel = Math.min(accel, idmAcceleration(car.v, down.gap, car.v - down.vLead, params[i]));
    // Signal stop lines on this lane (red phases).
    for (const obstacle of obstacles) {
      if (obstacle.lane !== car.lane) continue;
      const d = (obstacle.s - car.s) * dir;
      if (d > 0) accel = Math.min(accel, idmAcceleration(car.v, d - carLength / 2, car.v, params[i]));
    }
    return accel;
  });

  updateNetworkLanes(net, cars, params, carLength);

  for (let i = 0; i < cars.length; i++) {
    const car = cars[i];
    const { road: ri, lane: li } = locate(net, car.lane);
    const road = net.roads[ri];
    const dir = road.lanes[li].direction;
    car.a = accels[i];
    car.v = Math.max(0, car.v + car.a * dt);
    car.s += dir * car.v * dt;

    // Seam crossing: remap to the connected road's lane, keeping speed and overshoot.
    const exitS = dir > 0 ? road.length : 0;
    const overshoot = (car.s - exitS) * dir;
    if (overshoot > 0) {
      const conn = connectionFor(net, ri, li, car.route);
      if (conn) {
        const target = laneNode(net, connectionTargetLane(net, conn));
        car.s = conn.entranceS + overshoot * target.direction;
        car.lane = target.global;
        car.lateral = car.lane;
        car.laneFrom = car.lane;
        car.laneProgress = 1;
        car.lateralVel = 0;
      } else {
        car.s = exitS; // stop sign clamp
      }
    }
    advanceLateral(car, dt);
  }
}

/** Bumper gap to the nearest car ahead in the same lane, or null when alone. */
export function networkGapAhead(net: Network, cars: Car[], i: number, carLength: number): number | null {
  const lane = laneNode(net, cars[i].lane);
  const leader = nearestInLane(net, cars, i, lane.road, lane.lane, lane.direction, true);
  return leader ? leader.gap - carLength : null;
}

/**
 * Best lane entrance to spawn at, or null when none is safe. Only unfed entrances
 * (no connection flowing into them) are candidates. Safe = the new car would not
 * brake harder than B_SAFE behind its leader.
 */
export function networkSpawnSlot(
  net: Network,
  cars: Car[],
  params: IdmParams,
  carLength: number,
): { s: number; lane: number } | null {
  // Lanes already fed by any route connection are not spawn entrances.
  const fed = new Set<number>();
  net.roads.forEach((_, r) => {
    net.exit[r].forEach((conns) => {
      conns.forEach((conn) => {
        if (conn) fed.add(connectionTargetLane(net, conn));
      });
    });
  });

  let best: { s: number; lane: number } | null = null;
  let bestGap = -Infinity;
  net.roads.forEach((road, ri) => {
    road.lanes.forEach((lane, li) => {
      const g = globalLane(net, ri, li);
      if (fed.has(g) || net.closedLanes.has(g)) return;
      const s = lane.direction > 0 ? 0 : road.length;
      let leaderGap = Infinity;
      let leaderV = params.v0;
      for (const car of cars) {
        if (car.lane !== g) continue;
        const gap = (car.s - s) * lane.direction;
        if (gap > 0 && gap < leaderGap) {
          leaderGap = gap;
          leaderV = car.v;
        }
      }
      if (idmAcceleration(params.v0, leaderGap - carLength, params.v0 - leaderV, params) < -B_SAFE)
        return;
      if (leaderGap > bestGap) {
        bestGap = leaderGap;
        best = { s, lane: g };
      }
    });
  });
  return best;
}
