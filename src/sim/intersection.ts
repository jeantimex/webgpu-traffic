/**
 * Intersection: a 4-way crossing of two two-way roads, built from the Road/Network
 * primitives. Four approach roads (S/N/E/W) meet two short connector roads (NS and
 * EW) inside the zone; links wire them end-to-start so cars flow straight through.
 * Right-of-way is a 2-phase signal with an all-red clearance: the stopped direction's
 * entries get a virtual standing car at their stop line.
 *
 * Road order in the network: [S, N, E, W, NS connector, EW connector].
 */
import type { Car } from './idm';
import {
  buildNetwork,
  globalLane,
  locate,
  ROUTE_LEFT,
  ROUTE_RIGHT,
  ROUTE_STRAIGHT,
  type LaneConnection,
  type NetObstacle,
  type Network,
  type RoadLink,
} from './network';
import { Road } from './road';

export interface IntersectionConfig {
  approach: number; // m, length of each approach road
  lanesEachWay: number; // lanes per direction on every approach
}

export interface IntersectionState {
  net: Network;
  zoneHalf: number; // half the zone's side length (m)
  /** Stop-line obstacles per direction group (global lane + s). */
  entries: { ns: NetObstacle[]; ew: NetObstacle[] };
  /** Road indices. */
  idx: { s: number; n: number; e: number; w: number; nsConn: number; ewConn: number };
  /** Opposing stream lanes per approach lane (for left-turn yield). */
  opposing: Map<number, number[]>;
}

export const IDX = { s: 0, n: 1, e: 2, w: 3, nsConn: 4, ewConn: 5 } as const;

/** Turn-arc road indices: [approach][route]. */
export const ARC = {
  sRight: 6,
  sLeft: 7,
  nRight: 8,
  nLeft: 9,
  eRight: 10,
  eLeft: 11,
  wRight: 12,
  wLeft: 13,
} as const;

/** Distance from the zone edge to the stop line (leaves room for the crosswalk after it). */
export const STOP_BACK = 4.5;

export function buildIntersection(cfg: IntersectionConfig): IntersectionState {
  const lanes = cfg.lanesEachWay;
  const mk = (length: number): Road =>
    new Road({
      shape: 'straight',
      length,
      radius: 50,
      angle: 90,
      lanesForward: lanes,
      lanesBackward: lanes,
    });
  const mkArc = (radius: number, angle: number): Road => {
    // Turn paths run a single lane centered on the arc, from approach lane to exit lane.
    const road = new Road({ shape: 'arc', length: 0, radius, angle, lanesForward: 1, lanesBackward: 0 });
    road.lanes[0].offset = 0;
    return road;
  };

  const zoneHalf = 4 * lanes; // the zone is exactly as wide as the roads it joins
  const roads = [
    ...[cfg.approach, cfg.approach, cfg.approach, cfg.approach].map(mk),
    ...[zoneHalf * 2, zoneHalf * 2].map(mk),
    // Turn arcs, exact for one lane each way: right turns sweep 2 m, left turns 6 m.
    // ponytail: with more lanes the radii only approximate the outer/inner turn lanes.
    mkArc(2, -90), // S right → W
    mkArc(6, 90), // S left → E
    mkArc(2, -90), // N right → E
    mkArc(6, 90), // N left → W
    mkArc(2, -90), // E right → S
    mkArc(6, 90), // E left → N
    mkArc(2, -90), // W right → N
    mkArc(6, 90), // W left → S
  ];
  const links: RoadLink[] = [
    // straight flow
    [IDX.s, IDX.nsConn],
    [IDX.nsConn, IDX.n],
    [IDX.w, IDX.ewConn],
    [IDX.ewConn, IDX.e],
    // approach → turn arcs (S/W enter the zone from their end side, N/E from their start side)
    [IDX.s, ARC.sRight],
    [IDX.s, ARC.sLeft],
    [IDX.n, ARC.nRight, 0, 0],
    [IDX.n, ARC.nLeft, 0, 0],
    [IDX.e, ARC.eRight, 0, 0],
    [IDX.e, ARC.eLeft, 0, 0],
    [IDX.w, ARC.wRight],
    [IDX.w, ARC.wLeft],
    // turn arcs → exit roads (right-hand traffic: right turns to the near road, left
    // turns across to the far road)
    [ARC.sRight, IDX.e],
    [ARC.sLeft, IDX.w, 1, 1],
    [ARC.nRight, IDX.w, 1, 1],
    [ARC.nLeft, IDX.e],
    [ARC.eRight, IDX.n],
    [ARC.eLeft, IDX.s, 1, 1],
    [ARC.wRight, IDX.s, 1, 1],
    [ARC.wLeft, IDX.n],
  ];
  const net = buildNetwork(roads, links);

  // Route connections per approach lane: right turns from the lane farthest from the
  // yellow line, left turns from the lane closest to it; other lanes straight-only.
  const approaches: { road: number; rightArc: number; leftArc: number }[] = [
    { road: IDX.s, rightArc: ARC.sRight, leftArc: ARC.sLeft },
    { road: IDX.n, rightArc: ARC.nRight, leftArc: ARC.nLeft },
    { road: IDX.e, rightArc: ARC.eRight, leftArc: ARC.eLeft },
    { road: IDX.w, rightArc: ARC.wRight, leftArc: ARC.wLeft },
  ];
  for (const { road: r, rightArc, leftArc } of approaches) {
    net.roads[r].lanes.forEach((lane, li) => {
      const conns = net.exit[r][li];
      if (conns.length === 0) return; // lanes leaving the zone, not entering it
      const straight = conns[0];
      const right: LaneConnection = { toRoad: rightArc, toLane: 0, entranceS: 0 };
      const left: LaneConnection = { toRoad: leftArc, toLane: 0, entranceS: 0 };
      const forwardOuter = lane.direction > 0 && li === lanes - 1;
      const forwardInner = lane.direction > 0 && li === 0;
      const backwardOuter = lane.direction < 0 && li === net.roads[r].lanes.length - 1;
      const backwardInner = lane.direction < 0 && li === lanes;
      conns[ROUTE_RIGHT] = forwardOuter || backwardOuter ? right : straight;
      conns[ROUTE_LEFT] = forwardInner || backwardInner ? left : straight;
      conns[ROUTE_STRAIGHT] = straight;
    });
  }

  // Signalized entries: lanes whose travel exits into the zone, stopping at their stop
  // line (set back from the zone edge so the crosswalk fits after it).
  const ns: NetObstacle[] = [];
  const ew: NetObstacle[] = [];
  roads[IDX.s].lanes.forEach((lane, li) => {
    if (lane.direction > 0) ns.push({ lane: globalLane(net, IDX.s, li), s: cfg.approach - STOP_BACK });
  });
  roads[IDX.n].lanes.forEach((lane, li) => {
    if (lane.direction < 0) ns.push({ lane: globalLane(net, IDX.n, li), s: STOP_BACK });
  });
  roads[IDX.w].lanes.forEach((lane, li) => {
    if (lane.direction > 0) ew.push({ lane: globalLane(net, IDX.w, li), s: cfg.approach - STOP_BACK });
  });
  roads[IDX.e].lanes.forEach((lane, li) => {
    if (lane.direction < 0) ew.push({ lane: globalLane(net, IDX.e, li), s: STOP_BACK });
  });

  // Opposing stream lanes per approach lane (used by left-turn yield).
  const group = (road: number, dir: number): number[] =>
    net.roads[road].lanes.flatMap((lane, li) => (lane.direction === dir ? [globalLane(net, road, li)] : []));
  const opposing = new Map<number, number[]>();
  for (const [a, b, conn] of [
    [IDX.s, IDX.n, IDX.nsConn],
    [IDX.w, IDX.e, IDX.ewConn],
  ] as const) {
    group(a, 1).forEach((g) => opposing.set(g, [...group(b, -1), ...group(conn, -1)]));
    group(b, -1).forEach((g) => opposing.set(g, [...group(a, 1), ...group(conn, 1)]));
  }

  return { net, zoneHalf, entries: { ns, ew }, idx: IDX, opposing };
}

export type IntersectionPhase = 'nsGreen' | 'nsYellow' | 'ewGreen' | 'ewYellow' | 'allRed';

export interface SignalSettings {
  green: number;
  yellow: number;
  red: number;
  override: string;
}

/** Complementary 2-phase cycle: nsGreen → nsYellow → ewGreen → ewYellow. One pair is always
 * green or yellow while the other is red — they are never red together (except the manual
 * all-red override). The red slider is unused here: each pair's red is the other's green. */
export function intersectionPhaseAt(clock: number, light: SignalSettings): IntersectionPhase {
  if (light.override === 'green') return 'nsGreen';
  if (light.override === 'yellow') return 'nsYellow';
  if (light.override === 'red') return 'allRed';
  const { green: g, yellow: y } = light;
  const t = clock % (2 * (g + y));
  if (t < g) return 'nsGreen';
  if (t < g + y) return 'nsYellow';
  if (t < 2 * g + y) return 'ewGreen';
  return 'ewYellow';
}

/** Obstacles to enforce for a phase: the stopped direction(s) get their stop lines. */
export function intersectionObstacles(state: IntersectionState, phase: IntersectionPhase): NetObstacle[] {
  switch (phase) {
    case 'nsGreen':
      return state.entries.ew;
    case 'ewGreen':
      return state.entries.ns;
    default:
      return [...state.entries.ns, ...state.entries.ew]; // yellow transitions and all-red: full stop
  }
}

/** What one lamp stack shows. Lamp order: [S, N, E, W] — S/N are the NS group, E/W the EW group. */
export function intersectionLampColor(
  _state: IntersectionState,
  lamp: number,
  phase: IntersectionPhase,
): 'red' | 'yellow' | 'green' {
  const nsGroup = lamp < 2;
  switch (phase) {
    case 'nsGreen':
      return nsGroup ? 'green' : 'red';
    case 'nsYellow':
      return nsGroup ? 'yellow' : 'red';
    case 'ewGreen':
      return nsGroup ? 'red' : 'green';
    case 'ewYellow':
      return nsGroup ? 'red' : 'yellow';
    default:
      return 'red';
  }
}

const YIELD_DIST = 45; // m: opposing traffic closer than this to the zone blocks a left turn
const YIELD_MIN_SPEED = 1; // m/s: stopped opponents don't block

/**
 * Stop-line obstacles for left-turning cars that must give way: a left-routed car on
 * an approach is held at its stop line while an opposing-stream car is approaching
 * the zone (and actually moving).
 */
export function leftTurnYieldObstacles(state: IntersectionState, cars: Car[]): NetObstacle[] {
  const out: NetObstacle[] = [];
  cars.forEach((car, i) => {
    if (car.route !== ROUTE_LEFT) return;
    const entry = [...state.entries.ns, ...state.entries.ew].find((e) => e.lane === car.lane);
    if (!entry) return; // not on an approach lane: no yield inside the zone
    const { road: ri, lane: li } = locate(state.net, car.lane);
    const dir = state.net.roads[ri].lanes[li].direction;
    if ((entry.s - car.s) * dir < 0) return; // already past the stop line
    const blocked = (state.opposing.get(car.lane) ?? []).some((lane) =>
      cars.some((other, j) => {
        if (j === i || other.lane !== lane || other.v < YIELD_MIN_SPEED) return false;
        const ol = locate(state.net, lane);
        const oRoad = state.net.roads[ol.road];
        const dist = oRoad.lanes[ol.lane].direction > 0 ? oRoad.length - other.s : other.s;
        return dist < YIELD_DIST;
      }),
    );
    if (blocked) out.push(entry);
  });
  return out;
}
