/**
 * Intersection: a 4-way crossing of two two-way roads, built from the Road/Network
 * primitives. Four approach roads (S/N/E/W) meet two short connector roads (NS and
 * EW) inside the zone; links wire them end-to-start so cars flow straight through.
 * Each way can be open, entry-closed, exit-closed, or fully closed (not built);
 * roads, connectors, turn arcs, signals, and spawns derive from those states.
 */
import type { Car } from './idm';
import {
  buildNetwork,
  globalLane,
  locate,
  ROUTE_LEFT,
  type NetObstacle,
  type Network,
  type RoadLink,
} from './network';
import { Road } from './road';

export type WayState = 'open' | 'in' | 'out' | 'both';
export type Way = 'n' | 'e' | 's' | 'w';

export interface IntersectionConfig {
  approach: number; // m, length of each approach road
  lanesEachWay: number; // lanes per direction on every approach
  /** Per way: open / entry closed / exit closed / fully closed (not built). */
  closed: { n: WayState; e: WayState; s: WayState; w: WayState };
}

export interface Pose2 {
  x: number;
  z: number;
  hx: number;
  hz: number;
}

export interface TurnSpec {
  radius: number;
  entry: Pose2;
  exit: Pose2;
  srcLane: number;
  dstLane: number;
}

/**
 * Solves one turn movement's arc geometry: the entry pose on the source lane, the
 * exit pose on the matching destination lane (outer → outer, inner → inner), and the
 * arc radius that joins them with a 90° sweep. All numeric — no hand-tuned radii.
 */
export function turnArcSpec(from: Way, to: Way, kind: 'right' | 'left', lanes: number, handed: number): TurnSpec {
  const zoneHalf = 4 * lanes;
  const geom: Record<Way, { edge: [number, number]; h: [number, number]; r: [number, number] }> = {
    s: { edge: [0, -zoneHalf], h: [0, 1], r: [-1, 0] },
    n: { edge: [0, zoneHalf], h: [0, 1], r: [-1, 0] },
    e: { edge: [zoneHalf, 0], h: [1, 0], r: [0, 1] },
    w: { edge: [-zoneHalf, 0], h: [1, 0], r: [0, 1] },
  };
  const enteringDir = (way: Way): number => (way === 'n' || way === 'e' ? -1 : 1);
  const dirF = enteringDir(from);
  const dirT = -enteringDir(to);
  // Right turns from the outermost lane, left turns from the innermost, both sides.
  const srcLane = kind === 'right' ? (dirF > 0 ? lanes - 1 : 2 * lanes - 1) : dirF > 0 ? 0 : lanes;
  const dstLane = kind === 'right' ? (dirT > 0 ? lanes - 1 : 2 * lanes - 1) : dirT > 0 ? 0 : lanes;
  const o1 = dirF > 0 ? (2 + 4 * srcLane) * handed : -(2 + 4 * (srcLane - lanes)) * handed;
  const o2 = dirT > 0 ? (2 + 4 * dstLane) * handed : -(2 + 4 * (dstLane - lanes)) * handed;
  const g1 = geom[from];
  const g2 = geom[to];
  const h1: [number, number] = [g1.h[0] * dirF, g1.h[1] * dirF];
  const h2: [number, number] = [g2.h[0] * dirT, g2.h[1] * dirT];
  const entry: Pose2 = { x: g1.edge[0] + g1.r[0] * o1, z: g1.edge[1] + g1.r[1] * o1, hx: h1[0], hz: h1[1] };
  const exit: Pose2 = { x: g2.edge[0] + g2.r[0] * o2, z: g2.edge[1] + g2.r[1] * o2, hx: h2[0], hz: h2[1] };
  // 90° turn: the arc center is entry + R·n where n is the turn-side normal of h1.
  // Solve for R so that the exit point is on the circle and perpendicular to h2.
  const n = kind === 'right' ? [-h1[1], h1[0]] : [h1[1], -h1[0]];
  const denom = n[0] * h2[0] + n[1] * h2[1];
  const radius = Math.max(Math.abs(((exit.x - entry.x) * h2[0] + (exit.z - entry.z) * h2[1]) / denom), 1.5);
  return { radius, entry, exit, srcLane, dstLane };
}

export interface IntersectionState {
  net: Network;
  zoneHalf: number; // half the zone's side length (m)
  /** Stop-line obstacles per direction group (global lane + s). */
  entries: { ns: NetObstacle[]; ew: NetObstacle[] };
  /** Road indices by key: 's' | 'n' | 'e' | 'w' | 'nsConn' | 'ewConn' | 'sRight' | ... */
  roadIndex: Record<string, number>;
  /** Opposing stream lanes per approach lane (for left-turn yield). */
  opposing: Map<number, number[]>;
}

/** Distance from the zone edge to the stop line (leaves room for the crosswalk after it). */
export const STOP_BACK = 4.5;

export function buildIntersection(cfg: IntersectionConfig, handed = 1): IntersectionState {
  const lanes = cfg.lanesEachWay;
  const mk = (length: number): Road =>
    new Road(
      {
        shape: 'straight',
        length,
        radius: 50,
        angle: 90,
        lanesForward: lanes,
        lanesBackward: lanes,
      },
      handed,
    );
  const mkArc = (radius: number, angle: number): Road => {
    // Turn paths run a single lane centered on the arc, from approach lane to exit lane.
    const road = new Road({ shape: 'arc', length: 0, radius, angle, lanesForward: 1, lanesBackward: 0 }, handed);
    road.lanes[0].offset = 0;
    return road;
  };

  const built = (way: Way): boolean => cfg.closed[way] !== 'both';
  const canEnter = (way: Way): boolean => cfg.closed[way] === 'open' || cfg.closed[way] === 'out';
  const canExit = (way: Way): boolean => cfg.closed[way] === 'open' || cfg.closed[way] === 'in';
  /** Travel direction into the zone for a way's entering lanes. */
  const enteringDir = (way: Way): number => (way === 'n' || way === 'e' ? -1 : 1);

  const zoneHalf = 4 * lanes; // the zone is exactly as wide as the roads it joins
  const roadIndex: Record<string, number> = {};
  const roads: Road[] = [];
  const add = (key: string, road: Road): number => {
    roadIndex[key] = roads.length;
    roads.push(road);
    return roadIndex[key];
  };
  const has = (key: string): boolean => roadIndex[key] !== undefined;

  (['s', 'n', 'e', 'w'] as Way[]).forEach((way) => {
    if (built(way)) add(way, mk(cfg.approach));
  });
  // Connectors carry straight flow both ways; build one when either direction can flow.
  if ((canEnter('s') && canExit('n')) || (canEnter('n') && canExit('s'))) add('nsConn', mk(zoneHalf * 2));
  if ((canEnter('w') && canExit('e')) || (canEnter('e') && canExit('w'))) add('ewConn', mk(zoneHalf * 2));

  // Turn arcs: radius solved per movement from the entry/exit lane geometry, so
  // multi-lane intersections curve onto the matching lane instead of guessing radii.
  const moves: { key: string; from: Way; to: Way; spec: TurnSpec }[] = [
    { key: 'sRight', from: 's', to: 'w', spec: turnArcSpec('s', 'w', 'right', lanes, handed) },
    { key: 'sLeft', from: 's', to: 'e', spec: turnArcSpec('s', 'e', 'left', lanes, handed) },
    { key: 'nRight', from: 'n', to: 'e', spec: turnArcSpec('n', 'e', 'right', lanes, handed) },
    { key: 'nLeft', from: 'n', to: 'w', spec: turnArcSpec('n', 'w', 'left', lanes, handed) },
    { key: 'eRight', from: 'e', to: 's', spec: turnArcSpec('e', 's', 'right', lanes, handed) },
    { key: 'eLeft', from: 'e', to: 'n', spec: turnArcSpec('e', 'n', 'left', lanes, handed) },
    { key: 'wRight', from: 'w', to: 'n', spec: turnArcSpec('w', 'n', 'right', lanes, handed) },
    { key: 'wLeft', from: 'w', to: 's', spec: turnArcSpec('w', 's', 'left', lanes, handed) },
  ];
  for (const m of moves) {
    if (canEnter(m.from) && canExit(m.to)) add(m.key, mkArc(m.spec.radius, m.key.endsWith('Right') ? 90 : -90));
  }

  const links: RoadLink[] = [];
  if (has('s') && has('nsConn')) links.push([roadIndex.s, roadIndex.nsConn]);
  if (has('nsConn') && has('n')) links.push([roadIndex.nsConn, roadIndex.n]);
  if (has('w') && has('ewConn')) links.push([roadIndex.w, roadIndex.ewConn]);
  if (has('ewConn') && has('e')) links.push([roadIndex.ewConn, roadIndex.e]);
  for (const m of moves) {
    // approach → arc (S/W enter from their end side, N/E from their start side)
    if (has(m.from) && has(m.key)) links.push([roadIndex[m.from], roadIndex[m.key], enteringDir(m.from) > 0 ? 1 : 0, 0]);
  }
  for (const m of moves) {
    if (has(m.key) && has(m.to)) links.push([roadIndex[m.key], roadIndex[m.to], 1, m.to === 'w' || m.to === 's' ? 1 : 0] as RoadLink);
  }
  const net = buildNetwork(roads, links);

  // Turn arcs exit onto the matching destination lane (outer → outer, inner → inner).
  for (const m of moves) {
    if (has(m.key)) {
      const conn = net.exit[roadIndex[m.key]][0][0];
      if (conn) conn.toLane = m.spec.dstLane;
    }
  }

  // Exit-closed ways: nothing may flow into them.
  for (const way of ['s', 'n', 'e', 'w'] as Way[]) {
    if (!canExit(way) && has(way)) {
      net.exit.forEach((roadExits) =>
        roadExits.forEach((conns) => {
          conns.forEach((conn, i) => {
            if (conn && conn.toRoad === roadIndex[way]) conns[i] = null;
          });
        }),
      );
    }
    // Entry-closed ways: entering lanes dead-end at the zone edge and take no spawns.
    if (!canEnter(way) && has(way)) {
      net.roads[roadIndex[way]].lanes.forEach((lane, li) => {
        if (lane.direction === enteringDir(way)) {
          net.exit[roadIndex[way]][li] = [];
          net.closedLanes.add(globalLane(net, roadIndex[way], li));
        }
      });
    }
  }

  // Route connections per approach lane, identified by target: the connector is the
  // straight route (available only if it exits somewhere), turn arcs are right/left.
  // Right turns from the lane farthest from the yellow line, left from the closest;
  // others fall back to straight.
  const connectorViable = (r: number, dir: number): boolean =>
    net.roads[r].lanes.some(
      (lane, li) => lane.direction === dir && net.exit[r][li].some((c) => c !== null),
    );
  for (const way of ['s', 'n', 'e', 'w'] as Way[]) {
    if (!canEnter(way) || !has(way)) continue;
    const r = roadIndex[way];
    net.roads[r].lanes.forEach((lane, li) => {
      const conns = net.exit[r][li];
      if (conns.length === 0) return; // lanes leaving the zone, not entering it
      const found =
        conns.find((c) => c !== null && (c.toRoad === roadIndex.nsConn || c.toRoad === roadIndex.ewConn)) ??
        null;
      // Straight is only a real route when the connector still exits somewhere; a
      // closed far side means this approach must turn instead of stopping mid-zone.
      const straight = found && connectorViable(found.toRoad, lane.direction) ? found : null;
      const right = has(`${way}Right`)
        ? conns.find((c) => c !== null && c.toRoad === roadIndex[`${way}Right`]) ?? null
        : null;
      const left = has(`${way}Left`)
        ? conns.find((c) => c !== null && c.toRoad === roadIndex[`${way}Left`]) ?? null
        : null;
      const forwardOuter = lane.direction > 0 && li === lanes - 1;
      const forwardInner = lane.direction > 0 && li === 0;
      const backwardOuter = lane.direction < 0 && li === net.roads[r].lanes.length - 1;
      const backwardInner = lane.direction < 0 && li === lanes;
      net.exit[r][li] = [
        straight,
        forwardOuter || backwardOuter ? (right ?? straight) : straight,
        forwardInner || backwardInner ? (left ?? straight) : straight,
      ];
    });
  }

  // Signalized entries for ways that may enter.
  const ns: NetObstacle[] = [];
  const ew: NetObstacle[] = [];
  if (canEnter('s')) {
    net.roads[roadIndex.s].lanes.forEach((lane, li) => {
      if (lane.direction > 0) ns.push({ lane: globalLane(net, roadIndex.s, li), s: cfg.approach - STOP_BACK });
    });
  }
  if (canEnter('n')) {
    net.roads[roadIndex.n].lanes.forEach((lane, li) => {
      if (lane.direction < 0) ns.push({ lane: globalLane(net, roadIndex.n, li), s: STOP_BACK });
    });
  }
  if (canEnter('w')) {
    net.roads[roadIndex.w].lanes.forEach((lane, li) => {
      if (lane.direction > 0) ew.push({ lane: globalLane(net, roadIndex.w, li), s: cfg.approach - STOP_BACK });
    });
  }
  if (canEnter('e')) {
    net.roads[roadIndex.e].lanes.forEach((lane, li) => {
      if (lane.direction < 0) ew.push({ lane: globalLane(net, roadIndex.e, li), s: STOP_BACK });
    });
  }

  // Opposing stream lanes per approach lane (used by left-turn yield).
  const stream = (way: Way, dir: number): number[] =>
    built(way) && canEnter(way)
      ? net.roads[roadIndex[way]].lanes.flatMap((lane, li) =>
          lane.direction === dir ? [globalLane(net, roadIndex[way], li)] : [],
        )
      : [];
  const connStream = (conn: string, dir: number): number[] =>
    has(conn)
      ? net.roads[roadIndex[conn]].lanes.flatMap((lane, li) =>
          lane.direction === dir ? [globalLane(net, roadIndex[conn], li)] : [],
        )
      : [];
  const opposing = new Map<number, number[]>();
  stream('s', 1).forEach((g) => opposing.set(g, [...stream('n', -1), ...connStream('nsConn', -1)]));
  stream('n', -1).forEach((g) => opposing.set(g, [...stream('s', 1), ...connStream('nsConn', 1)]));
  stream('w', 1).forEach((g) => opposing.set(g, [...stream('e', -1), ...connStream('ewConn', -1)]));
  stream('e', -1).forEach((g) => opposing.set(g, [...stream('w', 1), ...connStream('ewConn', 1)]));

  return { net, zoneHalf, entries: { ns, ew }, roadIndex, opposing };
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

const YIELD_ETA = 4.5; // s: block a left turn when an opponent arrives sooner than this
const YIELD_MIN_SPEED = 1; // m/s: stopped opponents never block

/**
 * Gap-acceptance yield for left turns (replaces a blunt distance rule that starved
 * turns under continuous opposing traffic): a left-routed car is held at its stop
 * line only while an opposing-stream car is moving AND either inside the zone or
 * arriving at it within YIELD_ETA seconds.
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
        const inZone = oRoad.length <= 2 * state.zoneHalf; // connector or turn arc: inside the zone
        const dist = oRoad.lanes[ol.lane].direction > 0 ? oRoad.length - other.s : other.s;
        return inZone || dist < Math.max(other.v, 1) * YIELD_ETA; // arriving within YIELD_ETA seconds
      }),
    );
    if (blocked) out.push(entry);
  });
  return out;
}
