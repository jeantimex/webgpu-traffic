/**
 * Intersection: a 4-way crossing of two two-way roads, built from the Road/Network
 * primitives. Four approach roads (S/N/E/W) meet two short connector roads (NS and
 * EW) inside the zone; links wire them end-to-start so cars flow straight through.
 * Right-of-way is a 2-phase signal with an all-red clearance: the stopped direction's
 * entries get a virtual standing car at their stop line.
 *
 * Road order in the network: [S, N, E, W, NS connector, EW connector].
 */
import { buildNetwork, globalLane, type NetObstacle, type Network } from './network';
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
}

export const IDX = { s: 0, n: 1, e: 2, w: 3, nsConn: 4, ewConn: 5 } as const;

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

  const zoneHalf = 4 * lanes; // the zone is exactly as wide as the roads it joins
  const roads = [cfg.approach, cfg.approach, cfg.approach, cfg.approach, zoneHalf * 2, zoneHalf * 2].map(mk);
  const net = buildNetwork(roads, [
    [IDX.s, IDX.nsConn],
    [IDX.nsConn, IDX.n],
    [IDX.w, IDX.ewConn],
    [IDX.ewConn, IDX.e],
  ]);

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

  return { net, zoneHalf, entries: { ns, ew }, idx: IDX };
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
