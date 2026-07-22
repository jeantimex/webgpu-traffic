/**
 * Runnable self-check for the IDM sim. Not part of the app bundle.
 * Run: node_modules/.bin/esbuild src/sim/idm.check.ts --bundle --format=esm --outfile=.idm.check.mjs && node .idm.check.mjs && rm .idm.check.mjs
 */
import { idmAcceleration, type Car, type IdmParams } from './idm';
import { stepLaneLoop } from './loop';
import {
  buildScene3,
  buildScene4,
  SCENES,
  scene1State,
  scene2State,
  scene3State,
  scene4State,
} from '../renderer/scenes';
import {
  buildIntersection,
  intersectionObstacles,
  intersectionPhaseAt,
  leftTurnYieldObstacles,
  STOP_BACK,
  turnArcSpec,
  type IntersectionConfig,
  type IntersectionState,
  type Way,
} from './intersection';
import {
  availableRouteIndices,
  buildNetwork,
  connectionTargetLane,
  laneConnectionFor,
  laneHasRouteTable,
  laneNode,
  lanePathPoint,
  laneRouteConnections,
  lateralNeighbors,
  PriorityType,
  stepNetwork,
  type NetObstacle,
  type Network,
} from './network';
import { Road } from './road';

function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`IDM check failed: ${msg}`);
}

const fast: IdmParams = { v0: 30, T: 1.5, a: 2.0, b: 2.5, s0: 2, delta: 4 };
const slow: IdmParams = { v0: 12, T: 1.5, a: 1.2, b: 2.0, s0: 2, delta: 4 };

// Free road: accelerates below v0, holds v0.
assert(idmAcceleration(10, 1e6, 0, fast) > 0, 'accelerates below v0 on a free road');
assert(Math.abs(idmAcceleration(30, 1e6, 0, fast)) < 1e-6, '~zero acceleration at v0');
// Small gap at speed: brakes.
assert(idmAcceleration(25, 5, 25, fast) < 0, 'brakes for a close leader');

// Ring: a fast car stuck behind a slower leader changes lanes to overtake — safely.
const C = 2 * Math.PI * 40;
const CAR_LENGTH = 4.5;
const cars: Car[] = [
  { s: 0, v: 30, a: 0, lane: 0, route: 0, lateral: 0, lateralVel: 0, laneFrom: 0, laneProgress: 1, cooldown: 0 },
  { s: C / 2, v: 12, a: 0, lane: 0, route: 0, lateral: 0, lateralVel: 0, laneFrom: 0, laneProgress: 1, cooldown: 0 },
];
let minGap = Infinity;
let changedLanes = false;
let maxLateralJump = 0;
let prevLateralVel = 0;
for (let step = 0; step < 120 * 60; step++) {
  stepLaneLoop(scene1State.net, cars, [fast, slow], C, CAR_LENGTH, 1 / 60);
  if (cars[0].lane !== 0) changedLanes = true;
  maxLateralJump = Math.max(maxLateralJump, Math.abs(cars[0].lateralVel - prevLateralVel));
  prevLateralVel = cars[0].lateralVel;
  if (cars[0].lane === cars[1].lane) {
    const gap = ((((cars[1].s - cars[0].s) % C) + C) % C) - CAR_LENGTH;
    minGap = Math.min(minGap, gap);
  }
}
assert(changedLanes, 'fast car changed lanes to overtake');
assert(minGap > 0, `no collision (min same-lane gap ${minGap.toFixed(2)} m)`);
assert(cars[0].v > 25, `fast car was not stuck behind the slow one (v=${cars[0].v.toFixed(2)})`);
// Smooth lane change: lateral velocity ramps up and down, never snaps (linear slide jumps 0.5 units/s).
assert(maxLateralJump < 0.1, `lateral velocity is continuous (max jump ${maxLateralJump.toFixed(3)} units/s)`);

// Red light: a car must stop just before the stop line, then accelerate away once it turns green.
const lone: Car[] = [{ s: 0, v: 20, a: 0, lane: 0, route: 0, lateral: 0, lateralVel: 0, laneFrom: 0, laneProgress: 1, cooldown: 0 }];
const redLight = [{ s: C / 4 }];
for (let step = 0; step < 60 * 60; step++) stepLaneLoop(scene1State.net, lone, [fast], C, CAR_LENGTH, 1 / 60, redLight);
const distToLine = (((C / 4 - lone[0].s) % C) + C) % C;
assert(lone[0].v < 0.01, `stopped at the red light (v=${lone[0].v.toFixed(3)})`);
assert(
  distToLine > CAR_LENGTH / 2 && distToLine < 8,
  `stopped just before the line (${distToLine.toFixed(2)} m from center)`);
for (let step = 0; step < 10 * 60; step++) stepLaneLoop(scene1State.net, lone, [fast], C, CAR_LENGTH, 1 / 60);
assert(lone[0].v > 10, `accelerates on green (v=${lone[0].v.toFixed(1)})`);

// Different lanes: no car-following interaction — the fast car keeps its desired speed.
const twoLanes: Car[] = [
  { s: C / 2 + 30, v: 12, a: 0, lane: 0, route: 0, lateral: 0, lateralVel: 0, laneFrom: 0, laneProgress: 1, cooldown: 0 }, // slow, ahead in the inner lane
  { s: C / 2, v: 30, a: 0, lane: 1, route: 0, lateral: 1, lateralVel: 0, laneFrom: 1, laneProgress: 1, cooldown: 0 }, // fast, catching up in the outer lane
];
for (let step = 0; step < 60 * 60; step++) stepLaneLoop(scene1State.net, twoLanes, [slow, fast], C, CAR_LENGTH, 1 / 60);
assert(
  Math.abs(twoLanes[1].v - fast.v0) < 0.5,
  `fast car ignores the slow car in the other lane (v=${twoLanes[1].v.toFixed(2)})`,
);

console.log(`IDM checks passed (min gap ${minGap.toFixed(2)} m, settled at ${cars[0].v.toFixed(2)} m/s)`);

// ---------------------------------------------------------------------------
// Road + Network (building blocks)
// ---------------------------------------------------------------------------

const newCar = (s: number, v: number, lane: number, route = 0): Car => ({
  s,
  v,
  a: 0,
  lane,
  route,
  lateral: lane,
  lateralVel: 0,
  laneFrom: lane,
  laneProgress: 1,
  cooldown: 0,
});

const netOf = (...roads: Road[]): Network => buildNetwork(roads, roads.length > 1 ? [[0, 1]] : []);

function assertLanePointMatchesRoad(net: Network, road: Road, lane: number, s: number, msg: string): void {
  const base = road.point(s);
  const expectedX = base.x + base.rx * road.lanes[lane].offset;
  const expectedZ = base.z + base.rz * road.lanes[lane].offset;
  const p = lanePathPoint(net, lane, s);
  assert(Math.abs(p.x - expectedX) < 1e-9 && Math.abs(p.z - expectedZ) < 1e-9, `${msg} position`);
  assert(Math.abs(p.hx - base.hx) < 1e-9 && Math.abs(p.hz - base.hz) < 1e-9, `${msg} heading`);
}

function transformedLanePoint(net: Network, global: number, s: number): { x: number; z: number; hx: number; hz: number } {
  const lane = laneNode(net, global);
  const p = lanePathPoint(net, global, s);
  const t = scene3State.transforms[lane.road];
  return {
    x: p.x * t.cos + p.z * t.sin + t.tx,
    z: -p.x * t.sin + p.z * t.cos + t.tz,
    hx: p.hx * t.cos + p.hz * t.sin,
    hz: -p.hx * t.sin + p.hz * t.cos,
  };
}

// Path geometry: unit headings, continuous tangent (arc & S-curve derived by hand).
for (const shape of ['arc', 'scurve'] as const) {
  const road = new Road({ shape, radius: 50, angle: 60, length: 0, lanesForward: 1, lanesBackward: 0 });
  const expected = shape === 'arc' ? (50 * 60 * Math.PI) / 180 : (2 * 50 * 60 * Math.PI) / 180;
  assert(Math.abs(road.length - expected) < 1e-9, `${shape} length`);
  for (let s = 0; s < road.length - 0.5; s += 0.5) {
    const p = road.point(s);
    assert(Math.abs(Math.hypot(p.hx, p.hz) - 1) < 1e-9, `${shape} unit heading at s=${s}`);
    const p2 = road.point(s + 0.5);
    const err = Math.hypot(p2.x - p.x - p.hx * 0.5, p2.z - p.z - p.hz * 0.5);
    assert(err < 0.02, `${shape} discontinuity near s=${s} (err ${err.toFixed(4)})`);
  }
}
{
  const scurve = new Road({ shape: 'scurve', radius: 50, angle: 60, length: 0, lanesForward: 1, lanesBackward: 0 });
  const end = scurve.point(scurve.length);
  assert(Math.abs(end.hx - 1) < 1e-9 && Math.abs(end.hz) < 1e-9, 'scurve ends parallel to its start');
}

// Lane geometry: lane centerline accessors match the existing road centerline + lateral offset convention.
for (const shape of ['straight', 'arc', 'scurve'] as const) {
  const road = new Road({
    shape,
    length: 120,
    radius: 50,
    angle: 60,
    lanesForward: 2,
    lanesBackward: 1,
  });
  const net = netOf(road);
  for (const lane of [0, 1, 2]) {
    for (const s of [0, road.length / 2, road.length]) {
      assertLanePointMatchesRoad(net, road, lane, s, `${shape} lane ${lane} at s=${s.toFixed(1)}`);
    }
  }
}

// Lane geometry reads the lane's current offset, so custom/internal lanes can be adjusted after construction.
{
  const road = new Road({ shape: 'arc', length: 0, radius: 40, angle: 90, lanesForward: 1, lanesBackward: 0 });
  road.lanes[0].offset = 0;
  const net = netOf(road);
  assertLanePointMatchesRoad(net, road, 0, road.length / 2, 'mutated-offset lane geometry');
}

// Network scene poses consume lane geometry for settled cars.
{
  const s = 33;
  const theta = s / 40;
  const car = newCar(s, 12, 1);
  const pose = SCENES[0].carPose(car);
  const r = 42;
  assert(scene1State.net.numLanes === 2, 'scene 1 has a lane-backed loop network');
  assert(Math.abs(pose.x - r * Math.cos(theta)) < 1e-9, 'scene 1 pose x preserves ring geometry');
  assert(Math.abs(pose.z - r * Math.sin(theta)) < 1e-9, 'scene 1 pose z preserves ring geometry');
  assert(Math.abs(Math.cos(pose.angle) + Math.sin(theta)) < 1e-9, 'scene 1 pose heading x preserves ring geometry');
  assert(Math.abs(-Math.sin(pose.angle) - Math.cos(theta)) < 1e-9, 'scene 1 pose heading z preserves ring geometry');
}

{
  const s = 30;
  const p = scene2State.net.lanes[0].point(s);
  const car = newCar(s, 12, 0);
  const pose = SCENES[1].carPose(car);
  assert(scene2State.net.numLanes === 2, 'scene 2 has a lane-backed loop network');
  assert(Math.abs(pose.x - p.x) < 1e-9 && Math.abs(pose.z - p.z) < 1e-9, 'scene 2 pose uses square lane geometry');
  assert(Math.abs(Math.cos(pose.angle) - p.hx) < 1e-9, 'scene 2 pose heading x uses square lane geometry');
  assert(Math.abs(-Math.sin(pose.angle) - p.hz) < 1e-9, 'scene 2 pose heading z uses square lane geometry');
}

{
  buildScene3(
    { shape: 'straight', length: 120, radius: 50, angle: 90, lanesForward: 2, lanesBackward: 0 },
    { shape: 'arc', length: 0, radius: 50, angle: 60, lanesForward: 2, lanesBackward: 0 },
  );
  const net = scene3State.net!;
  const lane = 1;
  const s = 30;
  const car = newCar(s, 12, lane);
  const pose = SCENES[2].carPose(car);
  const expected = transformedLanePoint(net, lane, s);
  assert(Math.abs(pose.x - expected.x) < 1e-9 && Math.abs(pose.z - expected.z) < 1e-9, 'scene 3 pose uses lane geometry');
  assert(Math.abs(Math.cos(pose.angle) - expected.hx) < 1e-9, 'scene 3 pose heading x uses lane geometry');
  assert(Math.abs(-Math.sin(pose.angle) - expected.hz) < 1e-9, 'scene 3 pose heading z uses lane geometry');
}

// Negative angle: the arc turns right (heading angle decreases) and mirrors the left arc.
{
  const left = new Road({ shape: 'arc', radius: 50, angle: 60, length: 0, lanesForward: 1, lanesBackward: 0 });
  const right = new Road({ shape: 'arc', radius: 50, angle: -60, length: 0, lanesForward: 1, lanesBackward: 0 });
  assert(Math.abs(right.length - left.length) < 1e-9, 'same length for ±angle');
  const startAngle = Math.atan2(right.point(0).hz, right.point(0).hx);
  const endAngle = Math.atan2(right.point(right.length).hz, right.point(right.length).hx);
  assert(endAngle < startAngle, 'negative angle turns right');
  for (let s = 0; s <= right.length; s += 10) {
    const pl = left.point(s);
    const pr = right.point(s);
    assert(Math.abs(pr.x - pl.x) < 1e-9 && Math.abs(pr.z + pl.z) < 1e-9, `mirror at s=${s}`);
  }
}

// Open end is a full stop: a car must stop just before the road end.
{
  const net = netOf(new Road({ shape: 'straight', length: 120, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 0 }));
  const car = [newCar(0, 20, 0)];
  for (let step = 0; step < 30 * 60; step++) stepNetwork(net, car, [fast], CAR_LENGTH, 1 / 60);
  assert(car[0].v < 0.01, `stopped at road end (v=${car[0].v.toFixed(3)})`);
  assert(car[0].s > 112 && car[0].s < 118, `stopped just before the end (s=${car[0].s.toFixed(2)})`);
}

// One lane, no way around: fast car settles behind the slow one (mean speed over the last 10 s).
{
  const net = netOf(new Road({ shape: 'straight', length: 1500, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 0 }));
  const platoon = [newCar(0, 30, 0), newCar(150, 12, 0)];
  let vSum = 0;
  let vN = 0;
  for (let step = 0; step < 90 * 60; step++) {
    stepNetwork(net, platoon, [fast, slow], CAR_LENGTH, 1 / 60);
    if (step >= 80 * 60) {
      vSum += platoon[0].v;
      vN++;
    }
  }
  assert(Math.abs(vSum / vN - slow.v0) < 0.5, `platoon settled (mean v=${(vSum / vN).toFixed(2)})`);
}

// Two lanes one-way: fast car changes lanes to overtake.
{
  const net = netOf(new Road({ shape: 'straight', length: 400, radius: 50, angle: 90, lanesForward: 2, lanesBackward: 0 }));
  assert(net.lanes.length === net.numLanes, 'lane graph mirrors lane count');
  assert(net.lanes[0].rightNeighbor === 1 && net.lanes[1].leftNeighbor === 0, 'lane graph records lateral neighbors');
  assert(laneNode(net, 0).global === 0, 'laneNode resolves by global lane');
  assert(lateralNeighbors(net, 0)[0] === 1, 'lateralNeighbors exposes adjacent lanes');
  const race = [newCar(0, 30, 0), newCar(150, 12, 0)];
  let changed = false;
  let maxV = 0;
  for (let step = 0; step < 12 * 60; step++) {
    stepNetwork(net, race, [fast, slow], CAR_LENGTH, 1 / 60);
    if (race[0].lane !== 0) changed = true;
    maxV = Math.max(maxV, race[0].v);
  }
  assert(changed, 'fast car changed lanes to overtake on the road');
  assert(maxV > 25, `fast car was not stuck (max v=${maxV.toFixed(2)})`);
}

// Explicit lateral graph: removing a neighbor link blocks lane changes even when an empty lane exists.
{
  const road = new Road({ shape: 'straight', length: 400, radius: 50, angle: 90, lanesForward: 2, lanesBackward: 0 });
  road.lanes[0].rightNeighbor = null;
  road.lanes[1].leftNeighbor = null;
  const net = netOf(road);
  assert(lateralNeighbors(net, 0).length === 0, 'lane graph honors explicit missing lateral neighbor');
  const race = [newCar(0, 30, 0), newCar(150, 12, 0)];
  for (let step = 0; step < 12 * 60; step++) {
    stepNetwork(net, race, [fast, slow], CAR_LENGTH, 1 / 60);
  }
  assert(race[0].lane === 0, 'car cannot change lanes without an explicit lateral neighbor');
}

// Two-way: opposing traffic does not interact.
{
  const p20: IdmParams = { ...fast, v0: 20 };
  const net = netOf(new Road({ shape: 'straight', length: 1000, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 1 }));
  const both = [newCar(50, 20, 0), newCar(950, 20, 1)];
  let maxDev = 0;
  for (let step = 0; step < 22 * 60; step++) {
    stepNetwork(net, both, [p20, p20], CAR_LENGTH, 1 / 60);
    maxDev = Math.max(maxDev, Math.abs(both[0].v - 20), Math.abs(both[1].v - 20));
  }
  assert(maxDev < 0.5, `opposing cars unaffected (max deviation ${maxDev.toFixed(2)} m/s)`);
}

// Seam: a car crosses from road A to road B without slowing (no stop at the connection).
// Measured at the crossing moment, before braking for B's far end begins.
{
  const a = new Road({ shape: 'straight', length: 150, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 0 });
  const b = new Road({ shape: 'straight', length: 150, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 0 });
  const net = buildNetwork([a, b], [[0, 1]]);
  assert(net.lanes[0].outgoingConnections[0] === net.exit[0][0][0], 'lane graph mirrors outgoing connections');
  assert(net.lanes[1].incomingConnections.length === 1, 'lane graph mirrors incoming connections');
  assert(net.lanes[0].outgoingConnections[0].priority === PriorityType.DIRECT, 'connections default to direct priority');
  assert(laneConnectionFor(net, 0, 0) === net.exit[0][0][0], 'laneConnectionFor resolves route from global lane');
  assert(laneRouteConnections(net, 0) === net.exit[0][0], 'laneRouteConnections exposes the route table by global lane');
  assert(laneHasRouteTable(net, 0) && !laneHasRouteTable(net, 1), 'laneHasRouteTable distinguishes connected and open ends');
  assert(availableRouteIndices(net, 0)[0] === 0, 'availableRouteIndices exposes usable route slots');
  assert(connectionTargetLane(net, net.lanes[0].outgoingConnections[0]) === 1, 'connectionTargetLane resolves global target');
  const car = [newCar(0, 12, 0)];
  let crossV = -1;
  for (let step = 0; step < 14 * 60; step++) {
    stepNetwork(net, car, [slow], CAR_LENGTH, 1 / 60);
    if (car[0].lane === 1 && crossV < 0) crossV = car[0].v;
  }
  assert(car[0].lane === 1, `car is on road B (lane ${car[0].lane})`);
  assert(crossV > 11, `no slowdown at the seam (crossed at v=${crossV.toFixed(2)})`);
  assert(car[0].s > 10 && car[0].s < 25, `position carried across (s=${car[0].s.toFixed(1)} on B)`);
}

// Seam: the stop at the far end propagates upstream — a car on A queues behind one stopped on B.
{
  const a = new Road({ shape: 'straight', length: 150, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 0 });
  const b = new Road({ shape: 'straight', length: 30, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 0 });
  const net = buildNetwork([a, b], [[0, 1]]);
  const queue = [newCar(0, 20, 0), newCar(25, 10, 1)]; // car 1 starts on B near its end
  for (let step = 0; step < 30 * 60; step++) stepNetwork(net, queue, [fast, slow], CAR_LENGTH, 1 / 60);
  assert(queue[0].v < 0.01 && queue[1].v < 0.01, 'both cars stopped');
  const seamGap = (150 - queue[0].s) + queue[1].s - CAR_LENGTH;
  assert(queue[1].lane === 1 && seamGap > 0, `no collision across the seam (gap ${seamGap.toFixed(2)} m)`);
}

// Seam, two-way: a backward car crosses from B into A's backward lane.
{
  const a = new Road({ shape: 'straight', length: 150, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 1 });
  const b = new Road({ shape: 'straight', length: 150, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 1 });
  const net = buildNetwork([a, b], [[0, 1]]);
  const car = [newCar(140, 20, 3)]; // B's backward lane (global lane 3)
  let crossV = -1;
  for (let step = 0; step < 12 * 60; step++) {
    stepNetwork(net, car, [fast], CAR_LENGTH, 1 / 60);
    if (car[0].lane === 1 && crossV < 0) crossV = car[0].v;
  }
  assert(car[0].lane === 1, `car crossed into A's backward lane (lane ${car[0].lane})`);
  assert(crossV > 18, `no slowdown at the seam (crossed at v=${crossV.toFixed(2)})`);
}

console.log('Network checks passed');

// ---------------------------------------------------------------------------
// Intersection (scene 4 building block)
// ---------------------------------------------------------------------------

// Phase machine: complementary pairs, cycle order and override mapping.
{
  const light = { green: 10, yellow: 2, red: 8, override: 'auto' };
  assert(intersectionPhaseAt(0, light) === 'nsGreen', 'cycle starts nsGreen');
  assert(intersectionPhaseAt(11, light) === 'nsYellow', 'nsYellow after green');
  assert(intersectionPhaseAt(13, light) === 'ewGreen', 'ewGreen right after — pairs complement');
  assert(intersectionPhaseAt(23, light) === 'ewYellow', 'ewYellow next');
  assert(intersectionPhaseAt(24, light) === 'nsGreen', 'cycle wraps');
  assert(intersectionPhaseAt(0, { ...light, override: 'red' }) === 'allRed', 'override red');
  assert(intersectionPhaseAt(0, { ...light, override: 'green' }) === 'nsGreen', 'override green');
}

const OPEN = { n: 'open', e: 'open', s: 'open', w: 'open' } as const;
const intersectionOf = (
  closed: IntersectionConfig['closed'] = { ...OPEN },
  lanesEachWay = 1,
): IntersectionState => buildIntersection({ approach: 80, lanesEachWay, closed });

// nsGreen: a car on the S approach crosses the zone onto the N road.
{
  const state = intersectionOf();
  const car = [newCar(0, 12, 0)]; // S forward lane
  const obstacles = intersectionObstacles(state, 'nsGreen');
  let crossV = -1;
  for (let step = 0; step < 15 * 60; step++) {
    stepNetwork(state.net, car, [slow], CAR_LENGTH, 1 / 60, obstacles);
    if (crossV < 0 && laneNode(state.net, car[0].lane).road === state.roadIndex.n) crossV = car[0].v;
  }
  assert(laneNode(state.net, car[0].lane).road === state.roadIndex.n, `car crossed onto the N road (road ${laneNode(state.net, car[0].lane).road})`);
  assert(crossV > 8, `car flowed through the zone (crossed at v=${crossV.toFixed(1)})`);
}

// allRed: a car on the S approach stops before its stop line.
{
  const state = intersectionOf();
  const car = [newCar(0, 15, 0)];
  const obstacles = intersectionObstacles(state, 'allRed');
  for (let step = 0; step < 15 * 60; step++) {
    stepNetwork(state.net, car, [slow], CAR_LENGTH, 1 / 60, obstacles);
  }
  assert(car[0].v < 0.01, `stopped at the red (v=${car[0].v.toFixed(3)})`);
  assert(car[0].s > 68 && car[0].s < 74, `stopped at the line (s=${car[0].s.toFixed(2)})`);
}

// Conflict safety: while NS flows, the W approach car must wait at its line — no collision in the zone.
{
  const state = intersectionOf();
  const wFwd = state.net.laneOffsets[state.roadIndex.w]; // W forward lane (global)
  const cars = [newCar(0, 12, 0), newCar(0, 12, wFwd)];
  const obstacles = intersectionObstacles(state, 'nsGreen');
  for (let step = 0; step < 15 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, obstacles);
  }
  assert(laneNode(state.net, cars[0].lane).road === state.roadIndex.n, 'S car crossed');
  assert(cars[1].v < 0.01 && cars[1].s < 76, `W car held at its line (s=${cars[1].s.toFixed(1)}, v=${cars[1].v.toFixed(2)})`);
}

console.log('Intersection checks passed');

// ---------------------------------------------------------------------------
// Turns at the intersection
// ---------------------------------------------------------------------------

const signalWithYield = (state: IntersectionState, cars: Car[]): NetObstacle[] => [
  ...intersectionObstacles(state, 'nsGreen'),
  ...leftTurnYieldObstacles(state, cars),
];

// Right turn on green: an S car with route=right ends up on the E road.
{
  const state = intersectionOf();
  const car = [newCar(0, 12, 0, 1)];
  for (let step = 0; step < 20 * 60; step++) {
    stepNetwork(state.net, car, [slow], CAR_LENGTH, 1 / 60, signalWithYield(state, car));
  }
  assert(laneNode(state.net, car[0].lane).road === state.roadIndex.w, `S car turned right onto W (road ${laneNode(state.net, car[0].lane).road})`);
}

// Right turn does NOT yield to opposing traffic.
{
  const state = intersectionOf();
  const nBack = state.net.laneOffsets[state.roadIndex.n] + 1; // N backward lane (global)
  const cars = [newCar(0, 12, 0, 1), newCar(30, 12, nBack, 0)]; // S-right + N backward approaching
  let minV = Infinity;
  for (let step = 0; step < 8 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, signalWithYield(state, cars));
    if (laneNode(state.net, cars[0].lane).road === state.roadIndex.s) minV = Math.min(minV, cars[0].v);
  }
  assert(minV > 10, `right turn never yielded (min approach speed ${minV.toFixed(2)})`);
}

// Left turn yields: the S car waits at the line until the opposing car has passed, then turns onto E.
{
  const state = intersectionOf();
  const nBack = state.net.laneOffsets[state.roadIndex.n] + 1;
  const cars = [newCar(60, 12, 0, 2), newCar(30, 12, nBack, 0)]; // S-left + N backward opposing (ETA ~2.5 s)
  const stopLine = 80 - STOP_BACK;
  for (let step = 0; step < 2 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, signalWithYield(state, cars));
  }
  assert(cars[0].s <= stopLine + 0.01, `left turn held at the line while opposing passes (s=${cars[0].s.toFixed(1)})`);
  for (let step = 0; step < 28 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, signalWithYield(state, cars));
  }
  assert(laneNode(state.net, cars[0].lane).road === state.roadIndex.e, `S car turned left onto E after yielding (road ${laneNode(state.net, cars[0].lane).road})`);
}

// Gap acceptance: a far opponent (ETA > 4.5 s) does not hold the turn.
{
  const state = intersectionOf();
  const nBack = state.net.laneOffsets[state.roadIndex.n] + 1;
  const cars = [newCar(60, 12, 0, 2), newCar(75, 12, nBack, 0)]; // opponent 75 m out, ETA ~6 s
  let crossed = false;
  for (let step = 0; step < 4 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, signalWithYield(state, cars));
    if (laneNode(state.net, cars[0].lane).road !== state.roadIndex.s) crossed = true;
  }
  assert(crossed, 'left turn went through a safe gap');
}

// Gap acceptance: a stopped opponent near the zone does not hold the turn.
{
  const state = intersectionOf();
  const nBack = state.net.laneOffsets[state.roadIndex.n] + 1;
  const cars = [newCar(60, 12, 0, 2), newCar(10, 0, nBack, 0)]; // opponent parked, v = 0
  let crossed = false;
  for (let step = 0; step < 6 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, signalWithYield(state, cars));
    if (laneNode(state.net, cars[0].lane).road !== state.roadIndex.s) crossed = true;
  }
  assert(crossed, 'left turn went past a stopped opponent');
}

console.log('Turn checks passed');

// ---------------------------------------------------------------------------
// Turn-arc geometry: every arc's end pose must equal the solver's exit pose, for
// any lane count and driving side.
// ---------------------------------------------------------------------------
{
  const MOVES: [Way, Way, 'right' | 'left'][] = [
    ['s', 'w', 'right'],
    ['s', 'e', 'left'],
    ['n', 'e', 'right'],
    ['n', 'w', 'left'],
    ['e', 's', 'right'],
    ['e', 'n', 'left'],
    ['w', 'n', 'right'],
    ['w', 's', 'left'],
  ];
  for (const lanesEachWay of [1, 2]) {
    for (const handed of [1, -1]) {
      buildScene4({ approach: 80, lanesEachWay, closed: { n: 'open', e: 'open', s: 'open', w: 'open' } }, handed);
      const def = SCENES[3];
      const net = scene4State.state!.net;
      for (const [from, to, kind] of MOVES) {
        const spec = turnArcSpec(from, to, kind, lanesEachWay, handed);
        const key = `${from}${kind === 'right' ? 'Right' : 'Left'}`;
        const roadIdx = scene4State.state!.roadIndex[key];
        const g = net.laneOffsets[roadIdx];
        const car: Car = {
          s: net.roads[roadIdx].length,
          v: 10,
          a: 0,
          lane: g,
          route: 0,
          lateral: g,
          lateralVel: 0,
          laneFrom: g,
          laneProgress: 1,
          cooldown: 0,
        };
        const pose = def.carPose(car);
        assert(
          Math.abs(pose.x - spec.exit.x) < 0.75 && Math.abs(pose.z - spec.exit.z) < 0.75,
          `${key} ends at the matching lane (lanes ${lanesEachWay}, handed ${handed}; got ${pose.x.toFixed(1)},${pose.z.toFixed(1)} want ${spec.exit.x.toFixed(1)},${spec.exit.z.toFixed(1)})`,
        );
        assert(
          Math.abs(Math.cos(pose.angle) - spec.exit.hx) < 0.2 && Math.abs(-Math.sin(pose.angle) - spec.exit.hz) < 0.2,
          `${key} exits with the right heading (lanes ${lanesEachWay}, handed ${handed})`,
        );
        // The sim connection must land on the matching lane index too.
        const conn = laneConnectionFor(net, g, 0);
        assert(conn !== null && conn.toLane === spec.dstLane, `${key} connects to the matching lane index`);
      }
    }
  }
  console.log('Turn geometry checks passed');
}

// ---------------------------------------------------------------------------
// Way closures: T (one way fully closed), L (two adjacent closed), one-way states
// ---------------------------------------------------------------------------

// T intersection: N fully closed — no N road, no straight from S, turns still work.
{
  const state = intersectionOf({ n: 'both', e: 'open', s: 'open', w: 'open' });
  assert(state.roadIndex.n === undefined, 'N road is not built');
  assert(state.roadIndex.nsConn === undefined, 'NS connector dropped when N is closed');
  const sFwd = state.net.laneOffsets[state.roadIndex.s];
  assert(!laneConnectionFor(state.net, sFwd, 0), 'S straight route is unavailable');

  const car = [newCar(0, 12, sFwd, 2)]; // S-left still flows
  for (let step = 0; step < 20 * 60; step++) {
    stepNetwork(state.net, car, [slow], CAR_LENGTH, 1 / 60, signalWithYield(state, car));
  }
  assert(laneNode(state.net, car[0].lane).road === state.roadIndex.e, 'S turns left onto E in a T');
}

// L corner: N and W fully closed — the surviving movements are S-left → E and E-right → S.
{
  const state = intersectionOf({ n: 'both', e: 'open', s: 'open', w: 'both' });
  assert(state.roadIndex.n === undefined && state.roadIndex.w === undefined, 'N and W not built');
  const sFwd = state.net.laneOffsets[state.roadIndex.s];
  const eBack = state.net.laneOffsets[state.roadIndex.e] + 1;
  const cars = [newCar(0, 12, sFwd, 2), newCar(20, 12, eBack, 1)];
  for (let step = 0; step < 25 * 60; step++) {
    // No signal needed here: with N/W closed there is no opposing stream to yield to.
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, leftTurnYieldObstacles(state, cars));
  }
  assert(laneNode(state.net, cars[0].lane).road === state.roadIndex.e, 'S turns left onto E in an L');
  assert(laneNode(state.net, cars[1].lane).road === state.roadIndex.s, 'E turns right onto S in an L');
}

// Multi-lane L corner: when straight and one side are closed, every entering lane
// gets a lane-matched forced turn instead of parking at the zone edge.
for (const lanesEachWay of [2, 3]) {
  const state = intersectionOf({ n: 'both', e: 'open', s: 'open', w: 'both' }, lanesEachWay);
  const sLanes = Array.from({ length: lanesEachWay }, (_, li) => state.net.laneOffsets[state.roadIndex.s] + li);
  const eLanes = Array.from({ length: lanesEachWay }, (_, i) => {
    const li = lanesEachWay + i;
    return state.net.laneOffsets[state.roadIndex.e] + li;
  });
  for (const [i, g] of sLanes.entries()) {
    const conn = laneConnectionFor(state.net, g, 2);
    assert(conn !== null && conn.toRoad !== state.roadIndex.nsConn, `S lane ${i} has a forced left turn (${lanesEachWay} lanes)`);
    const car = [newCar(0, 12, g, 2)];
    for (let step = 0; step < 25 * 60; step++) {
      stepNetwork(state.net, car, [slow], CAR_LENGTH, 1 / 60, leftTurnYieldObstacles(state, car));
    }
    const lane = laneNode(state.net, car[0].lane);
    assert(lane.road === state.roadIndex.e && lane.lane === i, `S lane ${i} turns into matching E lane (${lanesEachWay} lanes)`);
  }
  for (const [i, g] of eLanes.entries()) {
    const conn = laneConnectionFor(state.net, g, 1);
    assert(conn !== null && conn.toRoad !== state.roadIndex.ewConn, `E lane ${i} has a forced right turn (${lanesEachWay} lanes)`);
    const car = [newCar(20, 12, g, 1)];
    for (let step = 0; step < 25 * 60; step++) {
      stepNetwork(state.net, car, [slow], CAR_LENGTH, 1 / 60, leftTurnYieldObstacles(state, car));
    }
    const lane = laneNode(state.net, car[0].lane);
    assert(lane.road === state.roadIndex.s && lane.lane === i + lanesEachWay, `E lane ${i} turns into matching S lane (${lanesEachWay} lanes)`);
  }
}

// One-way-in: S exit closed — S traffic enters and crosses, but nothing flows back into S.
{
  const state = intersectionOf({ n: 'open', e: 'open', s: 'out', w: 'open' });
  const sFwd = state.net.laneOffsets[state.roadIndex.s];
  const car = [newCar(0, 12, sFwd, 0)];
  for (let step = 0; step < 15 * 60; step++) {
    stepNetwork(state.net, car, [slow], CAR_LENGTH, 1 / 60, signalWithYield(state, car));
  }
  assert(laneNode(state.net, car[0].lane).road === state.roadIndex.n, 'S traffic still crosses one-way-in');
  // No connection anywhere targets the S road (exit side closed).
  const targetsS = state.net.lanes.some((lane) =>
    lane.outgoingConnections.some((conn) => conn.toRoad === state.roadIndex.s),
  );
  assert(!targetsS, 'no traffic flows into S when its exit is closed');
}

// Exit-closed far side kills the straight route: with N exit closed, S must turn —
// nobody may park in the zone.
{
  const state = intersectionOf({ n: 'out', e: 'open', s: 'open', w: 'open' });
  const sFwd = state.net.laneOffsets[state.roadIndex.s];
  assert(!laneConnectionFor(state.net, sFwd, 0), 'S straight route unavailable when N exit is closed');
  // N backward straight still flows (into S, which is open).
  const nBack = state.net.laneOffsets[state.roadIndex.n] + 1;
  const cars = [newCar(0, 12, sFwd, 1), newCar(30, 12, nBack, 0)];
  for (let step = 0; step < 20 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, signalWithYield(state, cars));
  }
  assert(laneNode(state.net, cars[0].lane).road === state.roadIndex.w, 'S turns right onto W');
  assert(laneNode(state.net, cars[1].lane).road === state.roadIndex.s, 'N straight still reaches S');
}

// One-way-out: W entry closed — no spawns on W's entering lanes, entry dead-ends.
{
  const state = intersectionOf({ n: 'open', e: 'open', s: 'open', w: 'in' });
  const wFwd = state.net.laneOffsets[state.roadIndex.w];
  assert(state.net.closedLanes.has(wFwd), 'W entering lane is spawn-blocked');
  assert(!laneHasRouteTable(state.net, wFwd), 'W entry dead-ends at the zone');
}

console.log('Closure checks passed');

// ---------------------------------------------------------------------------
// Lane changes near intersections (route meaning is lane-indexed)
// ---------------------------------------------------------------------------

// No lane changes within 40 m of the lane end, even to pass a slow leader.
{
  const road = new Road({ shape: 'straight', length: 120, radius: 50, angle: 90, lanesForward: 2, lanesBackward: 0 });
  const net = netOf(road);
  const cars = [newCar(55, 8, 0), newCar(0, 30, 0)]; // slow leader, fast follower on lane 0
  const slowpokes: IdmParams = { ...slow, v0: 8 };
  let lateChange = false;
  for (let step = 0; step < 30 * 60; step++) {
    const before = cars[1].lane;
    stepNetwork(net, cars, [slowpokes, fast], CAR_LENGTH, 1 / 60);
    if (cars[1].s > 40 && cars[1].lane !== before) lateChange = true;
  }
  assert(!lateChange, 'no lane changes within 40 m of the lane end');
}

// The T-intersection trap: a right-routed car baited into the inner lane must not
// park at the zone edge — it turns (or takes the remapped route) instead.
{
  const state = intersectionOf({ n: 'both', e: 'open', s: 'open', w: 'open' }, 2);
  const sOuter = state.net.laneOffsets[state.roadIndex.s] + 1; // S forward outer lane (of 2)
  const slowpokes: IdmParams = { ...slow, v0: 8 };
  const cars = [newCar(10, 25, sOuter, 1), newCar(35, 8, sOuter, 1)]; // fast right-turner behind a slow car
  for (let step = 0; step < 45 * 60; step++) {
    stepNetwork(state.net, cars, [fast, slowpokes], CAR_LENGTH, 1 / 60, signalWithYield(state, cars));
  }
  const { road } = laneNode(state.net, cars[0].lane);
  const parkedAtEdge = road === state.roadIndex.s && cars[0].s > 74 && cars[0].v < 0.01;
  assert(!parkedAtEdge, `turning car did not park at the zone edge (road ${road}, s=${cars[0].s.toFixed(1)})`);
}

console.log('Lane-commit checks passed');
