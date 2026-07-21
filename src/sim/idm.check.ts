/**
 * Runnable self-check for the IDM sim. Not part of the app bundle.
 * Run: node_modules/.bin/esbuild src/sim/idm.check.ts --bundle --format=esm --outfile=.idm.check.mjs && node .idm.check.mjs && rm .idm.check.mjs
 */
import { idmAcceleration, stepRing, type Car, type IdmParams } from './idm';
import {
  buildScene4,
  SCENES,
  scene4State,
} from '../renderer/scenes';
import {
  buildIntersection,
  IDX,
  intersectionObstacles,
  intersectionPhaseAt,
  leftTurnYieldObstacles,
  STOP_BACK,
  type IntersectionState,
} from './intersection';
import { buildNetwork, locate, stepNetwork, type NetObstacle, type Network } from './network';
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
  stepRing(cars, [fast, slow], C, CAR_LENGTH, 1 / 60);
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
for (let step = 0; step < 60 * 60; step++) stepRing(lone, [fast], C, CAR_LENGTH, 1 / 60, redLight);
const distToLine = (((C / 4 - lone[0].s) % C) + C) % C;
assert(lone[0].v < 0.01, `stopped at the red light (v=${lone[0].v.toFixed(3)})`);
assert(
  distToLine > CAR_LENGTH / 2 && distToLine < 8,
  `stopped just before the line (${distToLine.toFixed(2)} m from center)`);
for (let step = 0; step < 10 * 60; step++) stepRing(lone, [fast], C, CAR_LENGTH, 1 / 60);
assert(lone[0].v > 10, `accelerates on green (v=${lone[0].v.toFixed(1)})`);

// Different lanes: no car-following interaction — the fast car keeps its desired speed.
const twoLanes: Car[] = [
  { s: C / 2 + 30, v: 12, a: 0, lane: 0, route: 0, lateral: 0, lateralVel: 0, laneFrom: 0, laneProgress: 1, cooldown: 0 }, // slow, ahead in the inner lane
  { s: C / 2, v: 30, a: 0, lane: 1, route: 0, lateral: 1, lateralVel: 0, laneFrom: 1, laneProgress: 1, cooldown: 0 }, // fast, catching up in the outer lane
];
for (let step = 0; step < 60 * 60; step++) stepRing(twoLanes, [slow, fast], C, CAR_LENGTH, 1 / 60);
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

const intersectionOf = (): IntersectionState => buildIntersection({ approach: 80, lanesEachWay: 1 });

// nsGreen: a car on the S approach crosses the zone onto the N road.
{
  const state = intersectionOf();
  const car = [newCar(0, 12, 0)]; // S forward lane
  const obstacles = intersectionObstacles(state, 'nsGreen');
  let crossV = -1;
  for (let step = 0; step < 15 * 60; step++) {
    stepNetwork(state.net, car, [slow], CAR_LENGTH, 1 / 60, obstacles);
    if (crossV < 0 && locate(state.net, car[0].lane).road === IDX.n) crossV = car[0].v;
  }
  assert(locate(state.net, car[0].lane).road === IDX.n, `car crossed onto the N road (road ${locate(state.net, car[0].lane).road})`);
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
  // laneOffsets: 2 lanes per road → S fwd = 0, W fwd = 6.
  const cars = [newCar(0, 12, 0), newCar(0, 12, 6)];
  const obstacles = intersectionObstacles(state, 'nsGreen');
  for (let step = 0; step < 15 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, obstacles);
  }
  assert(locate(state.net, cars[0].lane).road === IDX.n, 'S car crossed');
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
  assert(locate(state.net, car[0].lane).road === IDX.w, `S car turned right onto W (road ${locate(state.net, car[0].lane).road})`);
}

// Right turn does NOT yield to opposing traffic.
{
  const state = intersectionOf();
  const cars = [newCar(0, 12, 0, 1), newCar(30, 12, 3, 0)]; // S-right + N backward approaching
  let minV = Infinity;
  for (let step = 0; step < 8 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, signalWithYield(state, cars));
    if (locate(state.net, cars[0].lane).road === IDX.s) minV = Math.min(minV, cars[0].v);
  }
  assert(minV > 10, `right turn never yielded (min approach speed ${minV.toFixed(2)})`);
}

// Left turn yields: the S car waits at the line until the opposing car has passed, then turns onto E.
{
  const state = intersectionOf();
  const cars = [newCar(60, 12, 0, 2), newCar(30, 12, 3, 0)]; // S-left + N backward opposing
  const stopLine = 80 - STOP_BACK;
  for (let step = 0; step < 2 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, signalWithYield(state, cars));
  }
  assert(cars[0].s <= stopLine + 0.01, `left turn held at the line while opposing passes (s=${cars[0].s.toFixed(1)})`);
  for (let step = 0; step < 28 * 60; step++) {
    stepNetwork(state.net, cars, [slow, slow], CAR_LENGTH, 1 / 60, signalWithYield(state, cars));
  }
  assert(locate(state.net, cars[0].lane).road === IDX.e, `S car turned left onto E after yielding (road ${locate(state.net, cars[0].lane).road})`);
}

console.log('Turn checks passed');

// ---------------------------------------------------------------------------
// Turn-arc geometry: every arc's end pose must land on its exit lane, for both
// driving sides (the bug the topology checks could not see).
// ---------------------------------------------------------------------------
{
  // End poses per driving side (turn lanes mirror across each road's centerline).
  const tables: Record<number, [number, number, number, number][]> = {
    '1': [
      [-4, -2, -1, 0], // S right → W
      [4, 2, 1, 0], // S left → E
      [4, 2, 1, 0], // N right → E
      [-4, -2, -1, 0], // N left → W
      [2, -4, 0, -1], // E right → S
      [-2, 4, 0, 1], // E left → N
      [-2, 4, 0, 1], // W right → N
      [2, -4, 0, -1], // W left → S
    ],
    '-1': [
      [-4, 2, -1, 0], // S right → W
      [4, -2, 1, 0], // S left → E
      [4, -2, 1, 0], // N right → E
      [-4, 2, -1, 0], // N left → W
      [-2, -4, 0, -1], // E right → S
      [2, 4, 0, 1], // E left → N
      [2, 4, 0, 1], // W right → N
      [-2, -4, 0, -1], // W left → S
    ],
  };
  for (const handed of [1, -1]) {
    buildScene4({ approach: 80, lanesEachWay: 1 }, handed);
    const def = SCENES[3];
    const net = scene4State.state!.net;
    const expected = handed === 1 ? tables[1] : tables[-1];
    for (let k = 0; k < 8; k++) {
      const roadIdx = 6 + k;
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
      const [ex, ez, ehx, ehz] = expected[k];
      assert(
        Math.abs(pose.x - ex) < 1 && Math.abs(pose.z - ez) < 1,
        `arc ${roadIdx} ends at its exit lane (handed ${handed}, got ${pose.x.toFixed(1)}, ${pose.z.toFixed(1)})`,
      );
      assert(
        Math.abs(Math.cos(pose.angle) - ehx) < 0.2 && Math.abs(-Math.sin(pose.angle) - ehz) < 0.2,
        `arc ${roadIdx} exits with the right heading (handed ${handed})`,
      );
    }
  }
  console.log('Turn geometry checks passed');
}
