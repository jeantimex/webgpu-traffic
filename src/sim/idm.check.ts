/**
 * Runnable self-check for the IDM sim. Not part of the app bundle.
 * Run: node_modules/.bin/esbuild src/sim/idm.check.ts --bundle --format=esm --outfile=.idm.check.mjs && node .idm.check.mjs && rm .idm.check.mjs
 */
import { idmAcceleration, stepRing, type Car, type IdmParams } from './idm';
import { Road, stepRoad } from './road';

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
  { s: 0, v: 30, a: 0, lane: 0, lateral: 0, lateralVel: 0, laneFrom: 0, laneProgress: 1, cooldown: 0 },
  { s: C / 2, v: 12, a: 0, lane: 0, lateral: 0, lateralVel: 0, laneFrom: 0, laneProgress: 1, cooldown: 0 },
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
const lone: Car[] = [{ s: 0, v: 20, a: 0, lane: 0, lateral: 0, lateralVel: 0, laneFrom: 0, laneProgress: 1, cooldown: 0 }];
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
  { s: C / 2 + 30, v: 12, a: 0, lane: 0, lateral: 0, lateralVel: 0, laneFrom: 0, laneProgress: 1, cooldown: 0 }, // slow, ahead in the inner lane
  { s: C / 2, v: 30, a: 0, lane: 1, lateral: 1, lateralVel: 0, laneFrom: 1, laneProgress: 1, cooldown: 0 }, // fast, catching up in the outer lane
];
for (let step = 0; step < 60 * 60; step++) stepRing(twoLanes, [slow, fast], C, CAR_LENGTH, 1 / 60);
assert(
  Math.abs(twoLanes[1].v - fast.v0) < 0.5,
  `fast car ignores the slow car in the other lane (v=${twoLanes[1].v.toFixed(2)})`,
);

console.log(`IDM checks passed (min gap ${minGap.toFixed(2)} m, settled at ${cars[0].v.toFixed(2)} m/s)`);

// ---------------------------------------------------------------------------
// Road (scene 3 building block)
// ---------------------------------------------------------------------------

const newCar = (s: number, v: number, lane: number): Car => ({
  s,
  v,
  a: 0,
  lane,
  lateral: lane,
  lateralVel: 0,
  laneFrom: lane,
  laneProgress: 1,
  cooldown: 0,
});

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

// Open end is a full stop: a car must stop just before the road end.
{
  const road = new Road({ shape: 'straight', length: 120, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 0 });
  const car = [newCar(0, 20, 0)];
  for (let step = 0; step < 30 * 60; step++) stepRoad(road, car, [fast], CAR_LENGTH, 1 / 60);
  assert(car[0].v < 0.01, `stopped at road end (v=${car[0].v.toFixed(3)})`);
  assert(car[0].s > 112 && car[0].s < 118, `stopped just before the end (s=${car[0].s.toFixed(2)})`);
}

// One lane, no way around: fast car settles behind the slow one (mean speed over the last 10 s).
{
  const road = new Road({ shape: 'straight', length: 1500, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 0 });
  const platoon = [newCar(0, 30, 0), newCar(150, 12, 0)];
  let vSum = 0;
  let vN = 0;
  for (let step = 0; step < 90 * 60; step++) {
    stepRoad(road, platoon, [fast, slow], CAR_LENGTH, 1 / 60);
    if (step >= 80 * 60) {
      vSum += platoon[0].v;
      vN++;
    }
  }
  assert(Math.abs(vSum / vN - slow.v0) < 0.5, `platoon settled (mean v=${(vSum / vN).toFixed(2)})`);
}

// Two lanes one-way: fast car changes lanes to overtake.
{
  const road = new Road({ shape: 'straight', length: 400, radius: 50, angle: 90, lanesForward: 2, lanesBackward: 0 });
  const race = [newCar(0, 30, 0), newCar(150, 12, 0)];
  let changed = false;
  let maxV = 0;
  for (let step = 0; step < 12 * 60; step++) {
    stepRoad(road, race, [fast, slow], CAR_LENGTH, 1 / 60);
    if (race[0].lane !== 0) changed = true;
    maxV = Math.max(maxV, race[0].v);
  }
  assert(changed, 'fast car changed lanes to overtake on the road');
  assert(maxV > 25, `fast car was not stuck (max v=${maxV.toFixed(2)})`);
}

// Two-way: opposing traffic does not interact.
{
  const p20: IdmParams = { ...fast, v0: 20 };
  const road = new Road({ shape: 'straight', length: 1000, radius: 50, angle: 90, lanesForward: 1, lanesBackward: 1 });
  const both = [newCar(50, 20, 0), newCar(950, 20, 1)];
  let maxDev = 0;
  for (let step = 0; step < 22 * 60; step++) {
    stepRoad(road, both, [p20, p20], CAR_LENGTH, 1 / 60);
    maxDev = Math.max(maxDev, Math.abs(both[0].v - 20), Math.abs(both[1].v - 20));
  }
  assert(maxDev < 0.5, `opposing cars unaffected (max deviation ${maxDev.toFixed(2)} m/s)`);
}

console.log('Road checks passed');
