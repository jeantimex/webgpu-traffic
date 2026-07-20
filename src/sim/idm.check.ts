/**
 * Runnable self-check for the IDM sim. Not part of the app bundle.
 * Run: node_modules/.bin/esbuild src/sim/idm.check.ts --bundle --format=esm --outfile=.idm.check.mjs && node .idm.check.mjs && rm .idm.check.mjs
 */
import { idmAcceleration, stepRing, type Car, type IdmParams } from './idm';

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
