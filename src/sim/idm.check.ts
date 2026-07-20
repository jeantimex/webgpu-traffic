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

// Ring: fast car catches a slower leader, must not collide, must settle near leader speed.
const C = 2 * Math.PI * 40;
const CAR_LENGTH = 4.5;
const cars: Car[] = [
  { s: 0, v: 30, a: 0 },
  { s: C / 2, v: 12, a: 0 },
];
let minGap = Infinity;
for (let step = 0; step < 120 * 60; step++) {
  stepRing(cars, [fast, slow], C, CAR_LENGTH, 1 / 60);
  const gap = ((((cars[1].s - cars[0].s) % C) + C) % C) - CAR_LENGTH;
  minGap = Math.min(minGap, gap);
}
assert(minGap > 0, `no collision (min gap ${minGap.toFixed(2)} m)`);
assert(Math.abs(cars[0].v - slow.v0) < 0.2, `fast car settles to leader speed (v=${cars[0].v.toFixed(2)})`);

console.log(`IDM checks passed (min gap ${minGap.toFixed(2)} m, settled at ${cars[0].v.toFixed(2)} m/s)`);
