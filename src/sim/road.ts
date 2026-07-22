/**
 * Road: the foundation building block of the traffic system. A road is an open
 * (non-looping) segment with a centerline path of any shape — straight, arc, or
 * S-curve — carrying N forward lanes and M backward lanes (M = 0 for one-way).
 * The path's offset 0 is the yellow line separating the directions.
 *
 * Geometry only: the sim lives in network.ts, where unconnected road ends act as
 * permanent stop signs.
 */

export interface PathPoint {
  x: number;
  z: number;
  hx: number; // heading (unit vector)
  hz: number;
  rx: number; // right normal of the heading (unit vector)
  rz: number;
}

export type RoadShape = 'straight' | 'arc' | 'scurve';

export interface RoadConfig {
  shape: RoadShape;
  length: number; // straight only (m)
  radius: number; // arc & scurve (m)
  angle: number; // arc & scurve (degrees; per arc for scurve; negative = turn right)
  lanesForward: number; // 1..3
  lanesBackward: number; // 0..3 (0 = one-way)
}

export interface Lane {
  direction: 1 | -1;
  offset: number; // signed lateral offset from the path, + = right of forward heading
  leftNeighbor: number | null; // local lane index, null when no legal lane change exists
  rightNeighbor: number | null; // local lane index, null when no legal lane change exists
  point(s: number): PathPoint; // lane centerline point at arc position s
}

const LANE_WIDTH = 4;

function makeLane(direction: 1 | -1, offset: number): Lane {
  return {
    direction,
    offset,
    leftNeighbor: null,
    rightNeighbor: null,
    point: () => {
      throw new Error('Lane geometry is not initialized.');
    },
  };
}

function offsetPoint(p: PathPoint, offset: number): PathPoint {
  return {
    x: p.x + p.rx * offset,
    z: p.z + p.rz * offset,
    hx: p.hx,
    hz: p.hz,
    rx: p.rx,
    rz: p.rz,
  };
}

function straightPath(length: number): (s: number) => PathPoint {
  return (s) => ({ x: s - length / 2, z: 0, hx: 1, hz: 0, rx: 0, rz: 1 });
}

const MIN_ANGLE_DEG = 5; // a zero-length road is degenerate

/** Signed-angle helper: magnitude of the sweep (radians) and the turn direction mirror. */
function sweep(angleDeg: number): { theta: number; mirror: number } {
  const clamped = Math.max(MIN_ANGLE_DEG, Math.abs(angleDeg));
  return { theta: (clamped * Math.PI) / 180, mirror: angleDeg < 0 ? -1 : 1 };
}

/** Circular arc of `radius` sweeping `angleDeg` (negative = right turn), centered in view. */
function arcPath(radius: number, angleDeg: number): (s: number) => PathPoint {
  const { theta, mirror } = sweep(angleDeg);
  const zOffset = (radius * (1 - Math.cos(theta / 2))) / 2;
  return (s) => {
    const phi = s / radius - theta / 2;
    const hx = Math.cos(phi);
    const hz = mirror * Math.sin(phi);
    return {
      x: radius * Math.sin(phi),
      z: mirror * (radius * (1 - Math.cos(phi)) - zOffset),
      hx,
      hz,
      rx: -hz,
      rz: hx,
    };
  };
}

/** S-curve: an arc of `angleDeg` followed by an equal opposite arc, tangent-continuous. */
function scurvePath(radius: number, angleDeg: number): (s: number) => PathPoint {
  const { theta, mirror } = sweep(angleDeg);
  const leg = radius * theta; // arc length of one side
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  // End of the first arc.
  const p1x = radius * sinT;
  const p1z = radius * (1 - cosT);
  // Center the whole S in view.
  const cx = p1x;
  const cz = p1z;
  return (s) => {
    let hx: number;
    let hz: number;
    let x: number;
    let z: number;
    if (s <= leg) {
      const phi = s / radius;
      x = radius * Math.sin(phi);
      z = radius * (1 - Math.cos(phi));
      hx = Math.cos(phi);
      hz = Math.sin(phi);
    } else {
      const psi = (s - leg) / radius;
      // Second arc turns right back to the original heading (derived by rotating the
      // first arc's local frame by theta at P1).
      const lx = radius * Math.sin(psi);
      const lz = -radius * (1 - Math.cos(psi));
      x = p1x + lx * cosT - lz * sinT;
      z = p1z + lx * sinT + lz * cosT;
      hx = Math.cos(theta - psi);
      hz = Math.sin(theta - psi);
    }
    hz *= mirror;
    return { x: x - cx, z: mirror * (z - cz), hx, hz, rx: -hz, rz: hx };
  };
}

export class Road {
  readonly lanes: Lane[] = [];
  readonly length: number;
  private readonly path: (s: number) => PathPoint;

  constructor(
    readonly config: RoadConfig,
    handed = 1, // 1 = right-hand traffic, -1 = left-hand traffic (mirrors lane sides)
  ) {
    switch (config.shape) {
      case 'straight':
        this.length = config.length;
        this.path = straightPath(this.length);
        break;
      case 'arc':
        this.length = (config.radius * Math.max(MIN_ANGLE_DEG, Math.abs(config.angle)) * Math.PI) / 180;
        this.path = arcPath(config.radius, config.angle);
        break;
      case 'scurve':
        this.length = (2 * config.radius * Math.max(MIN_ANGLE_DEG, Math.abs(config.angle)) * Math.PI) / 180;
        this.path = scurvePath(config.radius, config.angle);
        break;
    }
    for (let i = 0; i < config.lanesForward; i++) {
      this.lanes.push(makeLane(1, (LANE_WIDTH / 2 + i * LANE_WIDTH) * handed));
    }
    for (let j = 0; j < config.lanesBackward; j++) {
      this.lanes.push(makeLane(-1, -(LANE_WIDTH / 2 + j * LANE_WIDTH) * handed));
    }
    this.assignLaneGeometry();
    this.assignDefaultLateralNeighbors();
  }

  /** Centerline point at arc position s (clamped to the road). */
  point(s: number): PathPoint {
    return this.path(Math.min(Math.max(s, 0), this.length));
  }

  lanePoint(lane: number, s: number): PathPoint {
    return this.lanes[lane].point(s);
  }

  private assignLaneGeometry(): void {
    this.lanes.forEach((lane) => {
      lane.point = (s: number): PathPoint => offsetPoint(this.point(s), lane.offset);
    });
  }

  private assignDefaultLateralNeighbors(): void {
    ([1, -1] as const).forEach((direction) => {
      const sameDirection = this.lanes
        .flatMap((lane, i) => (lane.direction === direction ? [i] : []))
        .sort((a, b) => direction * this.lanes[a].offset - direction * this.lanes[b].offset);
      sameDirection.forEach((lane, i) => {
        this.lanes[lane].leftNeighbor = sameDirection[i - 1] ?? null;
        this.lanes[lane].rightNeighbor = sameDirection[i + 1] ?? null;
      });
    });
  }
}
