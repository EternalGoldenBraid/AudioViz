export const DEFAULTS = Object.freeze({ baseline: 6, height: 1.4, azimuth: 25, elevation: 22 });
export const SIZE = Object.freeze({ width: 800, height: 420 });
export const add = (a, b) => a.map((x, i) => x + b[i]);
export const sub = (a, b) => a.map((x, i) => x - b[i]);
export const scale = (a, s) => a.map(x => x * s);
export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
export function unit(a) {
  const length = Math.hypot(...a);
  if (!length) throw new RangeError("A direction must be nonzero.");
  return scale(a, 1 / length);
}
const transpose = a => a[0].map((_, i) => a.map(row => row[i]));
export const mv = (a, v) => a.map(row => dot(row, v));
const mm = (a, b) => a.map(row => transpose(b).map(col => dot(row, col)));
const matrixAdd = (a, b) => a.map((row, i) => row.map((x, j) => x + b[i][j]));
const diagonal = v => v.map((x, i) => v.map((_, j) => i === j ? x : 0));
const outer = (a, b) => a.map(x => b.map(y => x * y));
const skew = ([x, y, z]) => [[0, -z, y], [z, 0, -x], [-y, x, 0]];
export const homogeneous = ([x, y]) => [x, y, 1];
export function warp(matrix, uv) {
  const p = mv(matrix, homogeneous(uv));
  if (Math.abs(p[2]) < 1e-12) throw new RangeError("The mapped point is at infinity.");
  return p.slice(0, 2).map(x => x / p[2]);
}
function bounded(value, min, max, name) {
  if (!Number.isFinite(value) || value < min || value > max) throw new RangeError(`${name} must be in [${min}, ${max}].`);
}
export function camera(origin, normal, focal = 0.6) {
  const n = unit(normal);
  const u = unit(cross([0, 1, 0], n));
  const v = cross(n, u);
  return { origin, n, u, v, focal, halfWidth: 2, halfHeight: 0.7 };
}
export function filmPoint(cam, [u, v]) {
  return add(add(add(cam.origin, scale(cam.n, cam.focal)), scale(cam.u, u)), scale(cam.v, v));
}
export function project(cam, point) {
  const delta = sub(point, cam.origin);
  const depth = dot(delta, cam.n);
  if (depth <= 0) throw new RangeError("Point must be in front of the observer.");
  const uv = [dot(delta, cam.u), dot(delta, cam.v)].map(x => cam.focal * x / depth);
  return { uv, point: filmPoint(cam, uv), depth };
}
export function planeHit(origin, direction, z) {
  if (Math.abs(direction[2]) < 1e-12) throw new RangeError("Ray is parallel to the reference plane.");
  const distance = (z - origin[2]) / direction[2];
  if (distance <= 0) throw new RangeError("Reference plane is behind the ray.");
  return add(origin, scale(direction, distance));
}
export function filmCorners(cam) {
  return [[-1,-1], [1,-1], [1,1], [-1,1]].map(([u, v]) =>
    filmPoint(cam, [u * cam.halfWidth, v * cam.halfHeight]));
}
export function epipolarSegment(cam, epipole, planeNormal) {
  const direction = unit(cross(cam.n, planeNormal));
  const p = project(cam, epipole).uv;
  const d = [dot(direction, cam.u), dot(direction, cam.v)];
  let low = -Infinity, high = Infinity;
  for (const [i, bound] of [[0, cam.halfWidth], [1, cam.halfHeight]]) {
    if (Math.abs(d[i]) < 1e-12) continue;
    const ends = [(-bound - p[i]) / d[i], (bound - p[i]) / d[i]].sort((a, b) => a - b);
    low = Math.max(low, ends[0]);
    high = Math.min(high, ends[1]);
  }
  if (low > high) throw new RangeError("Epipolar line misses the image rectangle.");
  return [low, high].map(t => filmPoint(cam, p.map((x, i) => x + t * d[i])));
}
function sphereHit(origin, direction, centre, radius) {
  const delta = sub(origin, centre), a = dot(direction, direction);
  const b = 2 * dot(delta, direction), c = dot(delta, delta) - radius * radius;
  const discriminant = b*b - 4*a*c;
  if (discriminant < 0) return null;
  const t = (-b - Math.sqrt(discriminant)) / (2*a);
  return t > 0 ? add(origin, scale(direction, t)) : null;
}
export function sampleFlow(world) {
  const { c1, c2, feature, planeZ } = world;
  const motion = [0.6, 0.35, 0], radius = 0.42;
  const movedCentre = add(feature, motion), samples = [];
  for (let row = 0; row < 7; row++) for (let col = 0; col < 9; col++) {
    const source = [-0.15 + col * 0.45 / 8, -0.12 + row * 0.04];
    const direction = sub(filmPoint(c1, source), c1.origin);
    const sphere = sphereHit(c1.origin, direction, feature, radius);
    const point = sphere || planeHit(c1.origin, direction, planeZ);
    const target = sphere ? add(point, motion) : point;
    const obstruction = sphereHit(c2.origin, sub(target, c2.origin), movedCentre, radius);
    if (obstruction && Math.hypot(...sub(obstruction, c2.origin)) < Math.hypot(...sub(target, c2.origin)) - 1e-7) continue;
    samples.push({ source, target: project(c2, target).uv, point, movedPoint: target, moving: Boolean(sphere) });
  }
  return samples;
}
export function createWorld({ baseline = DEFAULTS.baseline, height = DEFAULTS.height } = {}) {
  bounded(baseline, 0, 6, "Baseline");
  bounded(height, 0, 2.5, "Feature height");
  const c1 = camera([-3, 0, 0], [0.3, 0, Math.sqrt(0.91)]);
  const c2 = camera([-3 + baseline, 0, 0], [-0.3, 0, Math.sqrt(0.91)]);
  const planeZ = 6, feature = [0.2, 0.7, planeZ - height];
  const planePoint = planeHit(c1.origin, sub(feature, c1.origin), planeZ);
  // Rows of each camera basis map world vectors into camera coordinates.
  const R = mm([c2.u, c2.v, c2.n], transpose([c1.u, c1.v, c1.n]));
  const t = mv([c2.u, c2.v, c2.n], sub(c1.origin, c2.origin));
  const K = diagonal([c1.focal, c1.focal, 1]), inverseK = diagonal([1/c1.focal, 1/c1.focal, 1]);
  const E = mm(skew(t), R), F = mm(mm(inverseK, E), inverseK);
  const normalInCamera1 = mv([c1.u, c1.v, c1.n], [0, 0, 1]);
  const H = mm(mm(K, matrixAdd(R, outer(t, scale(normalInCamera1, 1 / planeZ)))), inverseK);
  const observations = [project(c1, feature), project(c2, feature)];
  const epipoles = baseline === 0 ? null : [project(c1, c2.origin), project(c2, c1.origin)];
  const planeNormal = baseline === 0 ? null : unit(cross(sub(c2.origin, c1.origin), sub(feature, c1.origin)));
  const world = {
    baseline, height, c1, c2, planeZ, feature, planePoint, observations, epipoles,
    planeNormal, R, t, E, F, H, predicted: project(c2, planePoint),
    planeCorners: [[-2.8,-1.7,planeZ], [2.8,-1.7,planeZ], [2.8,1.7,planeZ], [-2.8,1.7,planeZ]],
    planeFeatures: [[-1,-0.8,planeZ], [1.2,-0.4,planeZ], [-0.3,1.1,planeZ]],
  };
  world.flow = sampleFlow(world);
  return world;
}
export function readerView({ azimuth = DEFAULTS.azimuth, elevation = DEFAULTS.elevation } = {}) {
  bounded(azimuth, -40, 40, "Reader azimuth");
  bounded(elevation, 8, 45, "Reader elevation");
  const a = azimuth * Math.PI / 180, e = elevation * Math.PI / 180;
  const target = [0, 0, 2.5];
  const origin = add(target, [18*Math.sin(a)*Math.cos(e), 18*Math.sin(e), -18*Math.cos(a)*Math.cos(e)]);
  const observer = camera(origin, sub(target, origin), 1);
  const bounds = [];
  for (const x of [-5.5,5.5]) for (const y of [-2,2]) for (const z of [-0.1,6]) bounds.push(project(observer, [x,y,z]).uv);
  const low = [0,1].map(i => Math.min(...bounds.map(p => p[i])));
  const high = [0,1].map(i => Math.max(...bounds.map(p => p[i])));
  const zoom = Math.min((SIZE.width-120)/(high[0]-low[0]), (SIZE.height-90)/(high[1]-low[1]));
  const centre = low.map((x, i) => (x + high[i]) / 2);
  return {
    observer,
    project(point) {
      const uv = project(observer, point).uv;
      return [SIZE.width/2 + zoom*(uv[0]-centre[0]), SIZE.height/2 - zoom*(uv[1]-centre[1])];
    },
  };
}
