import { add, sub, scale, dot, unit, camera, cameraGeometry, filmPoint, intersectPlane, project, warp } from "./scene.mjs";

export const PROGRESSION_BOUNDS = [[-5.5,5.5], [-2.5,2.5], [-0.1,8.5]];
export const IMAGE_SCALE = 360;
export const FAMILIES = ["Similarity", "Affine", "Homography", "Plane + parallax"];
const phase = (value, start) => Math.max(0, Math.min(1, value-start));

function leastSquares(rows, values) {
  const n = rows[0].length;
  const a = Array.from({ length: n }, (_, i) => [
    ...Array.from({ length: n }, (_, j) => rows.reduce((sum, row) => sum+row[i]*row[j], 0)),
    rows.reduce((sum, row, k) => sum+row[i]*values[k], 0),
  ]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col+1; row < n; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    if (Math.abs(a[pivot][col]) < 1e-12) throw new RangeError("Transform samples do not determine the fitted model.");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    a[col] = a[col].map(x => x/divisor);
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      a[row] = a[row].map((x, i) => x-factor*a[col][i]);
    }
  }
  return a.map(row => row[n]);
}

function fittedModels(samples) {
  const similarity = [], affine = [], values = [];
  for (const { source: [x,y], target: [u,v] } of samples) {
    similarity.push([x,-y,1,0], [y,x,0,1]);
    affine.push([x,y,1,0,0,0], [0,0,0,x,y,1]);
    values.push(u,v);
  }
  const [a,b,tx,ty] = leastSquares(similarity, values);
  const [xx,xy,xt,yx,yy,yt] = leastSquares(affine, values);
  return {
    similarity: [[a,-b,tx], [b,a,ty], [0,0,1]],
    affine: [[xx,xy,xt], [yx,yy,yt], [0,0,1]],
  };
}

export function createProgression(value = 0) {
  if (!Number.isFinite(value) || value < 0 || value > 4) throw new RangeError("Transform progression must be in [0, 4].");
  const zoom = 1+0.12*phase(value,0), roll = 10*phase(value,0);
  const tilt = phase(value,1), yaw = 14*phase(value,2), height = 1.6*phase(value,3);
  const angle = yaw*Math.PI/180, rotation = roll*Math.PI/180;
  const c1 = camera([-3,0,0], [0,0,1]);
  const c2 = camera([-1,0,0], [-Math.sin(angle),0,Math.cos(angle)], c1.focal*zoom);
  for (const cam of [c1,c2]) {
    cam.halfWidth = 0.7;
    cam.halfHeight = 0.4;
  }
  const { u,v } = c2;
  c2.u = add(scale(u,Math.cos(rotation)), scale(v,Math.sin(rotation)));
  c2.v = add(scale(u,-Math.sin(rotation)), scale(v,Math.cos(rotation)));
  const normal = unit([0.6*tilt,0.3*tilt,1]), anchor = [0,0,6];
  const geometry = cameraGeometry(c1,c2,normal,anchor);
  const distance = dot(normal,sub(anchor,c1.origin));
  const hit = uv => intersectPlane(c1.origin,sub(filmPoint(c1,uv),c1.origin),normal,anchor);
  const samples = [], edges = [], side = 5;
  for (let row = 0; row < side; row++) for (let col = 0; col < side; col++) {
    const source = [0.08+0.44*col/(side-1), -0.12+0.24*row/(side-1)];
    const planePoint = hit(source);
    const offset = row && col && row < side-1 && col < side-1
      ? height*(Math.sin(Math.PI*col/(side-1))*Math.sin(Math.PI*row/(side-1)))**2 : 0;
    // Moving along the source ray keeps the image-1 sample fixed; offset is normal distance to the plane.
    const point = add(planePoint,scale(sub(c1.origin,planePoint),offset/distance));
    const id = row*side+col;
    samples.push({ id, source, planePoint, point, offset, target: project(c2,point).uv, predicted: warp(geometry.H,source) });
    if (col) edges.push([id-1,id]);
    if (row) edges.push([id-side,id]);
  }
  const fitted = fittedModels(samples);
  const error = matrix => Math.sqrt(samples.reduce((sum, sample) =>
    sum+sub(warp(matrix,sample.source),sample.target).reduce((s,x) => s+x*x,0),0)/samples.length);
  const errors = {
    similarity: error(fitted.similarity), affine: error(fitted.affine), homography: error(geometry.H),
  };
  const family = errors.similarity < 1e-9 ? 0 : errors.affine < 1e-9 ? 1 : errors.homography < 1e-9 ? 2 : 3;
  const centre = samples[Math.floor(samples.length/2)];
  return {
    value, zoom, roll, tilt, planeTilt: Math.atan(Math.hypot(0.6*tilt,0.3*tilt))*180/Math.PI,
    yaw, height, samples, edges, fitted, errors, family,
    world: {
      ...geometry, c1, c2, baseline: 2, height, samples, edges,
      planeCorners: [[0.03,-0.17],[0.57,-0.17],[0.57,0.17],[0.03,0.17]].map(hit),
      planeFeatures: samples.filter(s => !s.offset).map(s => s.point),
      feature: centre.point, planePoint: centre.planePoint,
      observations: [project(c1,centre.point),project(c2,centre.point)],
      predicted: project(c2,centre.planePoint),
    },
  };
}
