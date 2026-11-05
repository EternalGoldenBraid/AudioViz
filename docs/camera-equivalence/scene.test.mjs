import assert from "node:assert/strict";
import test from "node:test";
import { createWorld, readerView, filmCorners, filmPoint, project, epipolarSegment, warp, mv, homogeneous, dot, sub, cross } from "./scene.mjs";

const near = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a-b) < tolerance, `${a} != ${b}`);
const same = (a, b) => a.forEach((x, i) => near(x, b[i]));
const onPlane = (cam, p) => {
  near(dot(sub(p, cam.origin), cam.n), cam.focal);
  const uv = project(cam, p).uv;
  assert.ok(Math.abs(uv[0]) <= cam.halfWidth + 1e-10);
  assert.ok(Math.abs(uv[1]) <= cam.halfHeight + 1e-10);
};

test("one world gives consistent image planes, rays, H, F and parallax across the control range", () => {
  for (const baseline of [0, 0.1, 3, 6]) for (const height of [0, 1.4, 2.5]) {
    const w = createWorld({ baseline, height });
    for (const cam of [w.c1, w.c2]) {
      near(dot(cam.n, cam.u), 0);
      near(dot(cam.n, cam.v), 0);
      same(cross(cam.u, cam.v), cam.n);
      for (const corner of filmCorners(cam)) onPlane(cam, corner);
      for (const feature of [...w.planeFeatures, w.feature, w.planePoint]) {
        const observation = project(cam, feature);
        onPlane(cam, observation.point);
        near(Math.hypot(...cross(sub(observation.point, cam.origin), sub(feature, cam.origin))), 0);
      }
    }
    for (const point of [...w.planeFeatures, w.planePoint]) {
      same(warp(w.H, project(w.c1, point).uv), project(w.c2, point).uv);
    }
    same(warp(w.H, w.observations[0].uv), w.predicted.uv);
    const observed = homogeneous(w.observations[1].uv);
    near(dot(observed, mv(w.F, homogeneous(w.observations[0].uv))), 0);
    if (!baseline || !height) same(w.predicted.uv, w.observations[1].uv);
    else assert.ok(Math.hypot(...sub(w.predicted.uv, w.observations[1].uv)) > 1e-5);
    if (!baseline) { assert.equal(w.epipoles, null); continue; }
    for (const [i, cam] of [w.c1, w.c2].entries()) {
      const epipole = w.epipoles[i].point;
      onPlane(cam, epipole);
      near(dot(w.planeNormal, sub(epipole, w.c1.origin)), 0);
      for (const endpoint of epipolarSegment(cam, epipole, w.planeNormal)) {
        onPlane(cam, endpoint);
        near(dot(w.planeNormal, sub(endpoint, w.c1.origin)), 0);
      }
    }
    const parallaxLine = cross(homogeneous(w.predicted.uv), homogeneous(w.observations[1].uv));
    near(dot(parallaxLine, homogeneous(w.epipoles[1].uv)), 0);
  }
});

test("sampled flow uses actual visible 3D correspondences; independent motion can violate F", () => {
  const w = createWorld();
  const staticSamples = w.flow.filter(p => !p.moving);
  const movingSamples = w.flow.filter(p => p.moving);
  assert.ok(staticSamples.length > 30 && movingSamples.length > 0);
  for (const p of w.flow) {
    same(project(w.c1, p.point).uv, p.source);
    same(project(w.c2, p.movedPoint).uv, p.target);
    if (!p.moving) near(dot(homogeneous(p.target), mv(w.F, homogeneous(p.source))), 0);
  }
  assert.ok(movingSamples.some(p => Math.abs(dot(homogeneous(p.target), mv(w.F, homogeneous(p.source)))) > 1e-4));
});

test("the reader is a perspective observer with fixed scene bounds, independent of feature height", () => {
  for (const azimuth of [-40, 25, 40]) for (const elevation of [8, 22, 45]) {
    const reader = readerView({ azimuth, elevation });
    for (const baseline of [0, 6]) for (const height of [0, 2.5]) {
      const w = createWorld({ baseline, height });
      for (const p of [...w.planeCorners, ...filmCorners(w.c1), ...filmCorners(w.c2), w.feature, w.planePoint]) {
        const [x,y] = reader.project(p);
        assert.ok(x > 0 && x < 800 && y > 0 && y < 420);
      }
    }
  }
  const w = createWorld(), front = readerView(), side = readerView({ azimuth: -30 });
  assert.notDeepEqual(front.project(w.feature), side.project(w.feature));
  assert.throws(() => project(w.c1, sub(w.c1.origin, w.c1.n)), /in front/);
  assert.throws(() => createWorld({ height: 6 }), /Feature height/);
  assert.throws(() => readerView({ elevation: 90 }), /Reader elevation/);
});
