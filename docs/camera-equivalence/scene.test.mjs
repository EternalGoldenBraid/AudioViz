import assert from "node:assert/strict";
import test from "node:test";
import { createWorld, readerView, filmCorners, project, epipolarSegment, warp, mv, homogeneous, dot, sub, cross, strokeFragments, add, scale } from "./scene.mjs";
import { buildFigure, MODES } from "./figures.mjs";

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
  for (const azimuth of [-180, -90, 25, 90, 180]) for (const elevation of [-80, 0, 22, 80]) {
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

test("renderers share finite bounded primitives; reader motion leaves sensor-coordinate plots unchanged", () => {
  for (const baseline of [0,6]) for (const height of [0,2.5]) {
    const w = createWorld({baseline,height});
    for (const azimuth of [-180,-90,25,90,180]) for (const elevation of [-80,0,22,80]) {
      const reader = readerView({azimuth,elevation});
      for (const mode of Object.keys(MODES)) {
        const figure = buildFigure(mode,w,reader);
        for (const shape of figure.shapes) for (const [x,y] of shape.points || [shape.p]) {
          assert.ok(Number.isFinite(x) && Number.isFinite(y));
          assert.ok(x >= 0 && x <= figure.width && y >= 0 && y <= figure.height,
            `${mode}: (${x}, ${y}) leaves the canvas`);
        }
      }
      const h = buildFigure("homography",w,reader), p = buildFigure("parallax",w,reader);
      same(h.shapes.find(s => s.id==="plane-point").p, p.shapes.find(s => s.id==="plane-point").p);
      same(h.shapes.find(s => s.id==="predicted").p, reader.project(w.predicted.point));
      same(buildFigure("world",w,reader).shapes.find(s => s.text==="W").p,
        add(reader.project([0,0,0]),[-12,12]));
    }
    for (const mode of ["flow","predictions"]) {
      assert.deepEqual(buildFigure(mode,w,readerView()), buildFigure(mode,w,readerView({azimuth:-30})));
    }
  }
});

test("projected crossings gap the farther stroke; a true 3D intersection stays connected", () => {
  const reader = readerView(), cam = reader.observer;
  const horizontal = [-2,2].map(x => add(add(cam.origin,scale(cam.n,8)),scale(cam.u,x)));
  const vertical = depth => [-2,2].map(y => add(add(cam.origin,scale(cam.n,depth)),scale(cam.v,y)));
  for (const segments of [[horizontal,vertical(12)], [vertical(12),horizontal]]) {
    const fragments = strokeFragments(segments,reader);
    const far = segments.indexOf(horizontal) === 0 ? 1 : 0;
    assert.equal(fragments.filter(p => p.index === far).length,2);
    assert.equal(fragments.filter(p => p.index !== far).length,1);
    const [before,after] = fragments.filter(p => p.index === far);
    assert.ok(before.range[1] < 0.5 && after.range[0] > 0.5);
  }
  assert.equal(strokeFragments([horizontal,vertical(8)],reader).length,2);
  const world = createWorld(), corners = filmCorners(world.c1);
  const lowerEdge = [corners[0],corners[1]], baseline = [world.c1.origin,world.c2.origin];
  const actual = strokeFragments([lowerEdge,baseline],readerView({elevation:45}));
  assert.equal(actual.filter(p => p.index===0).length,2);
  assert.equal(actual.filter(p => p.index===1).length,1);
});
