import assert from "node:assert/strict";
import test from "node:test";
import { createProgression, PROGRESSION_BOUNDS, IMAGE_SCALE } from "./progression.mjs";
import { buildFigure } from "./figures.mjs";
import { readerView, project, homogeneous, warp, mv, dot, sub, cross } from "./scene.mjs";

const near = (a,b,tolerance=1e-10) => assert.ok(Math.abs(a-b) < tolerance, `${a} != ${b}`);
const same = (a,b) => a.forEach((x,i) => near(x,b[i]));

test("the slider successively requires similarity, affine, homography, then plane-plus-parallax", () => {
  for (const [value,family] of [[0,0],[0.5,0],[1,0],[1.01,1],[2,1],[2.01,2],[3,2],[3.01,3],[4,3]]) {
    const model = createProgression(value), { H } = model.world;
    assert.equal(model.family,family);
    for (const [i,error] of Object.values(model.errors).entries()) {
      assert.equal(error < 1e-9,i >= family);
    }
    if (family < 2) {
      near(H[2][0],0);
      near(H[2][1],0);
    } else assert.ok(Math.hypot(H[2][0],H[2][1]) > 1e-5);
    if (family === 0) {
      near(H[0][0],H[1][1]);
      near(H[0][1],-H[1][0]);
    }
    if (family < 3) assert.ok(model.errors.affine <= model.errors.similarity+1e-10);
  }
  assert.throws(() => createProgression(NaN),/progression/);
  assert.throws(() => createProgression(4.1),/progression/);
});

test("every observation, plane prediction and parallax residual comes from the same pinhole scene", () => {
  const source = createProgression().samples.map(s => s.source);
  for (const value of [0,1,1.5,2,2.5,3,3.5,4]) {
    const model = createProgression(value), { c1,c2,H,F } = model.world;
    for (const sample of model.samples) {
      same(sample.source,source[sample.id]);
      same(project(c1,sample.point).uv,sample.source);
      same(project(c2,sample.point).uv,sample.target);
      for (const cam of [c1,c2]) {
        const uv = project(cam,sample.point).uv;
        assert.ok(Math.abs(uv[0]) < cam.halfWidth && Math.abs(uv[1]) < cam.halfHeight);
      }
      same(project(c2,sample.planePoint).uv,sample.predicted);
      same(warp(H,sample.source),sample.predicted);
      near(dot(homogeneous(sample.target),mv(F,homogeneous(sample.source))),0);
      near(Math.hypot(...cross(sub(sample.point,c1.origin),sub(sample.planePoint,c1.origin))),0);
    }
    for (const id of [0,4,20,24]) {
      near(model.samples[id].offset,0);
      same(model.samples[id].target,model.samples[id].predicted);
    }
    if (value > 3) {
      const sample = model.samples[12];
      assert.ok(Math.hypot(...sub(sample.target,sample.predicted)) > 0.001);
      const epipole = [c2.focal*model.world.t[0],c2.focal*model.world.t[1],model.world.t[2]];
      near(dot(cross(homogeneous(sample.target),homogeneous(sample.predicted)),epipole),0);
    }
  }
});

test("the controlled experiment has fixed bounded 3D framing over every regime and reader orbit", () => {
  for (let step=0;step<=400;step++) {
    const value = step/100;
    const model = createProgression(value);
    const azimuths = step%100 ? [25] : [-180,-90,25,90,180];
    const elevations = step%100 ? [22] : [-80,22,80];
    for (const azimuth of azimuths) for (const elevation of elevations) {
      const reader = readerView({azimuth,elevation,bounds:PROGRESSION_BOUNDS});
      const figure = buildFigure("world",model.world,reader);
      for (const shape of figure.shapes) for (const [x,y] of shape.points || [shape.p]) {
        assert.ok(Number.isFinite(x) && Number.isFinite(y));
        assert.ok(x >= 0 && x <= figure.width && y >= 0 && y <= figure.height,`${value}: ${x}, ${y}`);
      }
      for (const sample of model.samples) {
        const marker = figure.shapes.find(shape => shape.id === `construction-${sample.id}`);
        same(marker.p,reader.project(sample.point));
      }
    }
    for (const sample of model.samples) for (const uv of [sample.target,sample.predicted]) {
      const [x,y] = [600+IMAGE_SCALE*(uv[0]-0.3),140-IMAGE_SCALE*uv[1]];
      assert.ok(x > 423.9 && x < 776.1 && y > 43.9 && y < 256.1,`${value}: ${x}, ${y}`);
    }
  }
});
