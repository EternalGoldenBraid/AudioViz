import { SIZE, add, scale, sub, filmPoint, filmCorners, project, epipolarSegment, homogeneous, mv, strokeFragments } from "./scene.mjs";

export const COLORS = Object.freeze({
  bg: "#1B1F27", fg: "#E6EDF3", dim: "#9AA5B1", c1: "#7BC47F",
  c2: "#E5826F", prediction: "#6EA8FE", observed: "#F0C060", epipolar: "#B39DDB",
});
export const MODES = Object.freeze({
  world: "Observers in one 3D world", homography: "Plane-induced homography",
  parallax: "Plane and parallax", epipolar: "Epipolar plane and image intersections",
  flow: "Sampled image-coordinate flow", predictions: "Point, line and displacement predictions",
});
const mixedFill = (color, fraction) => "#" + [1,3,5].map(i => {
  const bg = parseInt(COLORS.bg.slice(i,i+2),16), fg = parseInt(color.slice(i,i+2),16);
  return Math.round(bg + fraction*(fg-bg)).toString(16).padStart(2,"0");
}).join("");

export function buildFigure(mode, world, reader) {
  if (!Object.hasOwn(MODES, mode)) throw new RangeError(`Unknown figure mode: ${mode}`);
  const surfaces = [], strokes = [], worldStrokes = [], dots = [], labels = [];
  const polygon = (points, color, opacity = 0.08, width = 1.2) => surfaces.push({ type: "polygon", points, color, fill: mixedFill(color,opacity), opacity, width });
  const line = (a, b, color, dashed = false, arrow = false, width = 1.2) => {
    strokes.push({ type: "line", points: [a,b], color, dashed, width });
    const length = Math.hypot(...sub(b,a));
    if (arrow && length > 5) {
      const d = scale(sub(b,a), 1/length), v = [-d[1], d[0]];
      strokes.push({ type: "polygon", points: [b, add(add(b,scale(d,-5)),scale(v,2.2)), add(add(b,scale(d,-5)),scale(v,-2.2))], color, fill: color, opacity: 1, width: 0 });
    }
  };
  const dotAt = (p, color, id, radius = 3) => dots.push({ type: "dot", p, color, id, radius });
  const label = (p, text, tex, color = COLORS.fg, offset = [0,-14], anchor = "middle") =>
    labels.push({ type: "label", p: add(p, offset), text, tex, color, anchor });
  const p3 = p => reader.project(p);
  const line3 = (a,b,color,dashed=false) => worldStrokes.push({ points: [a,b], color, dashed, width: 1.2 });
  const point3 = (p,color,id,radius) => dotAt(p3(p),color,id,radius);
  const label3 = (p,text,tex,color,offset,anchor) => label(p3(p),text,tex,color,offset,anchor);
  const finish = () => {
    const projected = strokeFragments(worldStrokes.map(s => s.points),reader).map(s => ({
      ...worldStrokes[s.index], type: "line", points: s.points,
    }));
    return { width: SIZE.width, height: SIZE.height, title: MODES[mode], shapes: [...surfaces, ...projected, ...strokes, ...dots, ...labels] };
  };
  const surface3 = (points,color,opacity=0.08,edges=true) => {
    polygon(points.map(p3),color,opacity,0);
    if (edges) points.forEach((p,i) => line3(p,points[(i+1)%points.length],color));
  };
  const { c1, c2, feature, planePoint, observations, epipoles } = world;
  const construction = mode === "world" && Boolean(world.samples);

  if (["world", "homography", "parallax", "epipolar"].includes(mode)) {
    if (mode === "epipolar" && world.baseline > 0) surface3([c1.origin,c2.origin,feature], COLORS.epipolar,0.12,false);
    surface3(world.planeCorners, COLORS.prediction, 0.06);
    label3(world.planeCorners[3], "Π", String.raw`$\Pi$`, COLORS.prediction, [-12,-8]);
    for (const [i, cam] of [c1,c2].entries()) {
      const color = COLORS[i ? "c2" : "c1"];
      surface3(filmCorners(cam), color);
      label3(filmPoint(cam, [(i ? 1 : -1)*cam.halfWidth*0.75, cam.halfHeight*(0.45/0.7)]), i ? "I₂" : "I₁", `$I_${i+1}$`, color);
      point3(cam.origin, color, `camera-${i+1}`, 3.5);
      if (world.baseline || i === 1) label3(cam.origin, !world.baseline ? "C₁ = C₂" : `C${i ? "₂" : "₁"}`,
        !world.baseline ? "$C_1=C_2$" : `$C_${i+1}$`, color, [i ? 12 : 0,26], i ? "start" : "middle");
    }
    line3(c1.origin,c2.origin,COLORS.dim,true);
    const selected = mode === "homography" ? planePoint : feature;
    const imagePoints = [c1,c2].map(cam => project(cam, selected).point);
    line3(c1.origin, mode === "parallax" || construction ? planePoint : selected, COLORS.dim, true);
    line3(c2.origin, selected, COLORS.dim, true);
    point3(selected,COLORS.observed,mode === "homography" ? "plane-point" : "feature",4);
    label3(selected, mode === "homography" ? "Xπ" : (mode === "parallax" || construction) && !world.height ? "X = Xπ" : "X",
      mode === "homography" ? String.raw`$X_\pi$` : (mode === "parallax" || construction) && !world.height ? String.raw`$X=X_\pi$` : "$X$", COLORS.observed);
    for (const [i, point] of imagePoints.entries()) {
      const coincident = mode === "parallax" && i === 1 && (!world.baseline || !world.height);
      if (mode === "homography" && i === 1) {
        point3(point,COLORS.prediction,"predicted",3);
        label3(point,"Hx₁","$Hx_1$",COLORS.prediction,[12,-5],"start");
        continue;
      }
      point3(point,COLORS.observed,`observation-${i+1}`,3);
      label3(point, coincident ? "Hx₁ = x₂" : `x${i ? "₂" : "₁"}`,
        coincident ? "$Hx_1=x_2$" : `$x_${i+1}$`, COLORS.observed,
        [i ? -12 : 12, -14], i ? "end" : "start");
    }
    if (construction) {
      for (const [a,b] of world.edges) {
        const color = world.samples[a].offset || world.samples[b].offset ? COLORS.observed : COLORS.prediction;
        line3(world.samples[a].point,world.samples[b].point,color);
      }
      for (const sample of world.samples) point3(sample.point,
        sample.offset ? COLORS.observed : COLORS.prediction, `construction-${sample.id}`,1.5);
      if (world.height) {
        line3(c2.origin,planePoint,COLORS.prediction,true);
        point3(planePoint,COLORS.prediction,"plane-point");
        label3(planePoint,"Xπ",String.raw`$X_\pi$`,COLORS.prediction,[12,-10],"start");
      }
    } else if (mode === "world") {
      for (const [i,cam] of [c1,c2].entries()) {
        const origin = filmPoint(cam,[0,0]), p = p3(origin);
        line(add(p,[-3,0]),add(p,[3,0]),COLORS.dim);
        line(add(p,[0,-3]),add(p,[0,3]),COLORS.dim);
        label(p, i ? "O₂" : "O₁", `$O_${i+1}$`, COLORS.dim, i ? [20,0] : [-16,-14], i ? "start" : "end");
      }
      const origin = [0,0,0];
      for (const [axis, color, text, tex] of [
        [[0.8,0,0],COLORS.c2,"x","$x$"], [[0,0.8,0],COLORS.c1,"y","$y$"], [[0,0,0.8],COLORS.prediction,"z","$z$"],
      ]) {
        const end = add(origin,axis);
        line(p3(origin),p3(end),color,false,true);
        label3(end,text,tex,color,[10,0],"start");
      }
      label3(origin,"W","$W$",COLORS.dim,[-12,12]);
    }
    if (mode === "homography") {
      for (const point of world.planeFeatures) {
        point3(point,COLORS.observed,undefined,2);
        for (const cam of [c1,c2]) point3(project(cam,point).point,COLORS.observed,undefined,2);
      }
    }
    if (mode === "parallax") {
      line3(c2.origin,planePoint,COLORS.prediction,true);
      point3(planePoint,COLORS.prediction,"plane-point",3);
      if (world.height) label3(planePoint,"Xπ",String.raw`$X_\pi$`,COLORS.prediction,[12,-4],"start");
      if (world.baseline > 0) {
        const e = epipoles[1].point;
        const segment = epipolarSegment(c2,e,world.planeNormal);
        line3(...segment,COLORS.epipolar);
        point3(e,COLORS.epipolar,"epipole-2");
        label3(e,"e′","$e'$",COLORS.epipolar,[0,20]);
      }
      if (world.baseline && world.height) {
        const predicted = world.predicted.point, observed = observations[1].point;
        point3(predicted,COLORS.prediction,"predicted");
        label3(predicted,"Hx₁","$Hx_1$",COLORS.prediction,[12,-5],"start");
        line(p3(predicted),p3(observed),COLORS.observed,false,true,1.8);
      }
    }
    if (mode === "epipolar" && world.baseline > 0) {
      for (const [i,cam] of [c1,c2].entries()) {
        const e = epipoles[i].point, segment = epipolarSegment(cam,e,world.planeNormal);
        line3(...segment,COLORS.epipolar);
        point3(e,COLORS.epipolar,`epipole-${i+1}`);
        label3(e,i ? "e′" : "e",i ? "$e'$" : "$e$",COLORS.epipolar,[0,20]);
      }
    }
    return finish();
  }

  if (mode === "flow") {
    const uv = ([u,v]) => [400 + 388*(u-0.45), 220-388*v];
    polygon([[70,88],[730,88],[730,352],[70,352]],COLORS.dim,0);
    line(uv([-0.4,0]),uv([1.3,0]),COLORS.dim,true);
    label([70,60],"image-coordinate displacement","image-coordinate displacement",COLORS.fg,[0,0],"start");
    label([730,376],"u","$u$",COLORS.dim,[0,0],"end");
    label([55,90],"v","$v$",COLORS.dim,[0,0]);
    for (const p of world.flow) {
      const color = p.moving ? COLORS.c2 : COLORS.observed;
      line(uv(p.source),uv(p.target),color,false,true,0.9);
      dotAt(uv(p.source),color,undefined,1.5);
    }
    label([70,402],"gold: rigid plane     coral: translating sphere","gold: rigid plane\\quad coral: translating sphere",COLORS.dim,[0,0],"start");
    return finish();
  }

  const centreUV = observations[1].uv;
  const centres = [130,400,670], cy = 225, zoom = 250;
  const plot = (uv,i) => [centres[i]+zoom*(uv[0]-centreUV[0]),cy-zoom*(uv[1]-centreUV[1])];
  for (const [i,title,tex] of [[0,"H: point","$H$: point"],[1,"F: line","$F$: line"],[2,"flow: displacement","flow: displacement"]]) {
    const cx = centres[i];
    polygon([[cx-105,120],[cx+105,120],[cx+105,330],[cx-105,330]],COLORS.dim,0);
    label([cx,88],title,tex,COLORS.fg,[0,0]);
    const observed = plot(centreUV,i);
    dotAt(observed,COLORS.observed,`comparison-observed-${i}`);
    label(observed,"x₂","$x_2$",COLORS.observed,[-8,-14],"end");
    if (i === 0) {
      const predicted = plot(world.predicted.uv,i);
      dotAt(predicted,COLORS.prediction,"comparison-predicted");
      if (world.baseline && world.height) label(predicted,"Hx₁","$Hx_1$",COLORS.prediction,[8,-14],"start");
      line(predicted,observed,COLORS.observed,false,true);
    } else if (i === 1 && world.baseline) {
      const [a,b] = mv(world.F,homogeneous(observations[0].uv));
      const d = [-b,a], length = Math.max(Math.abs(d[0]),Math.abs(d[1]));
      const endpoint = sign => plot(add(centreUV,scale(d,sign*0.42/length)),i);
      line(endpoint(-1),endpoint(1),COLORS.epipolar);
    } else if (i === 2) {
      const source = plot(observations[0].uv,i);
      // A clipped tail keeps the crop bounded; the displacement uses full image coordinates.
      const delta = sub(source,observed), factor = Math.min(1, 100 / Math.max(...delta.map(Math.abs)));
      const tail = add(observed,scale(delta,factor));
      line(tail,observed,COLORS.c2,false,true,1.5);
      label([cx,365],"x₂ − x₁","$x_2-x_1$",COLORS.c2,[0,0]);
    }
  }
  if (!world.baseline) label([400,365],"F = 0: no line constraint","$F=0$: no line constraint",COLORS.epipolar,[0,0]);
  return finish();
}
