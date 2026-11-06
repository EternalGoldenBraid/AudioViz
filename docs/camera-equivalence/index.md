<script setup>
import { ref, computed } from "vue";
import SceneFigure from "./SceneFigure.vue";
import { DEFAULTS, createWorld, readerView } from "./scene.mjs";
const baseline = ref(DEFAULTS.baseline);
const height = ref(DEFAULTS.height);
const azimuth = ref(DEFAULTS.azimuth);
const elevation = ref(DEFAULTS.elevation);
const world = computed(() => createWorld({ baseline: baseline.value, height: height.value }));
const reader = computed(() => readerView({ azimuth: azimuth.value, elevation: elevation.value }));
</script>

# Two cameras as equivalent evidence on one canvas

## World

Observers (cameras) sit at unknown poses $T_k$ in a world frame $W$ and observe features $X_j$ with unknown intrinsics $\pi_k$:

$$x_{kj} = \pi_k\!\left(T_k^{-1} X_j\right)$$

$x_{kj}$ is feature $j$'s image position in camera $k$; $\pi_k$ projects a point from that camera's coordinates into its image.

The figures share one synthetic 3D scene. Their camera parameters are known for illustration; they are not estimates from the learned canvas. The reader is a third perspective observer: changing your viewpoint changes the drawing, not the cameras' observations.

<div class="scene-controls">
  <label for="reader-azimuth">reader azimuth</label><input id="reader-azimuth" type="range" min="-40" max="40" step="1" v-model.number="azimuth" /><output for="reader-azimuth">{{ azimuth }}°</output>
  <label for="reader-elevation">reader elevation</label><input id="reader-elevation" type="range" min="8" max="45" step="1" v-model.number="elevation" /><output for="reader-elevation">{{ elevation }}°</output>
</div>

<SceneFigure mode="world" :world="world" :reader="reader" />

**In the figure:** $C_k$ is a camera's projection centre; $I_k$ its virtual image plane (green or coral); $O_k$ its image-coordinate origin, where the optical axis meets that plane. $X$ is a test feature and $x_k$ its ray's intersection with $I_k$. $\Pi$ is the blue reference plane. Camera centres are not image origins.

The virtual surfaces are transparent. At projected edge/ray crossings, the farther stroke has a small gap to show depth. Only labelled dots mark the actual ray/plane intersections.

The two cameras are the same kind of observer. They differ only in $T_k$ and $\pi_k$.

## Model

One shared hidden canvas $h$. Each camera $k$ has its own learned readout:

$$\hat{x}_k = g_k(h), \qquad e_k = x_k - \hat{x}_k$$

$x_k$ is camera $k$'s observed image, $\hat{x}_k$ its prediction, and $e_k$ their difference. The learned map $g_k$ produces an image from canvas state $h$.

The canvas is corrected by the errors of every enabled camera at once. Cameras differ only through $g_k$ (pose and intrinsics are absorbed into it). No geometry is assumed.

## Phase 1 (multi-device agent)

- A second readout branch for camera 2 over the same canvas.
- Cover camera 1 and camera 2 still corrects the canvas.
- No coupling between branches; no homography.

## Phase 2: visualize correspondences

No transform is fitted for now. Once both cameras project onto the canvas, the correspondence should show itself there. Phase 2 only makes it visible:

Here $h$ is the shared canvas state, $g_k$ camera $k$'s learned readout, and $e_k$ its image prediction error. A shared unit $i$ has a receptive-field centre $c_i^{(k)}$ in each camera's image.

1. Show each camera's prediction $g_k(h)$ and error $e_k$ next to the canvas.
2. Show the underlying mappings: for a selected hidden unit $i$, its receptive field in each camera (from $g_k$).
3. Show which canvas regions both cameras excite.

Fitting a transform to the receptive-field centroids $c_i^{(k)}$ is deferred until the visualization shows which description fits (next section).

![Schematic: hidden units as correspondences between the two cameras](./correspondence.svg)

## Candidate transforms

Later, the correspondences can be compared against several transforms. The solid arrows show increasingly flexible descriptions of matches. Calibration converts $E$ and $F$ between coordinate systems; that link is not a special-case relation.

![Schematic: transforms and their constraints](./transforms.svg)

**Symbols:** $H$ is a plane's image mapping; $E,F$ constrain matches to epipolar lines; $\rho$ is scalar parallax along epipole $e'$; $u(x)$ is a displacement at image position $x$.

### Homography $H$

The homography $H$ maps reference-plane points between images. Under pure rotation, it is exact at every depth.

<SceneFigure mode="homography" :world="world" :reader="reader" />

**Symbols:** $C_1,C_2$ are projection centres. $X_\pi$ is where $C_1$'s ray through test feature $X$ meets reference plane $\Pi$; its image positions are $x_1$ in camera 1 and blue $Hx_1$ in camera 2. $K_1,K_2$ are intrinsics. With relative pose $X_{c2}=R X_{c1}+t$ and plane $n^\top X_{c1}=d$, the mapping is

$$H=K_2\left(R+\frac{t n^\top}{d}\right)K_1^{-1}.$$

Here $n,d$ are expressed in camera 1's coordinates. The same convention is used for every figure and calculation below.

### Plane + parallax

**Parallax** is the depth-dependent shift caused by camera translation. A plane's homography $H$ aligns that plane; the remaining shift of off-plane points carries information about relative depth.

**Symbols:** $C_1,C_2$ are projection centres; $X$ is an off-plane test feature; $x_1,x_2$ its image-plane intersections. $X_\pi$ is where $C_1$'s ray through $X$ meets reference plane $\Pi$. Blue $Hx_1$ projects $X_\pi$ into image 2; gold $x_2$ projects $X$.

<div class="scene-controls">
  <label for="scene-baseline">camera baseline</label><input id="scene-baseline" type="range" min="0" max="6" step="0.1" v-model.number="baseline" /><output for="scene-baseline">{{ baseline.toFixed(1) }}</output>
  <label for="scene-height">off-plane offset</label><input id="scene-height" type="range" min="0" max="2.5" step="0.1" v-model.number="height" /><output for="scene-height">{{ height.toFixed(1) }}</output>
</div>

<SceneFigure mode="parallax" :world="world" :reader="reader" />

These controls update the scene used by every figure. Distances are arbitrary scene units; the offset is perpendicular to $\Pi$. Camera orientations and intrinsics stay fixed. Purple $e'$ is $C_1$ projected into image 2; $Hx_1$ and $x_2$ lie on its line.

$x_1,x_2$ are the feature's positions in the two images; $H$ is the reference plane's mapping. In $x_2\sim Hx_1+\rho(x_1)\,e'$, $\rho$ is a scalar parallax coefficient and $e'$ the epipole. Without calibration, this constrains projective structure, not depth in metres.

### Fundamental $F$ and essential $E$

The fundamental matrix $F$ maps image position $x_1$ to an epipolar line $l'$ containing its match $x_2$. The essential matrix $E$ expresses this constraint in calibrated camera coordinates.

<SceneFigure mode="epipolar" :world="world" :reader="reader" />

**Symbols:** $C_1,C_2$ are camera projection centres; $x_1,x_2$ are where their viewing rays through feature $X$ intersect the virtual image planes. $e,e'$ are the cameras' projections into each other's images. $K_1,K_2$ are intrinsics; $R,t$ are relative rotation and translation; $[t]_\times$ represents the cross product with $t$.

$$x_2^\top F x_1=0,\qquad E=[t]_\times R,\qquad F=K_2^{-\top} E K_1^{-1}.$$

The purple triangle is the plane through $C_1,C_2,X$. Its intersections with the two image rectangles are the epipolar lines. $F$ constrains $x_2$ to a line; it does not choose a point on that line.

### Dense flow $u(x)$

The flow $u(x)$ gives a 2D displacement at image position $x$. Matching positions satisfy $x_2=x_1+u(x_1)$, with no rigid-scene or planar constraint.

<SceneFigure mode="flow" :world="world" :reader="reader" />

This is a sampled flow field from ray-cast 3D surfaces, not a dense estimator. Gold matches follow the rigid plane. Coral matches follow an independently translating sphere and can violate the rigid scene's $F$. This plot uses sensor coordinates, so changing the reader's viewpoint does not change it.

### Side by side

$H$ predicts a matching point, $F$ a line through epipole $e'$, and $u(x)$ a displacement at each image position $x$.

<SceneFigure mode="predictions" :world="world" :reader="reader" />

## Open questions

- A homography is exact for a plane or pure rotation; distant scenes can approximate it. Which description first fits the measured correspondences?
- EO and IR are not a matched pair. Do shared units emerge at all without cross-modal supervision?
- Should the checkerboard calibration be an independent baseline?
- Where does the visualization show up in the TUI: tile, overlay or log?

## Figure sources

`scene.mjs` owns the world and camera mathematics. `figures.mjs` projects it into shared drawing primitives. `SceneFigure.vue` displays those primitives; `render-figures.mjs` writes the same default scene as TikZ and outlined SVG. All 3D views share the reader camera and framing; sensor-coordinate plots have their own explicitly labelled frames. Correspondence and transform-hierarchy diagrams are schematic.

From the repository root:

```sh
node --test docs/camera-equivalence/scene.test.mjs
node docs/camera-equivalence/render-figures.mjs
```

The renderer requires LaTeX with TikZ/standalone and `dvisvgm`. Generated `.tex` and `.svg` files are committed together; geometry edits belong in the shared modules.

<style>
.scene-controls {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) 4ch;
  gap: 8px 16px;
  align-items: center;
  padding: 14px 16px;
  background: #1B1F27;
  color: #E6EDF3;
  font-family: "Latin Modern Roman", "CMU Serif", "Computer Modern", serif;
}
.scene-controls input { width: 100%; min-width: 0; accent-color: #F0C060; }
.scene-controls output { color: #F0C060; text-align: right; font-variant-numeric: tabular-nums; }
</style>
