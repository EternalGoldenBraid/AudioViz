# Two cameras as equivalent evidence on one canvas

## World

Observers (cameras) sit at unknown poses $T_k$ in a world frame $W$ and observe features $X_j$ with unknown intrinsics $\pi_k$:

$$x_{kj} = \pi_k\!\left(T_k^{-1} X_j\right)$$

![Observers in a world frame observing 3D features](./model.svg)

The two cameras are the same kind of observer. They differ only in $T_k$ and $\pi_k$.

## Model

One shared hidden canvas $h$. Each camera $k$ has its own learned readout:

$$\hat{x}_k = g_k(h), \qquad e_k = x_k - \hat{x}_k$$

The canvas is corrected by the errors of every enabled camera at once. Cameras differ only through $g_k$ (pose and intrinsics are absorbed into it). No geometry is assumed.

## Phase 1 (multi-device agent)

- A second readout branch for camera 2 over the same canvas.
- Cover camera 1 and camera 2 still corrects the canvas.
- No coupling between branches; no homography.

## Phase 2: visualize correspondences

No transform is fitted for now. Once both cameras project onto the canvas, the correspondence should show itself there. Phase 2 only makes it visible:

1. Show each camera's prediction $g_k(h)$ and error $e_k$ next to the canvas.
2. Show the underlying mappings: for a selected hidden unit $i$, its receptive field in each camera (from $g_k$).
3. Show which canvas regions both cameras excite.

Fitting a transform to the receptive-field centroids $c_i^{(k)}$ is deferred until the visualization shows which description fits (next section).

![Hidden units as correspondences between the two cameras](./correspondence.svg)

## Candidate transforms

Later, the correspondences can be compared against several transforms. They form a chain of generalizations, so a fit that fails at one level shows what the next level adds.

![Transforms ordered by generality](./transforms.svg)

### Homography $H$

Point to point. Exact for a plane or a pure rotation. Residual structure that grows with scene depth means $\rho\neq 0$.

![Homography induced by a plane](./fig-homography.svg)

### Plane + parallax

**Parallax** is the depth-dependent shift caused by camera translation. A plane's homography $H$ aligns that plane; the remaining shift of off-plane points carries information about relative depth.

<script setup>
import { ref, computed } from 'vue'
const baseline = ref(7)
const height = ref(1.1)
const c1 = { x: 1.5, y: 5 }
const planeY = 1.5
const normal = { x: -0.3, y: -Math.sqrt(0.91) }
const tangent = { x: -normal.y, y: normal.x }
const imageDistance = 0.55
const sx = x => 90 + x * 65
const sy = y => 380 - y * 55
const scene = computed(() => {
  const c2 = { x: c1.x + baseline.value, y: c1.y }
  const feature = { x: 5, y: planeY + height.value }
  const planeScale = (c1.y - planeY) / (c1.y - feature.y)
  const planePoint = { x: c1.x + (feature.x - c1.x) * planeScale, y: planeY }
  // These slider bounds keep both features in front of the fixed camera orientation.
  const project = point => {
    const dx = point.x - c2.x
    const dy = point.y - c2.y
    const scale = imageDistance / (dx * normal.x + dy * normal.y)
    return { x: c2.x + scale * dx, y: c2.y + scale * dy }
  }
  const observed = project(feature)
  const predicted = project(planePoint)
  const epipole = baseline.value === 0 ? null : project(c1)
  const centre = { x: c2.x + imageDistance * normal.x, y: c2.y + imageDistance * normal.y }
  const imagePoints = [observed, predicted, ...(epipole ? [epipole] : [])]
  const offsets = imagePoints.map(p => (p.x - centre.x) * tangent.x + (p.y - centre.y) * tangent.y)
  const endpoint = offset => ({ x: centre.x + offset * tangent.x, y: centre.y + offset * tangent.y })
  return {
    c2, feature, planePoint, observed, predicted, epipole,
    imageStart: endpoint(Math.min(...offsets) - 0.3),
    imageEnd: endpoint(Math.max(...offsets) + 0.3),
    coincident: baseline.value === 0 || height.value === 0,
  }
})
</script>

<div class="plx" role="figure" aria-label="Interactive plane and parallax">
  <p>Plane + parallax: <em>Hx₁</em> predicts the blue point; the feature actually projects to <em>x₂</em>, in gold.</p>
  <div class="plx-viewport">
  <svg viewBox="0 0 840 370" class="plx-scene" role="img" aria-label="Two cameras, a reference plane and the parallax of a feature above it">
    <defs>
      <marker id="plx-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 Z" fill="#F0C060" />
      </marker>
    </defs>
    <line :x1="sx(0)" :y1="sy(planeY)" :x2="sx(10)" :y2="sy(planeY)" stroke="#6EA8FE" stroke-width="1.6" />
    <text :x="sx(0)" :y="sy(planeY)+25" fill="#6EA8FE">plane Π</text>
    <line :x1="sx(c1.x)" :y1="sy(c1.y)" :x2="sx(scene.c2.x)" :y2="sy(scene.c2.y)" class="plx-ray" />
    <line :x1="sx(c1.x)" :y1="sy(c1.y)" :x2="sx(scene.planePoint.x)" :y2="sy(scene.planePoint.y)" class="plx-ray" />
    <line :x1="sx(scene.c2.x)" :y1="sy(scene.c2.y)" :x2="sx(scene.feature.x)" :y2="sy(scene.feature.y)" class="plx-ray" />
    <line :x1="sx(scene.c2.x)" :y1="sy(scene.c2.y)" :x2="sx(scene.planePoint.x)" :y2="sy(scene.planePoint.y)" class="plx-ray" />
    <line :x1="sx(scene.imageStart.x)" :y1="sy(scene.imageStart.y)" :x2="sx(scene.imageEnd.x)" :y2="sy(scene.imageEnd.y)" stroke="#E5826F" stroke-width="1.5" />
    <text :x="sx(scene.imageEnd.x)+10" :y="sy(scene.imageEnd.y)+5" fill="#E5826F">image 2</text>
    <circle v-if="baseline > 0" :cx="sx(c1.x)" :cy="sy(c1.y)" r="3.5" fill="#7BC47F" />
    <text v-if="baseline > 0" :x="sx(c1.x)-8" :y="sy(c1.y)-12" text-anchor="end" fill="#7BC47F">C₁</text>
    <circle :cx="sx(scene.c2.x)" :cy="sy(scene.c2.y)" r="3.5" fill="#E5826F" />
    <text :x="sx(scene.c2.x)+8" :y="sy(scene.c2.y)-12" fill="#E5826F">{{ baseline === 0 ? 'C₁ = C₂' : 'C₂' }}</text>
    <circle :cx="sx(scene.planePoint.x)" :cy="sy(scene.planePoint.y)" r="4" fill="#6EA8FE" />
    <text v-if="height > 0" :x="sx(scene.planePoint.x)+8" :y="sy(scene.planePoint.y)+25" fill="#6EA8FE">Xπ: C₁'s ray meets Π</text>
    <circle :cx="sx(scene.feature.x)" :cy="sy(scene.feature.y)" r="4.5" fill="#F0C060" />
    <text :x="sx(scene.feature.x)-30" :y="sy(scene.feature.y)+35" text-anchor="end" fill="#F0C060">{{ height === 0 ? 'X = Xπ: on the plane' : 'X: off the plane' }}</text>
    <circle v-if="scene.epipole" :cx="sx(scene.epipole.x)" :cy="sy(scene.epipole.y)" r="3" fill="#B39DDB" />
    <text v-if="scene.epipole" :x="sx(scene.epipole.x)-8" :y="sy(scene.epipole.y)+24" text-anchor="end" fill="#B39DDB">e′</text>
    <line v-if="!scene.coincident" :x1="sx(scene.predicted.x)" :y1="sy(scene.predicted.y)" :x2="sx(scene.observed.x)" :y2="sy(scene.observed.y)" stroke="#F0C060" stroke-width="1.8" marker-end="url(#plx-arrow)" />
    <circle data-point="predicted" :cx="sx(scene.predicted.x)" :cy="sy(scene.predicted.y)" r="3.5" fill="#6EA8FE" />
    <circle data-point="observed" :cx="sx(scene.observed.x)" :cy="sy(scene.observed.y)" r="3.5" fill="#F0C060" />
    <text :x="sx(scene.predicted.x)+8" :y="sy(scene.predicted.y)+22" fill="#6EA8FE">{{ scene.coincident ? 'Hx₁ = x₂' : 'Hx₁' }}</text>
    <text v-if="!scene.coincident" :x="sx(scene.observed.x)-8" :y="sy(scene.observed.y)-12" text-anchor="end" fill="#F0C060">x₂</text>
  </svg>
  </div>
  <div class="plx-controls">
    <label for="plx-baseline">camera baseline</label><input id="plx-baseline" type="range" min="0" max="7" step="0.1" v-model.number="baseline" /><output for="plx-baseline">{{ baseline.toFixed(1) }}</output>
    <label for="plx-height">feature height</label><input id="plx-height" type="range" min="0" max="1.8" step="0.1" v-model.number="height" /><output for="plx-height">{{ height.toFixed(1) }}</output>
  </div>

  <style scoped>
  .plx {
    margin: 16px 0;
    padding: 18px;
    background: #1B1F27;
    color: #E6EDF3;
    font-family: "Latin Modern Roman", "CMU Serif", "Computer Modern", serif;
  }
  .plx p {
    margin: 0 0 16px;
    line-height: 1.5;
  }
  .plx p:last-child {
    margin-bottom: 0;
    color: #9AA5B1;
    font-size: 0.9em;
  }
  .plx-viewport {
    overflow-x: auto;
    margin-bottom: 16px;
  }
  .plx-scene {
    display: block;
    width: 100%;
    min-width: 620px;
    font-family: inherit;
    font-size: 17px;
  }
  .plx-ray {
    stroke: #9AA5B1;
    stroke-width: 1;
    stroke-dasharray: 5 5;
  }
  .plx-controls {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) 3ch;
    gap: 8px 16px;
    align-items: center;
    margin-bottom: 16px;
  }
  .plx-controls input {
    width: 100%;
    min-width: 0;
    accent-color: #F0C060;
  }
  .plx-controls output {
    color: #F0C060;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  </style>
  <p v-if="baseline === 0"><strong>Cameras coincide:</strong> H's prediction and the observation agree at every height. There is no parallax and no epipole defined by translation; depth is unobservable.</p>
  <p v-else-if="height === 0"><strong>The feature is on the plane:</strong> X and Xπ coincide, so H predicts its position exactly even with separated cameras.</p>
  <p v-else><strong>Off the plane:</strong> C₁ sees X and Xπ along the same ray. C₂ sees them at different positions: the gold arrow is the residual after H.</p>
  <p>The purple epipole e′ is C₁ projected into image 2. Both matches lie on its line. Cameras keep a fixed orientation; slider distances use arbitrary scene units.</p>
</div>

$x_2\sim Hx_1+\rho(x_1)\,e'$: one scalar per point places its match on the epipolar line. Without calibration, this constrains projective structure, not depth in metres.

### Fundamental $F$ and essential $E$

Point to line. Holds for any rigid scene without needing depth; $E$ adds known intrinsics.

![Epipolar geometry](./fig-epipolar.svg)

### Dense flow $u(x)$

No geometry assumed. Covers non-rigid motion and anything the others miss.

![Dense flow with an independent mover](./fig-flow.svg)

### Side by side

![What each transform predicts in image 2](./predictions.svg)

## Open questions

- A homography is exact only for planar or distant scenes. Which level of the chain is the first that fits the measured correspondences?
- EO and IR are not a matched pair. Do shared units emerge at all without cross-modal supervision?
- Should the checkerboard calibration be an independent baseline?
- Where does the visualization show up in the TUI: tile, overlay or log?
