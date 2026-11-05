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
const t = ref(0.15)
const ex = ref(2.6)
const ey = ref(1.9)
const pts = [[0.8,0.8,0],[1.0,3.0,0],[4.1,3.1,0],[4.2,0.7,0],[1.4,2.4,1],[3.8,2.2,1.6],[3.4,0.9,2.4]]
const X = x => 40 + x * 60
const Y = y => 250 - y * 60
const moved = computed(() => pts.map(([px, py, h]) => ({
  px, py, h,
  mx: px + t.value * h * (px - ex.value),
  my: py + t.value * h * (py - ey.value),
})))
</script>

<div class="plx" role="figure" aria-label="Interactive plane and parallax">
  <p><strong>Drag the baseline slider.</strong> Hollow dots show H's prediction; filled dots show the observed positions. The four points on the plane stay aligned.</p>
  <svg viewBox="-90 -15 540 285" class="plx-scene" aria-label="Plane-aligned predictions and observed points">
    <rect :x="X(0)" :y="Y(3.8)" :width="5*60" :height="3.8*60" fill="#392e32" stroke="#E5826F" stroke-width="1.5" />
    <g v-for="(p,i) in moved" :key="i">
      <line v-if="p.h>0" :x1="X(ex)" :y1="Y(ey)" :x2="X(p.px)" :y2="Y(p.py)" stroke="#B39DDB" stroke-dasharray="4 3" stroke-width="0.8" />
      <line v-if="p.h>0" :x1="X(p.px)" :y1="Y(p.py)" :x2="X(p.mx)" :y2="Y(p.my)" stroke="#F0C060" stroke-width="1.5" />
      <circle :cx="X(p.px)" :cy="Y(p.py)" r="2.8" fill="#1B1F27" stroke="#6EA8FE" stroke-width="1" />
      <circle v-if="p.h>0" :cx="X(p.mx)" :cy="Y(p.my)" r="2.8" fill="#F0C060" />
    </g>
    <path :d="`M ${X(ex)-3} ${Y(ey)-3} l 6 6 m -6 0 l 6 -6`" stroke="#B39DDB" stroke-width="1" />
    <text :x="X(ex)+6" :y="Y(ey)+16" fill="#B39DDB" font-size="13" font-style="italic">e&#8242;</text>
  </svg>
  <div class="plx-controls">
    <label for="plx-baseline">baseline <em>t</em></label><input id="plx-baseline" type="range" min="0" max="0.5" step="0.01" v-model.number="t" /><output for="plx-baseline">{{ t.toFixed(2) }}</output>
    <label for="plx-epipole-x">epipole <em>x</em></label><input id="plx-epipole-x" type="range" min="-3" max="8" step="0.1" v-model.number="ex" /><output for="plx-epipole-x">{{ ex.toFixed(1) }}</output>
    <label for="plx-epipole-y">epipole <em>y</em></label><input id="plx-epipole-y" type="range" min="-3" max="7" step="0.1" v-model.number="ey" /><output for="plx-epipole-y">{{ ey.toFixed(1) }}</output>
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
  .plx-scene {
    display: block;
    width: 100%;
    margin-bottom: 16px;
    font-family: inherit;
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
  <p v-if="t === 0"><strong>No translation:</strong> prediction and observation coincide at every depth. H is exact for pure rotation, but depth cannot be recovered.</p>
  <p v-else><strong>{{ t < 0.25 ? 'Small baseline:' : 'Larger baseline:' }}</strong> off-plane points {{ t < 0.25 ? 'separate slightly from' : 'move farther from' }} H's predictions. Their residuals lie along lines through the epipole e'. H still aligns the plane.</p>
  <p>Move the epipole to change those lines. This is a schematic model: the baseline scales the residual, not a calibrated distance in metres.</p>
</div>

$x_2\sim Hx_1+\rho(x_1)\,e'$: one scalar per point places its match on the epipolar line. Without calibration, this constrains projective structure, not depth in metres.

![Parallax of a point off the plane](./fig-parallax.svg)

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
