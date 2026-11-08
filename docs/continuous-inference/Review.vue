<script setup>
import { computed, onBeforeUnmount, ref } from "vue";
import capture from "./capture.json";
import comparisonVideo from "./comparison.mp4?url";

const mode = ref("waves");
const frame = ref(0);
const hidden = ref(0);
const sensory = ref(0);
const matrix = ref("sensory_weights");
const weightMetric = ref("step");
const settings = capture.settings;
const records = computed(() => capture.snapshots.filter(record => record.mode === mode.value));
const legacyRecords = computed(() => capture.legacy_snapshots.filter(record => record.mode === mode.value));
const selected = computed(() => records.value[Math.min(frame.value, records.value.length - 1)]);
const video = ref(null);
const playing = ref(false);
const playbackError = ref("");
let animation = null;

function cancelFrame() {
  if (animation !== null) video.value.cancelVideoFrameCallback(animation);
  animation = null;
}

function syncFrame(time = video.value.currentTime) {
  frame.value = Math.min(settings.frames - 1, Math.floor(time * settings.fps + 1e-6));
}

function presentedFrame() {
  animation = video.value.requestVideoFrameCallback((now, metadata) => {
    animation = null;
    syncFrame(metadata.mediaTime);
    if (!video.value.paused) presentedFrame();
  });
}

function onPlay() {
  playing.value = true;
  playbackError.value = "";
  cancelFrame();
  presentedFrame();
}

function onPause() {
  playing.value = false;
  cancelFrame();
}

function pause() {
  video.value.pause();
}

async function play() {
  if (frame.value === settings.frames - 1) restart();
  try {
    await video.value.play();
  } catch (error) {
    onPause();
    playbackError.value = `Playback failed: ${error.message}`;
  }
}

function seek(index) {
  pause();
  video.value.currentTime = index / settings.fps;
}

function restart() {
  seek(0);
  frame.value = 0;
}

onBeforeUnmount(cancelFrame);
const f = value => value.toFixed(4);
const numeric = value => value !== 0 && Math.abs(value) < .0001 ? value.toExponential(2) : f(value);
const signed = value => `${value >= 0 ? "+" : ""}${numeric(value)}`;
const norm = values => Math.sqrt(values.reduce((sum, value) => sum + value*value, 0));
const hParts = computed(() => [
  ["State decay", selected.value.h_decay[hidden.value]],
  ["Local hidden residual", selected.value.h_local[hidden.value]],
  ["Recurrent feedback", selected.value.h_recurrent[hidden.value]],
  ["Sensory feedback", selected.value.h_sensory[hidden.value]],
]);
const zParts = computed(() => [
  ["State decay", selected.value.z_decay[sensory.value]],
  ["Coupling residual", selected.value.z_coupling[sensory.value]],
  ["Observation residual", selected.value.z_observation[sensory.value]],
]);
const matrixKey = computed(() =>
  ({ canvas_weights: "Wc", recurrent_weights: "Wh", sensory_weights: "Ws" })[matrix.value]
);
const weightSources = computed(() => [
  {
    id: "legacy", title: "Old · per-pixel inference", records: legacyRecords.value,
    available: matrix.value !== "recurrent_weights",
    description: `${legacyRecords.value[0].Wc.length} hidden states per pixel; shared RGB matrices, all entries shown.`,
  },
  {
    id: "continuous", title: "New · continuous inference", records: records.value,
    available: true,
    description: `${settings.hidden_nodes} hidden states for the whole image; Wc and Ws show nine sampled RGB coordinates.`,
  },
]);
function cumulativeChange(record, initial, r, c) {
  return record[matrixKey.value][r][c] - initial[matrixKey.value][r][c]
    + record.weight_updates[matrix.value][r][c];
}
function weightValues(record, initial) {
  if (weightMetric.value === "weight") return record[matrixKey.value];
  if (weightMetric.value === "step") return record.weight_updates[matrix.value];
  return record[matrixKey.value].map((row, r) =>
    row.map((value, c) => cumulativeChange(record, initial, r, c))
  );
}
const weightPanels = computed(() => weightSources.value.map(source => {
  const record = source.records[frame.value];
  const initial = source.records[0];
  return {
    ...source, record, initial,
    values: source.available ? weightValues(record, initial) : null,
  };
}));
const weightColorLimit = computed(() => {
  let maximum = 0;
  for (const source of weightSources.value) {
    if (!source.available) continue;
    for (const record of source.records) {
      for (const row of weightValues(record, source.records[0])) {
        for (const value of row) maximum = Math.max(maximum, Math.abs(value));
      }
    }
  }
  return maximum;
});
function weightTitle(panel, r, c) {
  const before = panel.record[matrixKey.value][r][c];
  const update = panel.record.weight_updates[matrix.value][r][c];
  return `row ${r}, col ${c}: W_before=${before}; applied ΔW=${update}; W_after=${before+update}; cumulative ΔW=${cumulativeChange(panel.record, panel.initial, r, c)}`;
}
const curve = (values, maximum=.6) => values.map((value, index) =>
  `${48 + index/Math.max(1, values.length-1)*580},${168 - value/maximum*135}`
).join(" ");
const microCurve = computed(() => {
  const values = selected.value.micro_energy;
  const minimum = Math.min(...values);
  const span = Math.max(...values)-minimum;
  return values.map((value, index) =>
    `${48 + index/Math.max(1, values.length-1)*580},${168 - (value-minimum)/Math.max(span,1e-12)*135}`
  ).join(" ");
});
const color = value => {
  const magnitude = weightColorLimit.value === 0 ? 0 : Math.abs(value)/weightColorLimit.value;
  return value >= 0 ? `rgba(195, 61, 93, ${.1 + .8*magnitude})`
    : `rgba(38, 122, 185, ${.1 + .8*magnitude})`;
};
const labels = ["Black background", "Red rectangle", "Blue rectangle"];
</script>

<template>
  <section class="inference-review" aria-label="Continuous inference diagnostics">
    <section class="comparison" aria-label="Current camera dynamics comparison">
      <h2>Current comparison · per-pixel and updated continuous inference</h2>
      <video ref="video" controls preload="metadata" playsinline
        :src="comparisonVideo"
        aria-label="Current per-pixel and plain continuous solver comparison, with and without waves"
        @play="onPlay" @pause="onPause" @ended="onPause" @seeked="syncFrame()"
        @error="playbackError = 'The comparison video could not be loaded.'">
        <a :href="comparisonVideo">Open the comparison video</a>
      </video>
      <p>Fresh {{ settings.frames/settings.fps }}-second run, same camera input and learning enabled in both models.
        The graph uses the exact continuous-model telemetry for each displayed video frame at {{ settings.fps }} fps.
        Play, pause and seek here or below; both views share one clock.</p>
    </section>
    <h2>Continuous solver · synchronized network diagnostics</h2>
    <div class="controls">
      <button type="button" :aria-pressed="playing" @click="playing ? pause() : play()">
        {{ playing ? "Pause" : "Play" }}
      </button>
      <button type="button" @click="restart">Restart</button>
      <label>Dynamics
        <select v-model="mode" @change="pause"><option value="waves">With waves</option><option value="held">Waves held</option></select>
      </label>
      <label class="scrubber">Recorded time: {{ f(selected.time) }} s
        <input :value="frame" type="range" min="0" :max="settings.frames-1" step="1"
          @input="seek(Number($event.target.value))">
      </label>
      <output>Energy {{ selected.energy.toFixed(2) }} · forecast RMSE {{ f(selected.forecast_rmse) }}</output>
    </div>
    <p v-if="playbackError" role="alert">{{ playbackError }}</p>
    <p class="settings">
      Linear visual output · unit precisions · state rate {{ settings.rate }} · learning rate {{ settings.learning_rate }} ·
      canvas rate {{ settings.canvas_rate }} · decay 1/τ = {{ (1/settings.tau).toFixed(1) }} ·
      {{ settings.inference_steps }} microsteps/frame · no clipping, metric rescaling or projection
    </p>
    <div class="panes">
      <section class="pane topology">
        <h2>Canvas → hidden → sensory</h2>
        <p>Schematic of dense connectivity. Canvas and sensory boxes group {{ settings.sensory_nodes.toLocaleString() }} RGB coordinates.
          The six hidden nodes are literal. Dashed arrows show gradient contributions; updates use the negative total.</p>
        <svg viewBox="0 0 720 435" role="img" aria-label="Canvas above six recurrent hidden nodes above grouped sensory states">
          <defs>
            <marker id="forward" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0 0L10 5L0 10Z" fill="#37699f" />
            </marker>
            <marker id="backward" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0 0L10 5L0 10Z" fill="#b96124" />
            </marker>
          </defs>
          <rect x="176" y="12" width="368" height="66" rx="9" class="group" />
          <text x="360" y="37" class="center">Canvas c · linear readout</text>
          <text x="360" y="59" class="center small">Root correction = {{ settings.canvas_rate }} Wcᵀ e_h</text>
          <path d="M212 82L212 140L600 140" class="forward" />
          <text x="360" y="107" class="center small">Wc · dense root-to-hidden weights</text>
          <path d="M651 187L651 46L552 46" class="backward" />
          <text x="645" y="123" text-anchor="end" class="small">∂E/∂c</text>
          <g v-for="(value, index) in selected.h" :key="index"
            tabindex="0" role="button" :aria-label="`Select hidden node ${index}`"
            @click="hidden=index" @keydown.enter="hidden=index" @keydown.space.prevent="hidden=index"
            class="select-node">
            <path :d="`M${90+index*105} 140L${90+index*105} 156`" class="forward" />
            <circle :cx="90+index*105" cy="189" r="33" :class="['node', { chosen: hidden===index }]" />
            <text :x="90+index*105" y="186" class="center">h{{ index }}</text>
            <text :x="90+index*105" y="204" class="center small">{{ f(value) }}</text>
            <path :d="`M${90+index*105} 227L${90+index*105} 274`" class="forward" />
            <text :x="90+index*105" y="245" class="center tiny">∇ {{ signed(selected.grad_h[index]) }}</text>
          </g>
          <path d="M71 166C23 103 662 103 634 166" class="recurrent" />
          <text x="360" y="151" class="center tiny">Wh connects every hidden pair · diagonal zero</text>
          <rect x="111" y="278" width="498" height="69" rx="9" class="group" />
          <text x="360" y="301" class="center">Sensory latent z · inferred from three energy terms</text>
          <text x="360" y="323" class="center small">Incoming prediction Ws tanh(h) · visual output z</text>
          <path d="M616 311L678 311L678 196L649 196" class="backward" />
          <text x="655" y="268" text-anchor="end" class="tiny">−Jsᵀ e_z</text>
          <path d="M315 393L315 351" class="backward" />
          <rect x="176" y="394" width="368" height="34" rx="7" class="evidence" />
          <text x="360" y="416" class="center small">Observed image y · all pixels observed · no fovea</text>
          <text x="456" y="371" class="center tiny">−(y − z)</text>
        </svg>
        <p>Js = Ws diag(tanh′(h)): hidden feedback includes the hidden activation slope.</p>
        <div class="metrics">
          <span>‖∇h‖ {{ numeric(norm(selected.grad_h)) }}</span>
          <span>Canvas peak |c| {{ selected.canvas_peak.toExponential(2) }}</span>
          <span>Learning uses settled states and pre-update weights.</span>
        </div>
      </section>
      <section class="pane gradients">
        <h2>Residuals → gradients → updates</h2>
        <label>Hidden node <select v-model.number="hidden">
          <option v-for="index in 6" :value="index-1">h{{ index-1 }}</option>
        </select></label>
        <table>
          <thead><tr><th>Contribution to ∂E/∂h{{ hidden }}</th><th>Signed value</th></tr></thead>
          <tbody><tr v-for="[label,value] in hParts" :key="label"><td>{{ label }}</td><td>{{ signed(value) }}</td></tr>
            <tr class="total"><td>Total gradient</td><td>{{ signed(selected.grad_h[hidden]) }}</td></tr>
          </tbody>
        </table>
        <p>Next microstep Δh = −{{ settings.rate }}/(30 × {{ settings.inference_steps }}) × ∂E/∂h.
          Values above describe the settled state's next direction, not the preceding microstep's delta.</p>
        <label>Sampled RGB coordinate
          <select v-model.number="sensory">
            <option v-for="(index, position) in settings.indices" :value="position">
              {{ labels[Math.floor(position/3)] }} · {{ ["R","G","B"][position%3] }} · z{{ index }}
            </option>
          </select>
        </label>
        <p>Observation {{ f(selected.observed[sensory]) }} · prediction {{ f(selected.predicted[sensory]) }}
          · latent z {{ f(selected.z[sensory]) }}</p>
        <table>
          <thead><tr><th>Contribution to ∂E/∂z</th><th>Signed value</th></tr></thead>
          <tbody><tr v-for="[label,value] in zParts" :key="label"><td>{{ label }}</td><td>{{ signed(value) }}</td></tr>
            <tr class="total"><td>Total gradient</td><td>{{ signed(selected.grad_z[sensory]) }}</td></tr>
          </tbody>
        </table>
        <p>At an equilibrium the total gradient can be zero while prediction error stays large.
          Opposite signs here show which terms balance.</p>
      </section>
      <section class="pane weights">
        <h2>Connection learning · old and new</h2>
        <label>Weight matrix <select v-model="matrix">
          <option value="canvas_weights">Wc: canvas → hidden</option>
          <option value="recurrent_weights">Wh: recurrent hidden connections</option>
          <option value="sensory_weights">Ws: hidden → sensory</option>
        </select></label>
        <label>Cell values <select v-model="weightMetric">
          <option value="step">Applied ΔW this frame</option>
          <option value="cumulative">Cumulative ΔW since start</option>
          <option value="weight">Pre-learning weight W</option>
        </select></label>
        <p>Both models use this video's frame and dynamics selection. Updates are measured as post-learning minus
          pre-learning weights, including the old engine's clipping. Cumulative changes include this frame's update.</p>
        <p>Red = positive; blue = negative. Shared color limits ±{{ numeric(weightColorLimit) }},
          fixed across the selected dynamics recording and cell-value mode.
          Hover for exact before/after weights and changes; small values use scientific notation.</p>
        <div class="weight-comparison">
          <section v-for="panel in weightPanels" :key="panel.id" :class="`${panel.id}-weights`">
            <h3>{{ panel.title }}</h3>
            <p>{{ panel.description }}</p>
            <template v-if="panel.available">
              <div class="matrix-scroll">
                <table class="matrix">
                  <tbody><tr v-for="(row, r) in panel.values" :key="r">
                    <td v-for="(value, c) in row" :key="c" :style="{background:color(value)}"
                      :title="weightTitle(panel, r, c)">{{ signed(value) }}</td>
                  </tr></tbody>
                </table>
              </div>
              <p>Full-matrix ‖applied ΔW‖ = {{ numeric(panel.record.weight_update_norms[matrix]) }}.</p>
            </template>
            <p v-else>No same-layer recurrent weight matrix in this reference camera model.</p>
          </section>
        </div>
        <p>Continuous full-matrix ‖∂E/∂W‖ = {{ numeric(selected.weight_gradient_norms[matrix]) }}.
          W ← W − {{ settings.learning_rate }} ∂E/∂W.</p>
      </section>
      <section class="pane curves">
        <h2>Pre-evidence forecast error · whole-image RGB RMSE</h2>
        <svg viewBox="0 0 680 210" role="img" aria-label="Whole-image forecast RMSE over the recorded run">
          <path d="M48 22V168H628" class="axis" />
          <text x="12" y="39" class="small">0.6</text><text x="19" y="170" class="small">0</text>
          <text x="48" y="190" class="small">0 s</text>
          <text x="590" y="190" class="small">{{ settings.frames/30 }} s</text>
          <polyline :points="curve(capture.curves.waves)" class="line waves" />
          <polyline :points="curve(capture.curves.held)" class="line held" />
          <path :d="`M${48+selected.frame/(settings.frames-1)*580} 22V168`" class="cursor" />
        </svg>
        <p><span class="waves-label">Blue: waves</span> · <span class="held-label">Orange dashed: held</span>.
          Overlapping curves mean their errors are indistinguishable at this scale.</p>
        <h3>Within-frame energy descent</h3>
        <svg viewBox="0 0 680 195" role="img" aria-label="Energy decreases across the selected frame's inference microsteps">
          <path d="M48 22V168H628" class="axis" />
          <polyline :points="microCurve" class="line waves" />
          <text x="48" y="188" class="small">Interval-start energy {{ selected.micro_energy[0].toFixed(4) }}</text>
          <text x="628" y="188" text-anchor="end" class="small">Settled {{ selected.energy.toFixed(4) }}</text>
        </svg>
        <p>The vertical axis spans this frame's minimum and maximum energy. A descent is not a claim of zero residual.</p>
      </section>
      <section class="pane diagnostics">
        <h2>Basic model checks</h2>
        <h3>Linear vs ReLU · 30 s with learning</h3>
        <table><thead><tr><th>Observed RGB</th><th>Linear prediction</th><th>ReLU prediction</th></tr></thead>
          <tbody><tr v-for="(result, index) in capture.black_gray_white" :key="result.observed">
            <td>{{ result.observed }}</td><td>{{ f(result.prediction) }}</td>
            <td>{{ f(capture.relu_black_gray_white[index].prediction) }}</td>
          </tr></tbody>
        </table>
        <p class="finding">Linear output predicts black exactly, but decay and coupling attenuate bright predictions.
          ReLU stalls at the unchanged zero initialization: its derivative at zero is zero, so observation error
          sends no initial state drive. No bias, jitter or positive warm start is added.</p>
        <h3>XOR · output withheld at evaluation</h3>
        <table><thead><tr><th>Inputs</th><th>Target</th><th>Prediction</th></tr></thead>
          <tbody><tr v-for="result in capture.xor.cases" :key="`${result.a}-${result.b}`">
            <td>{{ result.a }}, {{ result.b }}</td><td>{{ result.target }}</td><td>{{ f(result.prediction) }}</td>
          </tr></tbody>
        </table>
        <p class="finding">XOR is not solved. RMSE {{ f(capture.xor.rmse) }}.</p>
        <details><summary>Evaluation protocol</summary><p>{{ capture.xor.protocol }}</p></details>
      </section>
    </div>
    <p class="timing">Playback uses one recorded snapshot per video frame at {{ settings.fps }} fps.
      Recorded experiment, not a live app connection. Each simulation frame advances inference by 1/30 s
      and waves by one CFL step ({{ settings.wave_dt.toExponential(3) }} s); these clocks differ.
      No hardware camera, fovea, gaze actions or external canvas forcing is used in the camera run.</p>
  </section>
</template>

<style scoped>
.inference-review { color: var(--vp-c-text-1); }
.comparison video { display:block; width:100%; max-width:1104px; max-height:85vh; margin:auto; background:#080808; }
.comparison p { font-size:13px; line-height:1.65; }
.controls { display:flex; flex-wrap:wrap; gap:16px; align-items:center; padding:12px; background:var(--vp-c-bg-soft); border-radius:8px; }
.controls label { display:flex; gap:8px; align-items:center; }
.controls button { border:1px solid var(--vp-c-divider); border-radius:5px; padding:4px 12px; background:var(--vp-c-bg); }
.controls button[aria-pressed="true"] { border-color:#428ec6; }
.scrubber { flex:1; min-width:250px; }
.scrubber input { flex:1; min-width:100px; }
select { max-width:100%; border:1px solid var(--vp-c-divider); border-radius:5px; padding:4px 8px; background:var(--vp-c-bg); color:inherit; }
.settings, .timing { font-size:13px; line-height:1.6; }
.panes { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
.pane { min-width:0; padding:16px; border:1px solid var(--vp-c-divider); border-radius:10px; }
.pane h2 { border:0; margin:0 0 12px; padding:0; font-size:19px; }
.pane h3 { font-size:16px; margin:14px 0 8px; }
.pane p { font-size:13px; line-height:1.65; margin:10px 0; }
.pane label { font-size:13px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
svg { display:block; width:100%; overflow:visible; }
svg text { fill:var(--vp-c-text-1); font-family:inherit; font-size:17px; }
.center { text-anchor:middle; }
svg .small { font-size:14px; }
svg .tiny { font-size:12px; }
.group, .node { fill:var(--vp-c-bg-soft); stroke:#5484b6; stroke-width:2; }
.chosen { stroke:#cd752e; stroke-width:4; }
.evidence { fill:var(--vp-c-bg-soft); stroke:#b96124; stroke-width:2; }
.forward { stroke:#37699f; stroke-width:1.6; fill:none; marker-end:url(#forward); }
.backward { stroke:#b96124; stroke-width:2; fill:none; stroke-dasharray:6 4; marker-end:url(#backward); }
.recurrent { stroke:#68835a; stroke-width:1.8; fill:none; marker-end:url(#forward); }
.select-node { cursor:pointer; }
.metrics { display:flex; flex-wrap:wrap; gap:12px; font-size:12px; }
.pane table { width:100%; display:table; margin:12px 0; font-size:13px; }
.pane td, .pane th { padding:6px 9px; }
.pane td:last-child { font-variant-numeric:tabular-nums; }
.total { font-weight:700; }
.matrix-scroll { overflow-x:auto; }
.weights { grid-column:1 / -1; }
.weight-comparison { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:24px; }
.weight-comparison > section { min-width:0; }
.matrix { white-space:nowrap; }
.matrix td { padding:5px; font-size:11px; text-align:center; }
.axis { fill:none; stroke:var(--vp-c-text-2); stroke-width:1; }
.line { fill:none; stroke-width:2.5; }
.waves { stroke:#428ec6; }
.held { stroke:#dc8a35; stroke-dasharray:7 4; }
.cursor { stroke:var(--vp-c-text-2); stroke-width:1; stroke-dasharray:3 4; }
.waves-label { color:#428ec6; }
.held-label { color:#dc8a35; }
.finding { border-left:3px solid #cd752e; padding-left:10px; }
@media (max-width:1000px) { .panes, .weight-comparison { grid-template-columns:1fr; } }
</style>
