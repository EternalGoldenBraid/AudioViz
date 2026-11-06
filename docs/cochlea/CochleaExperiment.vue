<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { activity, AMPLITUDE, BURST_END, DURATION, FREQUENCIES, NODES, SAMPLE_RATE, simulate, stimulus } from './simulation.mjs'

const frequency = ref(371)
const qualityFactor = ref(16)
const correctionRate = ref(0.5)
const mode = ref('burst')
const sample = ref(Math.round(0.06 * SAMPLE_RATE))
const selected = ref(5)
const playing = ref(false)
let animation
const trace = computed(() => simulate(stimulus(frequency.value, mode.value === 'burst'), {
  qualityFactor: qualityFactor.value, correctionRate: correctionRate.value,
}))
const lastSample = computed(() => trace.value.raw.length - 1)
const time = computed(() => sample.value / SAMPLE_RATE)
const current = computed(() => ({
  raw: trace.value.raw[sample.value],
  prediction: trace.value.prediction[sample.value],
  error: trace.value.error[sample.value],
}))
const states = computed(() => FREQUENCIES.map((frequency, i) => {
  const offset = sample.value * NODES + i
  return {
    frequency, x: trace.value.beforeX[offset], q: trace.value.beforeQ[offset],
    correctedX: trace.value.afterX[offset], correctedQ: trace.value.afterQ[offset],
  }
}))
const chosen = computed(() => states.value[selected.value])
const stateScale = computed(() => Math.max(0.4, Math.ceil(trace.value.maximumRadius * 10) / 10))
const waveScale = computed(() => {
  let peak = 0.5
  for (const values of [trace.value.prediction, trace.value.error]) {
    for (const value of values) peak = Math.max(peak, Math.abs(value))
  }
  return Math.ceil(peak * 10) / 10
})
const signed = value => `${value >= 0 ? '+' : ''}${value.toFixed(4)}`
const nodePoint = value => 30 * value / stateScale.value
const left = 48
const width = 620
const overviewX = seconds => left + width * seconds / DURATION
const windowBounds = computed(() => {
  const count = Math.round(0.008 * SAMPLE_RATE)
  const start = Math.max(0, Math.min(sample.value - Math.floor(count / 2), trace.value.raw.length - count))
  return { start, end: start + count - 1 }
})
const waveX = n => left + width * (n - windowBounds.value.start) / (windowBounds.value.end - windowBounds.value.start)
const waveY = (value, centre) => centre - 35 * value / waveScale.value
const path = (values, centre) => {
  const { start, end } = windowBounds.value
  const points = []
  for (let n = start; n <= end; n++) points.push(`${n === start ? 'M' : 'L'}${waveX(n).toFixed(2)},${waveY(values[n], centre).toFixed(2)}`)
  return points.join(' ')
}
const paths = computed(() => ({
  raw: path(trace.value.raw, 65), prediction: path(trace.value.prediction, 65),
  error: path(trace.value.error, 172),
}))
const heat = amplitude => {
  const level = Math.max(0, Math.min(1, (20 * Math.log10(Math.max(amplitude, 1e-8)) + 80) / 80))
  const stops = [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]]
  const segment = Math.min(3, Math.floor(level * 4))
  const fraction = level * 4 - segment
  return `rgb(${stops[segment].map((v, i) => Math.round(v + fraction * (stops[segment + 1][i] - v))).join(',')})`
}
const history = computed(() => activity(trace.value).map((row, i) => row.map((amplitude, k) => ({
  x: overviewX(k * 32 / SAMPLE_RATE), y: 30 + (NODES - 1 - i) * 7,
  color: heat(amplitude),
}))))
const explanation = computed(() => {
  if (correctionRate.value === 0) return 'Correction is off. Starting from rest, every oscillator stays at rest: prediction is zero and the entire input remains as error.'
  if (sample.value === 0) return 'The bank starts at rest. It predicts before the first sample corrects any position; no future input is used.'
  if (mode.value === 'burst' && time.value >= BURST_END) return 'The input is now silent. Stored state still evolves, while the error from observing zero keeps correcting positions. This tail includes both damping and inference; it is not free decay.'
  return 'The blue prediction is the sum of all blue horizontal positions below. One raw-audio error then shifts every position by the same amount; quadrature is unchanged by this correction.'
})
function stop() {
  playing.value = false
  if (animation !== undefined) cancelAnimationFrame(animation)
}
function play() {
  if (playing.value) { stop(); return }
  if (sample.value === lastSample.value) sample.value = 0
  const origin = performance.now()
  const start = sample.value
  playing.value = true
  const frame = now => {
    sample.value = Math.min(lastSample.value, start + Math.floor((now - origin) * SAMPLE_RATE / 1e6))
    if (sample.value === lastSample.value) { stop(); return }
    animation = requestAnimationFrame(frame)
  }
  animation = requestAnimationFrame(frame)
}
function inspect(seconds) {
  stop()
  sample.value = Math.min(lastSample.value, Math.round(seconds * SAMPLE_RATE))
}
watch([frequency, qualityFactor, correctionRate, mode], stop)
onBeforeUnmount(stop)
</script>

<template>
  <section class="cochlea-experiment" aria-label="Interactive cochlea simulation" :data-sample="sample">
    <header>
      <strong>A tone enters. The bank predicts.</strong>
      <p>Each change simulates 160 ms from rest at 22,050 samples/s. Every view below reads that same run.</p>
    </header>
    <div class="experiment-controls">
      <label for="cochlea-input">Input</label>
      <select id="cochlea-input" v-model="mode">
        <option value="burst">80 ms tone burst</option>
        <option value="tone">Continuous tone</option>
      </select>
      <span class="control-unit">peak {{ AMPLITUDE }}</span>
      <label for="cochlea-frequency">Tone frequency</label>
      <input id="cochlea-frequency" v-model.number="frequency" type="range" min="80" max="8000" step="1" />
      <output for="cochlea-frequency">{{ frequency }} Hz</output>
      <label for="cochlea-q">Quality factor Q</label>
      <input id="cochlea-q" v-model.number="qualityFactor" type="range" min="2" max="32" step="0.5" />
      <output for="cochlea-q">{{ qualityFactor }}</output>
      <label for="cochlea-correction">Error correction &eta;</label>
      <input id="cochlea-correction" v-model.number="correctionRate" type="range" min="0" max="1" step="0.05" />
      <output for="cochlea-correction">{{ correctionRate.toFixed(2) }}</output>
    </div>
    <p class="explanation">{{ explanation }}</p>

    <div class="timeline">
      <label for="cochlea-time">Inspect time</label>
      <input id="cochlea-time" v-model.number="sample" type="range" min="0" :max="lastSample" step="1" @input="stop" />
      <output for="cochlea-time">{{ (time * 1000).toFixed(2) }} ms</output>
    </div>
    <div class="transport">
      <button type="button" @click="play">{{ playing ? 'Pause' : 'Play 1000x slower' }}</button>
      <button type="button" @click="inspect(0)">Start</button>
      <button type="button" @click="inspect(0.06)">Driven</button>
      <button v-if="mode === 'burst'" type="button" @click="inspect(0.084)">After sound</button>
    </div>

    <figure>
      <figcaption>Raw audio and its causal prediction <span>8 ms detail around the cursor</span></figcaption>
      <p class="scroll-hint">Swipe the plots horizontally to see the whole time interval.</p>
      <div class="plot-viewport">
      <svg class="trace" viewBox="0 0 704 238" role="img" aria-label="Raw input and prediction, with their signed error below">
        <text x="48" y="18" class="observed">Input</text>
        <text x="114" y="18" class="predicted">Prediction</text>
        <text x="668" y="18" text-anchor="end">shared amplitude &plusmn;{{ waveScale.toFixed(1) }}</text>
        <g v-for="centre in [65, 172]" :key="centre">
          <line x1="48" :y1="centre" x2="668" :y2="centre" class="zero" />
          <text x="39" :y="centre + 4" text-anchor="end">0</text>
          <line :x1="waveX(sample)" :y1="centre - 38" :x2="waveX(sample)" :y2="centre + 38" class="cursor" />
        </g>
        <path :d="paths.raw" class="raw-line" />
        <path :d="paths.prediction" class="prediction-line" />
        <text x="48" y="126" class="residual">Error = input &minus; prediction</text>
        <path :d="paths.error" class="error-line" />
        <circle :cx="waveX(sample)" :cy="waveY(current.raw, 65)" r="3" class="raw-dot" />
        <circle :cx="waveX(sample)" :cy="waveY(current.prediction, 65)" r="3" class="prediction-dot" />
        <circle :cx="waveX(sample)" :cy="waveY(current.error, 172)" r="3" class="error-dot" />
        <text x="48" y="228">{{ (windowBounds.start / SAMPLE_RATE * 1000).toFixed(2) }} ms</text>
        <text x="668" y="228" text-anchor="end">{{ (windowBounds.end / SAMPLE_RATE * 1000).toFixed(2) }} ms</text>
      </svg>
      </div>
      <div class="sample-values">
        <span class="observed">Input <output data-value="input">{{ signed(current.raw) }}</output></span>
        <span class="predicted">Sum of positions <output data-value="prediction">{{ signed(current.prediction) }}</output></span>
        <span class="residual">Error <output data-value="error">{{ signed(current.error) }}</output></span>
      </div>
    </figure>

    <figure>
      <figcaption>Sixteen actual oscillator states <span>select one to inspect</span></figcaption>
      <p class="legend"><span class="predicted">Blue: before hearing this sample.</span> <span class="corrected">Amber: after its error correction.</span> Horizontal = position x; vertical = quadrature q. Circle radius = {{ stateScale.toFixed(1) }} amplitude units, fixed for this run.</p>
      <div class="oscillators">
        <button v-for="(state, i) in states" :key="i" type="button" :class="{ selected: i === selected }"
                :aria-pressed="i === selected" :aria-label="`Oscillator ${i + 1}, ${state.frequency.toFixed(0)} Hz`" @click="selected = i">
          <span>{{ i + 1 }}</span>
          <svg viewBox="-36 -36 72 72" aria-hidden="true">
            <circle r="30" class="orbit" />
            <path d="M-30,0H30 M0,-30V30" class="axes" />
            <line x1="0" y1="0" :x2="nodePoint(state.x)" :y2="-nodePoint(state.q)" class="prediction-line" />
            <line :x1="nodePoint(state.x)" :y1="-nodePoint(state.q)" :x2="nodePoint(state.correctedX)" :y2="-nodePoint(state.correctedQ)" class="correction-line" />
            <circle :cx="nodePoint(state.correctedX)" :cy="-nodePoint(state.correctedQ)" r="2.2" class="corrected-dot" />
          </svg>
          <span>{{ state.frequency.toFixed(0) }} Hz</span>
        </button>
      </div>
      <p class="node-detail">
        <strong>Node {{ selected + 1 }}:</strong>
        x = <span class="predicted">{{ signed(chosen.x) }}</span>
        &rarr; <span class="corrected">{{ signed(chosen.correctedX) }}</span>;
        q = {{ signed(chosen.q) }}.
        Its position shift is &eta; &times; error / 16 = {{ signed(correctionRate * current.error / NODES) }}.
        Free-decay time constant Q / (&pi; f) = {{ (1000 * qualityFactor / (Math.PI * chosen.frequency)).toFixed(1) }} ms.
      </p>
      <button type="button" class="tune" @click="frequency = Math.round(chosen.frequency)">Tune input to node {{ selected + 1 }}</button>
    </figure>

    <figure>
      <figcaption>State amplitude through the whole run <span>RMS of x and q; &minus;80 to 0 dB re 1</span></figcaption>
      <div class="plot-viewport">
      <svg class="history" viewBox="0 0 704 176" role="img" aria-label="Oscillator amplitude history with selected node and shared time cursor">
        <rect x="48" y="30" width="620" height="112" fill="#000004" />
        <g v-for="(row, i) in history" :key="i">
          <rect v-for="(cell, k) in row" :key="k" :x="cell.x" :y="cell.y" :width="width * 32 / trace.raw.length + 0.05" height="7" :fill="cell.color" />
        </g>
        <rect x="48" :y="30 + (NODES - 1 - selected) * 7" width="620" height="7" fill="none" stroke="#e5edf8" stroke-width="0.8" />
        <line :x1="overviewX(time)" y1="25" :x2="overviewX(time)" y2="146" class="cursor" />
        <g v-if="mode === 'burst'">
          <line :x1="overviewX(BURST_END)" y1="27" :x2="overviewX(BURST_END)" y2="144" class="stop-line" />
          <text :x="overviewX(BURST_END) + 5" y="18">input stops</text>
        </g>
        <text x="39" y="37" text-anchor="end">16</text>
        <text x="39" y="142" text-anchor="end">1</text>
        <text x="48" y="166">0 ms</text>
        <text x="668" y="166" text-anchor="end">160 ms</text>
      </svg>
      </div>
    </figure>
    <p class="scope">Synthetic local inference only: fixed frequencies, unit synthesis weights, no graph feedback or weight learning. Playback moves the read cursor through computed samples; it does not invent rotations or change the simulation.</p>
  </section>
</template>

<style scoped>
.cochlea-experiment { margin: 24px 0; padding: 18px; color: #e5edf8; background: #111827; border: 1px solid #35435a; font-family: inherit; }
.cochlea-experiment p { margin: 8px 0 16px; line-height: 1.55; }
.cochlea-experiment header strong { font-size: 1.2em; }
.cochlea-experiment header p, .legend, .scope, .control-unit { color: #a8b7cc; font-size: .86em; }
.experiment-controls, .timeline { display: grid; grid-template-columns: auto minmax(80px, 1fr) 82px; gap: 10px 14px; align-items: center; font-size: .9em; }
.cochlea-experiment input { width: 100%; min-width: 0; accent-color: #dfb768; cursor: pointer; }
.cochlea-experiment output { font-variant-numeric: tabular-nums; }
.experiment-controls output, .timeline output, .control-unit { text-align: right; }
.cochlea-experiment select, .transport button, .tune { padding: 5px 8px; border: 1px solid #586a83; border-radius: 3px; background: #19263a; color: #e5edf8; font: inherit; }
.cochlea-experiment select { min-width: 0; width: 100%; }
.tune { font-size: .85em; }
.explanation { min-height: 4.7em; padding: 10px 12px; border-left: 2px solid #7dafdf; font-size: .9em; }
.transport { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 20px; }
.cochlea-experiment button { cursor: pointer; }
.cochlea-experiment button:hover { border-color: #dfb768; }
.cochlea-experiment button:focus-visible, .cochlea-experiment input:focus-visible, .cochlea-experiment select:focus-visible { outline: 2px solid #dfb768; outline-offset: 3px; }
.cochlea-experiment figure { margin: 22px 0 0; }
.cochlea-experiment figcaption { font-size: .95em; font-weight: 600; }
.cochlea-experiment figcaption > span { display: block; font-size: .86em; font-weight: 400; color: #a8b7cc; }
.plot-viewport { overflow-x: auto; }
.trace, .history { display: block; width: 100%; min-width: 560px; margin-top: 10px; }
.cochlea-experiment svg text { fill: #a8b7cc; font-family: inherit; font-size: 13px; }
.sample-values { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px 14px; font-size: .83em; }
.sample-values output { display: block; font-size: 1.15em; }
.cochlea-experiment .observed { color: #6fbc9e; fill: #6fbc9e; }
.cochlea-experiment .predicted { color: #7dafdf; fill: #7dafdf; }
.cochlea-experiment .residual { color: #e59398; fill: #e59398; }
.corrected { color: #dfb768; }
.raw-line { fill: none; stroke: #6fbc9e; stroke-width: 1.7; }
.prediction-line { fill: none; stroke: #7dafdf; stroke-width: 1.8; }
.error-line { fill: none; stroke: #e59398; stroke-width: 1.7; }
.correction-line { stroke: #dfb768; stroke-width: 2; }
.raw-dot { fill: #6fbc9e; }
.prediction-dot { fill: #7dafdf; }
.error-dot { fill: #e59398; }
.corrected-dot { fill: #dfb768; }
.zero { stroke: #35435a; stroke-width: 1; }
.cursor { stroke: #e5edf8; stroke-width: 1; stroke-dasharray: 3 3; }
.stop-line { stroke: #a8b7cc; stroke-dasharray: 2 3; }
.oscillators { display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 5px; }
.oscillators button { min-width: 0; padding: 4px 2px; background: transparent; color: #a8b7cc; border: 1px solid transparent; font: inherit; font-size: .7em; }
.oscillators button.selected { border-color: #7dafdf; color: #e5edf8; background: #172337; }
.oscillators svg { display: block; width: 100%; }
.orbit { fill: none; stroke: #586a83; stroke-width: .8; }
.axes { fill: none; stroke: #35435a; stroke-width: .7; }
.node-detail { font-size: .88em; font-variant-numeric: tabular-nums; }
.cochlea-experiment .scope { margin-bottom: 0; border-top: 1px solid #35435a; padding-top: 12px; }
.scroll-hint { display: none; }
@media (max-width: 540px) {
  .cochlea-experiment { padding: 12px; }
  .experiment-controls, .timeline { grid-template-columns: minmax(0, 1fr) 76px; gap: 6px 10px; }
  .experiment-controls label, .timeline label { grid-column: 1 / -1; }
  .oscillators { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .oscillators button { font-size: .8em; }
  .explanation { min-height: 0; }
  .cochlea-experiment svg text { font-size: 16px; }
  .scroll-hint { display: block; color: #a8b7cc; font-size: .8em; }
}
</style>
