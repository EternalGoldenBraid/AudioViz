<script setup>
/* One selector drives three views of the SAME completion attempt: the
 * recorded prediction playback (two patterns, side by side), the static
 * scene targets for comparison, and the six hidden-unit traces. No 3D, no
 * camera — this is a small, scrubbable real-time recording, not a live model.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import gridIndex from "./completion-grid.json";
import linesRef from "./pattern-lines.png?url";
import circlesRef from "./pattern-circles.png?url";

const summaries = import.meta.glob("./completion-*-summary.json", { eager: true, import: "default" });
const playbackUrls = import.meta.glob("./completion-*-playback.png", { eager: true, query: "?url", import: "default" });
const references = { lines: linesRef, circles: circlesRef };

const labels = { soft: "Soft clamp (0)", mixed: "Mixed clamp (.75)", hard: "Hard clamp (1.0)" };
const tauLabels = { small: "τ = 1/3", big: "τ = 3" };
const options = gridIndex.configs.map(config => ({
  value: config.label,
  title: `${labels[config.clamp_name]} · ${tauLabels[config.tau_name]}`,
}));
const selected = ref(options[0]?.value ?? "");

const summary = computed(() => summaries[`./${selected.value}-summary.json`] ?? null);
const playbackUrl = computed(() => playbackUrls[`./${selected.value}-playback.png`] ?? null);
const frames = computed(() => summary.value?.playback_atlas.columns ?? 1);
const patterns = computed(() => summary.value?.playback_atlas.patterns ?? []);
const shape = computed(() => summary.value?.settings.shape ?? [1, 1]);
const missing = computed(() => !summary.value || !playbackUrl.value);

const frame = ref(0);
const playing = ref(true);
const canvases = {};
const image = new Image();
let imageReady = false;
let animation = null;
let previousStamp = null;
const FRAME_SECONDS = .18;

function loadPlayback() {
  imageReady = false;
  frame.value = 0;
  if (!playbackUrl.value) return;
  image.onload = () => { imageReady = true; draw(); };
  image.onerror = () => { imageReady = false; };
  image.src = playbackUrl.value;
}

function draw() {
  if (!imageReady || !summary.value) return;
  const [height, width] = shape.value;
  for (const name of patterns.value) {
    const canvas = canvases[name];
    if (!canvas) continue;
    const row = patterns.value.indexOf(name);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(
      image, frame.value * width, row * height, width, height, 0, 0, width, height,
    );
  }
}

function tick(stamp) {
  if (previousStamp !== null && playing.value) {
    const elapsed = (stamp - previousStamp) / 1000;
    if (elapsed >= FRAME_SECONDS) {
      frame.value = (frame.value + 1) % frames.value;
      previousStamp = stamp;
      draw();
    }
  } else if (previousStamp === null) {
    previousStamp = stamp;
  }
  animation = requestAnimationFrame(tick);
}

function seek(value) {
  playing.value = false;
  frame.value = Math.min(frames.value - 1, Math.max(0, Number(value)));
  draw();
}

watch(selected, loadPlayback);
watch(frame, draw);

onMounted(() => {
  loadPlayback();
  animation = requestAnimationFrame(tick);
});
onBeforeUnmount(() => {
  if (animation !== null) cancelAnimationFrame(animation);
  image.onload = null;
  image.onerror = null;
});

const colors = ["#ffae91", "#76d3e9", "#c9a4f2", "#8ddbbb", "#f2d57e", "#d888b0"];
const chartWidth = 300, chartHeight = 130, chartPadding = 26;

function traceBound(trace) {
  return Math.max(1e-6, ...trace.flat().map(Math.abs)) * 1.1;
}
function tracePath(trace, unit, bound) {
  return trace.map((vector, index) => {
    const x = chartPadding + (index / Math.max(1, trace.length - 1)) * (chartWidth - 2 * chartPadding);
    const y = chartHeight / 2 - (vector[unit] / bound) * (chartHeight / 2 - chartPadding * .6);
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}
function markerX(index, length) {
  return chartPadding + (index / Math.max(1, length - 1)) * (chartWidth - 2 * chartPadding);
}
</script>

<template>
  <div class="explorer">
    <label class="config-select">Configuration
      <select v-model="selected">
        <option v-for="option in options" :key="option.value" :value="option.value">{{ option.title }}</option>
      </select>
    </label>
    <p v-if="missing" role="alert">This configuration has no recorded playback. Regenerate the grid below.</p>
    <template v-else>
      <div class="transport">
        <button @click="playing = !playing">{{ playing ? "Pause" : "Play" }}</button>
        <input class="scrubber" type="range" min="0" :max="frames-1" :value="frame" @input="seek($event.target.value)">
        <output>frame {{ frame+1 }} / {{ frames }}</output>
      </div>
      <div class="playback-row">
        <figure v-for="name in patterns" :key="name">
          <figcaption>Patch shows: {{ name }}</figcaption>
          <div class="side-by-side">
            <img :src="references[name]" :alt="`${name} target`" class="reference">
            <canvas :ref="element => canvases[name] = element" :width="shape[1]" :height="shape[0]" :aria-label="`${name} recorded prediction playback`" role="img"></canvas>
          </div>
        </figure>
      </div>
      <p class="legend">Left: fixed scene target. Right: recorded prediction, replayed frame by frame exactly as
        captured — this is the real completion attempt, not a live model.</p>
      <div class="trace-row">
        <figure v-for="name in patterns" :key="name">
          <figcaption>Hidden units · {{ name }}</figcaption>
          <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" role="img" :aria-label="`Hidden unit traces for ${name}`">
            <line :x1="chartPadding" :y1="chartHeight/2" :x2="chartWidth-chartPadding" :y2="chartHeight/2" stroke="#33404f" stroke-width="1" />
            <line :x1="markerX(frame, summary.hidden_traces[name].length)" y1="4" :x2="markerX(frame, summary.hidden_traces[name].length)" :y2="chartHeight-4" stroke="#f3e0bb" stroke-width="1" opacity=".6" />
            <path
              v-for="unit in summary.settings.hidden_nodes" :key="unit"
              :d="tracePath(summary.hidden_traces[name], unit-1, traceBound(summary.hidden_traces[name]))"
              fill="none" :stroke="colors[(unit-1) % colors.length]" stroke-width="1.6" opacity=".9"
            />
          </svg>
        </figure>
      </div>
    </template>
  </div>
</template>

<style scoped>
.explorer { margin: 14px 0; }
.config-select { display: flex; align-items: center; gap: 8px; font-size: 13px; margin-bottom: 10px; justify-content: center; }
.config-select select { border: 1px solid #3b4756; border-radius: 5px; background: #19212b; color: #e6edf4; padding: 5px 8px; }
.transport { display: flex; align-items: center; gap: 10px; font-size: 12px; margin-bottom: 12px; max-width: 640px; margin-left: auto; margin-right: auto; }
.transport button { border: 1px solid #3b4756; border-radius: 5px; background: #19212b; color: #e6edf4; padding: 5px 10px; }
.scrubber { flex: 1; min-width: 120px; accent-color: #82c7d3; }
.playback-row, .trace-row { display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; }
.playback-row figure, .trace-row figure { margin: 0; background: #0d141e; border: 1px solid #263140; border-radius: 10px; padding: 8px 10px; }
.playback-row figcaption, .trace-row figcaption { font-size: 11px; color: #8798ab; margin-bottom: 6px; }
.side-by-side { display: flex; gap: 8px; align-items: center; }
.reference, canvas { width: 220px; height: 220px; image-rendering: pixelated; border: 1px solid #263140; border-radius: 4px; }
.trace-row svg { width: 300px; max-width: 100%; height: auto; display: block; }
.legend { font-size: 12px; color: #98a8ba; margin: 4px 0 16px; }
</style>
