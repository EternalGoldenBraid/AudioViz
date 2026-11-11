<script setup>
/* Always-visible comparison: the same two curated configurations (soft vs
 * hard clamp, same τ) side by side, sharing one play/pause/scrubber. No
 * selector is needed to see the comparison — only to watch it move.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import softSummary from "./completion-hard-small-summary.json";
import hardSummary from "./completion-hard-big-summary.json";
import softPlaybackUrl from "./completion-hard-small-playback.png?url";
import hardPlaybackUrl from "./completion-hard-big-playback.png?url";
import linesRef from "./pattern-lines.png?url";
import circlesRef from "./pattern-circles.png?url";

const configs = [
  { key: "soft", title: "Hard clamp (1.0) · τ = 1/3", summary: softSummary, url: softPlaybackUrl },
  { key: "hard", title: "Hard clamp (1.0) · τ = 3", summary: hardSummary, url: hardPlaybackUrl },
];
const references = { lines: linesRef, circles: circlesRef };
const patterns = softSummary.playback_atlas.patterns;
const shape = softSummary.settings.shape;
const foveaRadius = softSummary.settings.foveal_radius;
const frames = softSummary.playback_atlas.columns;

const frame = ref(0);
const playing = ref(true);
const canvases = {};
const images = {};
let readyCount = 0;
let animation = null;
let previousStamp = null;
const FRAME_SECONDS = .06;

const currentFovea = computed(() => softSummary.fovea_trace[frame.value]);

function drawOne(config, name) {
  const canvas = canvases[`${config.key}-${name}`];
  const image = images[config.key];
  if (!canvas || !image?.complete || image.naturalWidth === 0) return;
  const [height, width] = shape;
  const row = patterns.indexOf(name);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(image, frame.value * width, row * height, width, height, 0, 0, width, height);
  const fovea = currentFovea.value;
  if (fovea) {
    ctx.save();
    ctx.strokeStyle = "#f3e0bb";
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ctx.arc(fovea.center[1] + .5, fovea.center[0] + .5, fovea.radius + .5, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }
}
function draw() {
  for (const config of configs) for (const name of patterns) drawOne(config, name);
}
function tick(stamp) {
  if (previousStamp !== null && playing.value) {
    const elapsed = (stamp - previousStamp) / 1000;
    if (elapsed >= FRAME_SECONDS) {
      frame.value = (frame.value + 1) % frames;
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
  frame.value = Math.min(frames - 1, Math.max(0, Number(value)));
  draw();
}
watch(frame, draw);

onMounted(() => {
  for (const config of configs) {
    const image = new Image();
    image.onload = () => { readyCount += 1; draw(); };
    image.src = config.url;
    images[config.key] = image;
  }
  animation = requestAnimationFrame(tick);
});
onBeforeUnmount(() => {
  if (animation !== null) cancelAnimationFrame(animation);
  for (const image of Object.values(images)) { image.onload = null; }
});

const colors = ["#ffae91", "#76d3e9", "#c9a4f2", "#8ddbbb", "#f2d57e", "#d888b0"];
const chartWidth = 300, chartHeight = 130, chartPadding = 26;

function pathFromValues(values, bound) {
  return values.map((value, position) => {
    const x = chartPadding + (position / Math.max(1, values.length - 1)) * (chartWidth - 2 * chartPadding);
    const y = chartHeight - chartPadding * .6 - (value / bound) * (chartHeight - chartPadding * 1.2);
    return `${position === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}
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
function markerX() {
  return chartPadding + (frame.value / Math.max(1, frames - 1)) * (chartWidth - 2 * chartPadding);
}
function errorBound(name) {
  return Math.max(
    ...configs.flatMap(config => [...config.summary.error_traces[name].inside, ...config.summary.error_traces[name].outside]),
    1e-6,
  ) * 1.1;
}
</script>

<template>
  <div class="highlights">
    <div class="transport">
      <button @click="playing = !playing">{{ playing ? "Pause" : "Play" }}</button>
      <input class="scrubber" type="range" min="0" :max="frames-1" :value="frame" @input="seek($event.target.value)">
      <output>frame {{ frame+1 }} / {{ frames }}</output>
    </div>

    <div class="config-group" v-for="config in configs" :key="config.key">
      <p class="config-title">{{ config.title }}</p>
      <div class="playback-row">
        <figure v-for="name in patterns" :key="name">
          <figcaption>{{ name }}</figcaption>
          <div class="side-by-side">
            <div class="stage">
              <img :src="references[name]" :alt="`${name} target`" class="reference">
              <svg class="ring" :viewBox="`0 0 ${shape[1]} ${shape[0]}`" preserveAspectRatio="none">
                <circle :cx="currentFovea.center[1]+.5" :cy="currentFovea.center[0]+.5" :r="foveaRadius+.5" fill="none" stroke="#f3e0bb" stroke-width=".6" />
              </svg>
            </div>
            <canvas :ref="element => canvases[`${config.key}-${name}`] = element" :width="shape[1]" :height="shape[0]" :aria-label="`${name}, ${config.title}, recorded prediction`" role="img"></canvas>
          </div>
        </figure>
      </div>
    </div>
    <p class="legend">Left: fixed scene target with the current fovea boundary. Right: recorded prediction for each
      configuration, replayed frame by frame exactly as captured, learning left on throughout.</p>

    <p class="section-title">Error inside vs outside the fovea</p>
    <div class="trace-row">
      <figure v-for="name in patterns" :key="`error-${name}`">
        <figcaption>{{ name }}</figcaption>
        <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" role="img" :aria-label="`Inside/outside error over time, ${name}`">
          <line :x1="chartPadding" :y1="chartHeight-chartPadding*.6" :x2="chartWidth-chartPadding" :y2="chartHeight-chartPadding*.6" stroke="#33404f" stroke-width="1" />
          <line :x1="markerX()" y1="4" :x2="markerX()" :y2="chartHeight-4" stroke="#f3e0bb" stroke-width="1" opacity=".6" />
          <template v-for="config in configs" :key="config.key">
            <path :d="pathFromValues(config.summary.error_traces[name].inside, errorBound(name))" fill="none" :stroke="config.key === 'soft' ? '#76d3e9' : '#ffae91'" stroke-width="1.6" />
            <path :d="pathFromValues(config.summary.error_traces[name].outside, errorBound(name))" fill="none" :stroke="config.key === 'soft' ? '#76d3e9' : '#ffae91'" stroke-width="1.6" stroke-dasharray="3,2" opacity=".7" />
          </template>
        </svg>
      </figure>
    </div>
    <p class="legend"><span class="soft-key">Blue: τ = 1/3</span> · <span class="hard-key">Orange: τ = 3</span> · Solid: inside the fovea · Dashed: outside</p>

    <p class="section-title">Hidden units</p>
    <div class="config-group" v-for="config in configs" :key="`trace-${config.key}`">
      <p class="config-title">{{ config.title }}</p>
      <div class="trace-row">
        <figure v-for="name in patterns" :key="name">
          <figcaption>{{ name }}</figcaption>
          <svg :viewBox="`0 0 ${chartWidth} ${chartHeight}`" role="img" :aria-label="`Hidden unit traces, ${name}, ${config.title}`">
            <line :x1="chartPadding" :y1="chartHeight/2" :x2="chartWidth-chartPadding" :y2="chartHeight/2" stroke="#33404f" stroke-width="1" />
            <line :x1="markerX()" y1="4" :x2="markerX()" :y2="chartHeight-4" stroke="#f3e0bb" stroke-width="1" opacity=".6" />
            <path
              v-for="unit in config.summary.settings.hidden_nodes" :key="unit"
              :d="tracePath(config.summary.hidden_traces[name], unit-1, traceBound(config.summary.hidden_traces[name]))"
              fill="none" :stroke="colors[(unit-1) % colors.length]" stroke-width="1.6" opacity=".9"
            />
          </svg>
        </figure>
      </div>
    </div>
  </div>
</template>

<style scoped>
.highlights { margin: 14px 0; }
.transport { display: flex; align-items: center; gap: 10px; font-size: 12px; margin-bottom: 14px; max-width: 640px; margin-left: auto; margin-right: auto; }
.transport button { border: 1px solid #3b4756; border-radius: 5px; background: #19212b; color: #e6edf4; padding: 5px 10px; }
.scrubber { flex: 1; min-width: 120px; accent-color: #82c7d3; }
.config-title { font-size: 12px; color: #9fb0c3; margin: 10px 0 6px; text-align: center; }
.section-title { font-size: 13px; color: #cfd9e5; margin: 22px 0 4px; text-align: center; font-weight: 600; }
.playback-row, .trace-row { display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; }
.playback-row figure, .trace-row figure { margin: 0; background: #0d141e; border: 1px solid #263140; border-radius: 10px; padding: 8px 10px; }
.playback-row figcaption, .trace-row figcaption { font-size: 11px; color: #8798ab; margin-bottom: 6px; }
.side-by-side { display: flex; gap: 8px; align-items: center; }
.stage { position: relative; width: 150px; height: 150px; }
.reference, canvas { width: 150px; height: 150px; image-rendering: pixelated; border: 1px solid #263140; border-radius: 4px; display: block; }
.stage .reference { position: absolute; inset: 0; }
.ring { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.trace-row svg { width: 300px; max-width: 100%; height: auto; display: block; }
.legend { font-size: 12px; color: #98a8ba; margin: 4px 0 16px; text-align: center; }
.soft-key { color: #76d3e9; } .hard-key { color: #ffae91; }
</style>
