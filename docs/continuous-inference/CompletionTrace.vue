<script setup>
/* Minimal static chart: six hidden-unit traces per pattern, two configurations
 * side by side. No playback, no 3D — this answers one question: do the six
 * hidden units move differently under the lines patch than the circles patch?
 */
import baseline from "./completion-baseline-trace.json";
import clamped from "./completion-clamped-trace.json";

const configs = [
  { title: "τ = 1/3, no clamp boost (observed pixels barely move)", data: baseline },
  { title: "τ = 1/3, clamp strength 0.75 (observed pixels reconstruct)", data: clamped },
];
const colors = ["#ffae91", "#76d3e9", "#c9a4f2", "#8ddbbb", "#f2d57e", "#d888b0"];
const width = 320, height = 140, padding = 28;

function scaled(config) {
  const values = Object.values(config.data.patterns).flat(2);
  const bound = Math.max(1e-6, ...values.map(Math.abs));
  return bound * 1.1;
}
function path(trace, unit, bound) {
  const frames = trace.length;
  return trace.map((vector, index) => {
    const x = padding + (index / Math.max(1, frames - 1)) * (width - 2 * padding);
    const y = height / 2 - (vector[unit] / bound) * (height / 2 - padding * .6);
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}
</script>

<template>
  <div class="trace-grid">
    <template v-for="config in configs" :key="config.title">
      <p class="trace-config">{{ config.title }}</p>
      <div class="trace-row">
        <figure v-for="pattern in Object.keys(config.data.patterns)" :key="pattern">
          <figcaption>Patch shows: {{ pattern }}</figcaption>
          <svg :viewBox="`0 0 ${width} ${height}`" role="img" :aria-label="`Hidden unit traces, ${pattern} patch, ${config.title}`">
            <line :x1="padding" :y1="height/2" :x2="width-padding" :y2="height/2" stroke="#33404f" stroke-width="1" />
            <path
              v-for="unit in config.data.hidden_nodes" :key="unit"
              :d="path(config.data.patterns[pattern], unit-1, scaled(config))"
              fill="none" :stroke="colors[(unit-1) % colors.length]" stroke-width="1.6" opacity=".9"
            />
          </svg>
        </figure>
      </div>
    </template>
  </div>
  <p class="legend trace-legend">
    Each line is one of the six shared hidden units, recorded every settle frame while only the
    center patch is observed (noisy evidence, fixed weights). Vertical scale is fixed per
    configuration, not per pattern, so panel heights are directly comparable.
  </p>
</template>

<style scoped>
.trace-grid { display: flex; flex-direction: column; gap: 6px; margin: 18px 0 6px; }
.trace-config { font-size: 12px; color: #9fb0c3; margin: 10px 0 2px; }
.trace-row { display: flex; gap: 14px; flex-wrap: wrap; }
.trace-row figure { margin: 0; background: #0d141e; border: 1px solid #263140; border-radius: 10px; padding: 8px 10px; }
.trace-row figcaption { font-size: 11px; color: #8798ab; margin-bottom: 4px; }
.trace-row svg { width: 320px; max-width: 100%; height: auto; display: block; }
.trace-legend { margin-top: 10px; }
</style>
