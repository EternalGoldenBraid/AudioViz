<script setup>
/* Always-visible comparison: the same two curated configurations as the
 * other static figures on this page, side by side, no selector required.
 */
import soft from "./completion-soft-small-summary.json";
import hard from "./completion-hard-small-summary.json";

const configs = [
  { title: "Soft clamp (0) · τ = 1/3", data: soft },
  { title: "Hard clamp (1.0) · τ = 3", data: hard },
];
const colors = ["#ffae91", "#76d3e9", "#c9a4f2", "#8ddbbb", "#f2d57e", "#d888b0"];
const width = 300, height = 130, padding = 26;

function bound(trace) {
  return Math.max(1e-6, ...trace.flat().map(Math.abs)) * 1.1;
}
function path(trace, unit, scale) {
  return trace.map((vector, index) => {
    const x = padding + (index / Math.max(1, trace.length - 1)) * (width - 2 * padding);
    const y = height / 2 - (vector[unit] / scale) * (height / 2 - padding * .6);
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}
</script>

<template>
  <div class="separability">
    <template v-for="config in configs" :key="config.title">
      <p class="config-title">{{ config.title }}</p>
      <div class="chart-row">
        <figure v-for="pattern in Object.keys(config.data.hidden_traces)" :key="pattern">
          <figcaption>Patch shows: {{ pattern }}</figcaption>
          <svg :viewBox="`0 0 ${width} ${height}`" role="img" :aria-label="`Hidden unit traces, ${pattern}, ${config.title}`">
            <line :x1="padding" :y1="height/2" :x2="width-padding" :y2="height/2" stroke="#33404f" stroke-width="1" />
            <path
              v-for="unit in config.data.settings.hidden_nodes" :key="unit"
              :d="path(config.data.hidden_traces[pattern], unit-1, bound(config.data.hidden_traces[pattern]))"
              fill="none" :stroke="colors[(unit-1) % colors.length]" stroke-width="1.6" opacity=".9"
            />
          </svg>
        </figure>
      </div>
    </template>
  </div>
</template>

<style scoped>
.separability { margin: 14px 0; }
.config-title { font-size: 12px; color: #9fb0c3; margin: 10px 0 6px; text-align: center; }
.chart-row { display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; }
.chart-row figure { margin: 0; background: #0d141e; border: 1px solid #263140; border-radius: 10px; padding: 8px 10px; }
.chart-row figcaption { font-size: 11px; color: #8798ab; margin-bottom: 4px; }
.chart-row svg { width: 300px; max-width: 100%; height: auto; display: block; }
</style>
