<script setup>
import { computed, ref } from "vue";
import SceneFigure from "./SceneFigure.vue";
import { COLORS } from "./figures.mjs";
import { readerView } from "./scene.mjs";
import { createProgression, FAMILIES, PROGRESSION_BOUNDS, IMAGE_SCALE } from "./progression.mjs";

const props = defineProps({ reader: { type: Object, required: true } });
const emit = defineEmits(["orbit", "reset-view"]);
const complexity = ref(0);
const sliderId = "transform-complexity";
const experiment = computed(() => createProgression(complexity.value));
const observer = computed(() => readerView({
  azimuth: props.reader.azimuth, elevation: props.reader.elevation, bounds: PROGRESSION_BOUNDS,
}));
const plot = ([u,v], panel) => [(panel ? 600 : 200)+IMAGE_SCALE*(u-0.3), 140-IMAGE_SCALE*v];
const segment = (edge, field, panel) => edge.map(i => plot(experiment.value.samples[i][field],panel));
const errors = computed(() => [
  ["Similarity fit", experiment.value.errors.similarity],
  ["Affine fit", experiment.value.errors.affine],
  ["Reference-plane H", experiment.value.errors.homography],
]);
const errorText = value => value < 1e-9 ? "exact" : value < 0.0001 ? value.toExponential(1) : value.toFixed(4);
const captions = [
  "Uniform zoom and camera roll change size and orientation. A similarity still explains every match; angles and relative lengths are preserved.",
  "The reference plane tilts while the cameras keep parallel optical axes. A similarity no longer fits; one affine map still explains every match. Parallel lines remain parallel.",
  "Camera 2 turns toward the scene. The mapping now needs a position-dependent perspective divide: one homography still explains every match on the plane.",
  "Interior samples move off the plane along camera-1 rays. The four plane corners fix H, but the raised points miss its predictions. No single homography explains the whole grid.",
];
</script>

<template>
  <section class="transform-progression" aria-label="Transform progression" :data-transform-family="experiment.family">
    <div class="progression-controls">
      <label :for="sliderId">Geometric complexity</label>
      <output :for="sliderId" aria-live="polite">{{ FAMILIES[experiment.family] }}</output>
      <input :id="sliderId" type="range" min="0" max="4" step="0.01" v-model.number="complexity" />
      <div class="stage-buttons">
        <button v-for="(family, i) in FAMILIES" :key="family" type="button"
          :data-transform-stage="i+1" :aria-pressed="experiment.family === i" @click="complexity = i+1">{{ family }}</button>
      </div>
      <p class="mechanism">
        Focal ratio {{ experiment.zoom.toFixed(2) }} · roll {{ experiment.roll.toFixed(1) }}°
        · plane tilt {{ experiment.planeTilt.toFixed(1) }}° · camera turn {{ experiment.yaw.toFixed(1) }}°
        · maximum off-plane distance {{ experiment.height.toFixed(2) }}
      </p>
    </div>
    <SceneFigure class="progression-world" mode="world" scene-id="progression-world" :world="experiment.world" :reader="observer"
      @orbit="emit('orbit',$event)" @reset-view="emit('reset-view')" />
    <div class="image-views">
      <figure v-for="panel in [0,1]" :key="panel" class="image-view">
        <figcaption>{{ panel ? "Image 2 · observed / H" : "Image 1 · fixed rays" }}</figcaption>
        <svg :viewBox="`${panel ? 410 : 10} 30 380 240`" role="img"
          :aria-label="panel ? 'Image 2: observed and plane-predicted sample grid' : 'Image 1: projected sample grid'">
          <defs v-if="panel">
            <marker id="progression-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
              <path d="M 0 0 L 8 4 L 0 8 z" :fill="COLORS.observed" />
            </marker>
          </defs>
          <rect width="800" height="290" :fill="COLORS.bg" />
          <rect :x="panel ? 420 : 20" y="40" width="360" height="220" fill="none" :stroke="COLORS.dim" stroke-width="0.7" />
          <g v-for="(edge, i) in experiment.edges" :key="`edge-${i}`">
            <polyline v-if="!panel" :points="segment(edge,'source',0).map(p => p.join(',')).join(' ')" fill="none"
              :stroke="COLORS.c1" stroke-width="1.1" />
            <template v-else>
              <polyline :points="segment(edge,'predicted',1).map(p => p.join(',')).join(' ')" fill="none"
                :stroke="COLORS.prediction" stroke-width="1.1" stroke-dasharray="4 3" />
              <polyline :points="segment(edge,'target',1).map(p => p.join(',')).join(' ')" fill="none"
                :stroke="COLORS.observed" stroke-width="1" />
            </template>
          </g>
          <g v-for="sample in experiment.samples" :key="sample.id">
            <circle v-if="!panel" :cx="plot(sample.source,0)[0]" :cy="plot(sample.source,0)[1]" r="2.5"
              :fill="sample.offset > 1e-9 ? COLORS.observed : COLORS.c1" :data-transform-source="sample.id" />
            <template v-else>
              <line v-if="sample.offset > 1e-9"
                :x1="plot(sample.predicted,1)[0]" :y1="plot(sample.predicted,1)[1]"
                :x2="plot(sample.target,1)[0]" :y2="plot(sample.target,1)[1]"
                :stroke="COLORS.observed" stroke-width="1.3" marker-end="url(#progression-arrow)" />
              <circle :cx="plot(sample.predicted,1)[0]" :cy="plot(sample.predicted,1)[1]" r="3.3"
                :fill="COLORS.bg" :stroke="COLORS.prediction" stroke-width="1.2" :data-transform-predicted="sample.id" />
              <circle :cx="plot(sample.target,1)[0]" :cy="plot(sample.target,1)[1]" r="1.9"
                :fill="COLORS.observed" :data-transform-target="sample.id" />
            </template>
          </g>
        </svg>
      </figure>
    </div>
    <p class="image-legend">Same image-coordinate scale · blue: reference-plane H · gold: observed matches</p>
    <div class="model-errors" aria-label="RMS correspondence errors">
      <div v-for="[name, error] in errors" :key="name" :class="{ exact: error < 1e-9 }">
        <span>{{ name }}</span><output>{{ errorText(error) }}</output>
      </div>
    </div>
    <p class="explanation" aria-live="polite">{{ captions[experiment.family] }}</p>
    <details class="construction-notes">
      <summary>Scene details and error units</summary>
      <p>Blue 3D grid: on Π; gold grid: off-plane. X is the centre sample; Xπ is its camera-1 ray's intersection with Π.
        Drag to rotate the reader without changing the observations.</p>
      <p>RMS errors use image-coordinate units. Similarity and affine are least-squares fits to all matches; H is fixed by the reference plane.</p>
    </details>
  </section>
</template>

<style scoped>
.transform-progression {
  margin: 20px 0;
  background: #1B1F27;
  color: #E6EDF3;
  font-family: "Latin Modern Roman", "CMU Serif", "Computer Modern", serif;
}
.progression-controls {
  position: sticky; top: var(--vp-nav-height,64px); z-index: 2;
  background: #1B1F27; padding: 10px 16px 6px;
  display: grid; grid-template-columns: 1fr auto; gap: 4px;
  border-bottom: 1px solid #525D69;
}
.progression-controls > output { color: #F0C060; }
input, .stage-buttons, .mechanism { grid-column: 1 / -1; }
input { width: 100%; margin: 0; accent-color: #F0C060; }
.stage-buttons { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 8px; }
button { font: inherit; font-size: 14px; line-height: 1.4; background: transparent; color: #9AA5B1; border: 1px solid #9AA5B1; padding: 4px; cursor: pointer; }
button[aria-pressed="true"] { border-color: #F0C060; color: #F0C060; }
.mechanism { font-size: 12px; color: #9AA5B1; line-height: 1.5; margin: 0; }
.progression-world { margin: 0; }
.progression-world :deep(.scene-orbit-controls) { padding: 4px 16px; }
.progression-world :deep(.scene-orbit-controls button) { line-height: 1.4; }
.progression-world :deep(svg) { min-width: 480px; max-height: 240px; font-size: 20px; }
.image-views { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; padding: 0 16px; }
.image-view { margin: 0; min-width: 0; }
.image-view figcaption { text-align: center; font-size: 14px; line-height: 1.3; padding-bottom: 4px; }
svg { display: block; width: 100%; }
.image-legend { font-size: 13px; color: #9AA5B1; line-height: 1.5; text-align: center; margin: 0; padding: 4px 16px 12px; }
.explanation { margin: 0; padding: 8px 16px 16px; line-height: 1.5; }
.model-errors { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px; padding: 0 16px; }
.model-errors > div { border-top: 1px solid #9AA5B1; padding-top: 6px; display: flex; flex-direction: column; gap: 2px; font-size: 14px; }
.model-errors output { color: #F0C060; font-variant-numeric: tabular-nums; }
.model-errors .exact output { color: #7BC47F; }
.construction-notes { padding: 0 16px 12px; color: #9AA5B1; font-size: 14px; line-height: 1.5; }
.construction-notes summary { cursor: pointer; color: #6EA8FE; }
@media (max-width: 1279px) {
  .progression-controls { top: calc(var(--vp-nav-height,64px) + 48px); }
}
@media (max-width: 560px) {
  .stage-buttons { grid-template-columns: repeat(2,minmax(0,1fr)); }
  .progression-controls { grid-template-columns: 1fr; }
  .progression-controls > output { grid-column: 1; }
}
</style>
