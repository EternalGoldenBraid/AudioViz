<script setup>
import { computed, ref } from "vue";
import SceneFigure from "./SceneFigure.vue";
import { COLORS } from "./figures.mjs";
import { readerView } from "./scene.mjs";
import { createProgression, FAMILIES, PROGRESSION_BOUNDS, IMAGE_SCALE } from "./progression.mjs";

const props = defineProps({
  reader: { type: Object, required: true },
  view: { type: String, default: "2d", validator: value => ["2d","3d"].includes(value) },
});
const emit = defineEmits(["orbit", "reset-view"]);
const complexity = ref(0);
const sliderId = computed(() => `transform-complexity-${props.view}`);
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
  <section class="transform-progression" :aria-label="`${view.toUpperCase()} transform progression`" :data-transform-family="experiment.family">
    <div class="progression-controls">
      <label :for="sliderId">{{ view.toUpperCase() }} geometric complexity</label>
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
    <template v-if="view === '2d'">
    <div class="image-viewport">
      <svg viewBox="0 0 800 290" role="img" aria-label="Image correspondences from the transform progression">
        <defs>
          <marker id="progression-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 8 4 L 0 8 z" :fill="COLORS.observed" />
          </marker>
        </defs>
        <rect width="800" height="290" :fill="COLORS.bg" />
        <g v-for="panel in [0,1]" :key="panel">
          <rect :x="panel ? 420 : 20" y="40" width="360" height="220" fill="none" :stroke="COLORS.dim" stroke-width="0.7" />
          <text :x="panel ? 600 : 200" y="23" text-anchor="middle" :fill="COLORS.fg">{{ panel ? "Image 2 · actual projection" : "Image 1 · fixed sample rays" }}</text>
        </g>
        <g v-for="(edge, i) in experiment.edges" :key="`edge-${i}`">
          <polyline :points="segment(edge,'source',0).map(p => p.join(',')).join(' ')" fill="none" :stroke="COLORS.c1" stroke-width="1.1" />
          <polyline :points="segment(edge,'predicted',1).map(p => p.join(',')).join(' ')" fill="none"
            :stroke="COLORS.prediction" stroke-width="1.1" stroke-dasharray="4 3" />
          <polyline :points="segment(edge,'target',1).map(p => p.join(',')).join(' ')" fill="none" :stroke="COLORS.observed" stroke-width="1" />
        </g>
        <g v-for="sample in experiment.samples" :key="sample.id">
          <circle :cx="plot(sample.source,0)[0]" :cy="plot(sample.source,0)[1]" r="2.5"
            :fill="sample.offset > 1e-9 ? COLORS.observed : COLORS.c1" :data-transform-source="sample.id" />
          <line v-if="sample.offset > 1e-9"
            :x1="plot(sample.predicted,1)[0]" :y1="plot(sample.predicted,1)[1]"
            :x2="plot(sample.target,1)[0]" :y2="plot(sample.target,1)[1]"
            :stroke="COLORS.observed" stroke-width="1.3" marker-end="url(#progression-arrow)" />
          <circle :cx="plot(sample.predicted,1)[0]" :cy="plot(sample.predicted,1)[1]" r="3.3"
            :fill="COLORS.bg" :stroke="COLORS.prediction" stroke-width="1.2" :data-transform-predicted="sample.id" />
          <circle :cx="plot(sample.target,1)[0]" :cy="plot(sample.target,1)[1]" r="1.9"
            :fill="COLORS.observed" :data-transform-target="sample.id" />
        </g>
        <text x="400" y="282" text-anchor="middle" :fill="COLORS.dim">Same coordinate scale · blue: reference-plane H · gold: observed matches</text>
      </svg>
    </div>
    <p class="explanation" aria-live="polite">{{ captions[experiment.family] }}</p>
    <div class="model-errors" aria-label="RMS correspondence errors">
      <div v-for="[name, error] in errors" :key="name" :class="{ exact: error < 1e-9 }">
        <span>{{ name }}</span><output>{{ errorText(error) }}</output>
      </div>
    </div>
    <p class="error-note">RMS errors use image-coordinate units. Similarity and affine are least-squares fits to all matches; H is fixed by the reference plane.</p>
    </template>
    <template v-else>
      <SceneFigure mode="world" scene-id="progression-world" :world="experiment.world" :reader="observer"
        @orbit="emit('orbit',$event)" @reset-view="emit('reset-view')" />
      <p class="explanation">Blue grid: on Π; gold grid: off-plane. X is the centre sample; Xπ is its camera-1 ray's intersection with Π.
        Drag to rotate the reader without changing the observations.</p>
    </template>
  </section>
</template>

<style scoped>
.transform-progression {
  margin: 20px 0;
  background: #1B1F27;
  color: #E6EDF3;
  font-family: "Latin Modern Roman", "CMU Serif", "Computer Modern", serif;
}
.progression-controls { padding: 16px 16px 0; display: grid; grid-template-columns: 1fr auto; gap: 12px; }
.progression-controls > output { color: #F0C060; }
input, .stage-buttons, .mechanism { grid-column: 1 / -1; }
input { width: 100%; margin: 0; accent-color: #F0C060; }
.stage-buttons { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 8px; }
button { font: inherit; background: transparent; color: #9AA5B1; border: 1px solid #9AA5B1; padding: 6px; cursor: pointer; }
button[aria-pressed="true"] { border-color: #F0C060; color: #F0C060; }
.mechanism, .error-note { font-size: 14px; color: #9AA5B1; line-height: 1.5; margin: 0; }
.image-viewport { overflow-x: auto; margin-top: 12px; }
svg { display: block; width: 100%; min-width: 620px; font-size: 16px; font-family: inherit; }
.explanation { margin: 0; padding: 8px 16px 16px; line-height: 1.5; }
.model-errors { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 12px; padding: 0 16px; }
.model-errors > div { border-top: 1px solid #9AA5B1; padding-top: 8px; display: flex; flex-direction: column; gap: 4px; font-size: 15px; }
.model-errors output { color: #F0C060; font-variant-numeric: tabular-nums; }
.model-errors .exact output { color: #7BC47F; }
.error-note { padding: 12px 16px; }
@media (max-width: 560px) {
  .stage-buttons { grid-template-columns: repeat(2,minmax(0,1fr)); }
  .progression-controls { grid-template-columns: 1fr; }
  .progression-controls > output { grid-column: 1; }
}
</style>
