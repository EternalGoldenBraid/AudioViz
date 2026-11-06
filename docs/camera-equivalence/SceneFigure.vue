<script setup>
import { computed, ref } from "vue";
import { buildFigure, COLORS } from "./figures.mjs";
import { READER_LIMITS } from "./scene.mjs";

const props = defineProps({
  mode: { type: String, required: true },
  world: { type: Object, required: true },
  reader: { type: Object, required: true },
});
const figure = computed(() => buildFigure(props.mode, props.world, props.reader));
const emit = defineEmits(["orbit", "reset-view"]);
const spatial = computed(() => !["flow", "predictions"].includes(props.mode));
const dragging = ref(false);
let gesture = null;
let moved = false;

function startDrag(event) {
  if (!spatial.value || !event.isPrimary || event.button !== 0 || event.pointerType === "touch") return;
  moved = false;
  gesture = {
    id: event.pointerId, x: event.clientX, y: event.clientY,
    azimuth: props.reader.azimuth, elevation: props.reader.elevation,
  };
}
function drag(event) {
  if (!gesture || event.pointerId !== gesture.id) return;
  const dx = event.clientX-gesture.x, dy = event.clientY-gesture.y;
  if (!moved && Math.hypot(dx,dy) < 4) return;
  if (!moved) event.currentTarget.setPointerCapture(event.pointerId);
  moved = dragging.value = true;
  event.preventDefault();
  const azimuth = ((Math.round(gesture.azimuth+dx*0.4)+180)%360+360)%360-180;
  const elevation = Math.round(Math.max(-READER_LIMITS.elevation,Math.min(READER_LIMITS.elevation,gesture.elevation-dy*0.4)));
  emit("orbit", { azimuth, elevation });
}
function endDrag(event) {
  if (gesture && event && event.pointerId !== gesture.id) return;
  gesture = null;
  dragging.value = false;
}
function cancelDrag() {
  endDrag();
  moved = false;
}
function finishMouse(event) {
  // A drag release is not a click-to-comment gesture.
  if (moved && event.button === 0) {
    event.stopPropagation();
    moved = false;
  }
}
const residual = computed(() => Math.hypot(
  ...props.world.predicted.uv.map((x, i) => x - props.world.observations[1].uv[i]),
).toFixed(3));
</script>

<template>
  <figure class="scene-figure" :data-scene-mode="mode">
    <div v-if="spatial" class="scene-orbit-controls">
      <span>Mouse drag to rotate</span>
      <button type="button" @click="emit('reset-view')">Reset view</button>
    </div>
    <div class="scene-viewport">
      <svg :viewBox="`0 0 ${figure.width} ${figure.height}`" role="img" :aria-label="figure.title"
        :class="{ orbitable: spatial, orbiting: dragging }"
        @pointerdown="startDrag" @pointermove="drag" @pointerup="endDrag"
        @pointercancel="cancelDrag" @lostpointercapture="endDrag"
        @pointerleave="!dragging && endDrag()" @mouseup="finishMouse">
        <rect width="100%" height="100%" :fill="COLORS.bg" />
        <g v-for="(shape, i) in figure.shapes" :key="i">
          <polygon v-if="shape.type === 'polygon'" :points="shape.points.map(p => p.join(',')).join(' ')"
            :fill="shape.fill" :stroke="shape.color" :stroke-width="shape.width" />
          <line v-else-if="shape.type === 'line'" :x1="shape.points[0][0]" :y1="shape.points[0][1]"
            :x2="shape.points[1][0]" :y2="shape.points[1][1]" :stroke="shape.color"
            :stroke-width="shape.width" :stroke-dasharray="shape.dashed ? '5 5' : undefined" />
          <circle v-else-if="shape.type === 'dot'" :cx="shape.p[0]" :cy="shape.p[1]"
            :r="shape.radius" :fill="shape.color" :data-point="shape.id" />
          <text v-else :x="shape.p[0]" :y="shape.p[1]" :fill="shape.color" :text-anchor="shape.anchor"
            :font-style="shape.tex.startsWith('$') && shape.tex.endsWith('$') && shape.text !== 'Π' ? 'italic' : undefined"
            dominant-baseline="central" :stroke="COLORS.bg" stroke-width="3" paint-order="stroke">{{ shape.text }}</text>
        </g>
      </svg>
    </div>
    <figcaption v-if="mode === 'parallax'" aria-live="polite">
      <template v-if="!world.baseline">The centres coincide; the orientations still differ. H accounts for the rotation. There is no translation, parallax or translation-defined epipole.</template>
      <template v-else-if="!world.height">X lies on Π, so X = Xπ and Hx₁ = x₂, even with separated cameras.</template>
      <template v-else>X and Xπ share C₁'s ray. C₂ separates their projections: blue Hx₁ and gold x₂. Residual: {{ residual }} image-coordinate units.</template>
    </figcaption>
    <figcaption v-else-if="mode === 'epipolar' && !world.baseline" aria-live="polite">With coincident centres, E = F = 0. There is no translation-defined epipolar plane or line constraint.</figcaption>
    <figcaption v-else-if="mode === 'flow'">Arrows join x₁ to x₂ in shared image-coordinate units. Samples ray-cast the plane and a sphere centred at X; the sphere then translates by (0.6, 0.35, 0). Only matches visible in both cameras are shown.</figcaption>
    <figcaption v-else-if="mode === 'predictions'">Image 2 detail, with a fixed coordinate scale. Blue predicts; gold observes; purple constrains to a line. The displacement arrow's tail is clipped when it leaves the crop.</figcaption>
  </figure>
</template>

<style scoped>
.scene-figure {
  margin: 16px 0;
  background: #1B1F27;
  color: #E6EDF3;
  font-family: "Latin Modern Roman", "CMU Serif", "Computer Modern", serif;
}
.scene-viewport { overflow-x: auto; }
.orbitable { cursor: grab; user-select: none; }
.orbiting { cursor: grabbing; }
.scene-orbit-controls {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 16px;
  color: #9AA5B1;
  font-size: 14px;
}
.scene-orbit-controls button {
  padding: 4px 8px;
  border: 1px solid #9AA5B1;
  color: #F0C060;
  background: #1B1F27;
  font: inherit;
  cursor: pointer;
}
svg {
  display: block;
  width: 100%;
  min-width: 620px;
  font-family: inherit;
  font-size: 16px;
}
figcaption {
  padding: 0 16px 16px;
  color: #9AA5B1;
  line-height: 1.5;
  font-size: 16px;
}
</style>
