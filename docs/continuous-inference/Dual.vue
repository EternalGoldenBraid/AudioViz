<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import data from "./dual-capture.json";
import inputAtlasUrl from "./dual-input.png?url";

const time = ref(0);
const playing = ref(true);
const loop = ref(true);
const speed = ref(1);
const metric = ref("change");
const channels = ref("both");
const focus = ref(-1);
const yaw = ref(.42);
const pitch = ref(.4);
const canvases = [];
const inputCanvas = ref(null);
const inputReady = ref(false);
const inputError = ref("");
let inputImage = null;
let inputFrame = -1;
let animation = null;
let observer = null;
let previousStamp = null;
let drag = null;
const valid = data.schema === "continuous-dual-v1";
const hasInput = valid && data.settings.input_atlas !== undefined;
const frame = computed(() => Math.min(data.records.length-1, Math.floor(time.value*data.settings.fps)));
const fraction = computed(() => Math.min(1, time.value*data.settings.fps-frame.value));
const current = computed(() => data.records[frame.value]);
const next = computed(() => data.records[Math.min(frame.value+1, data.records.length-1)]);
const scales = { nodes: {}, messages: {} };
if (valid) for (const record of data.records) {
  record.nodes.forEach((value, index) => {
    const key = data.nodes[index].layer;
    scales.nodes[key] = Math.max(scales.nodes[key] ?? 0, value[3]);
  });
  record.messages.forEach((value, index) => {
    const edge = data.edges[index];
    const key = `${edge.family}-${edge.kind}`;
    const old = scales.messages[key] ?? [0, 0];
    scales.messages[key] = [Math.max(old[0], value[1]), Math.max(old[1], value[3])];
  });
}
const number = value => value !== 0 && Math.abs(value) < .0001 ? value.toExponential(3) : value.toFixed(4);
const interpolate = (a, b) => a+(b-a)*fraction.value;
function signal(key, index) {
  return current.value[key][index].map((value, component) =>
    interpolate(value, next.value[key][index][component]),
  );
}
const selectedNode = computed(() => focus.value < 0 ? null : data.nodes[focus.value]);
const selectedSignal = computed(() => selectedNode.value ? signal("nodes", focus.value) : null);
const incident = computed(() => data.edges.map((edge, index) => ({ ...edge, index })).filter(
  edge => edge.source === focus.value || edge.target === focus.value,
));

function projection(width, height) {
  const cy = Math.cos(yaw.value), sy = Math.sin(yaw.value);
  const cp = Math.cos(pitch.value), sp = Math.sin(pitch.value);
  const scale = Math.min(width/7, height/7.6);
  return ([x, y, z]) => {
    const rx = cy*x+sy*z, rz = -sy*x+cy*z;
    const ry = cp*y-sp*rz, depth = sp*y+cp*rz;
    const perspective = 8/(8-depth);
    return { x: width/2+rx*scale*perspective, y: height/2-ry*scale*perspective, depth };
  };
}
function curve(edge, t) {
  const a = data.nodes[edge.source].position, b = data.nodes[edge.target].position;
  const lo = data.nodes[Math.min(edge.source, edge.target)].position;
  const hi = data.nodes[Math.max(edge.source, edge.target)].position;
  const dx = hi[0]-lo[0], dz = hi[2]-lo[2];
  const length = Math.hypot(dx, dz);
  const normal = length === 0 ? [.8, 0, .6] : [-dz/length, .12, dx/length];
  const bend = (edge.source < edge.target ? 1 : -1)*(edge.kind === "prediction" ? .2 : .35);
  const u = 1-t;
  return a.map((value, axis) => {
    const c1 = value+(b[axis]-value)*.33+normal[axis]*bend;
    const c2 = value+(b[axis]-value)*.67+normal[axis]*bend;
    return u*u*u*value+3*u*u*t*c1+3*u*t*t*c2+t*t*t*b[axis];
  });
}
function color(sign) {
  if (sign === 0) return [185, 196, 215];
  return sign < 0 ? [72, 211, 236] : [255, 158, 119];
}
function dot(ctx, x, y, radius, rgb, opacity) {
  const glow = ctx.createRadialGradient(x, y, 0, x, y, radius);
  glow.addColorStop(0, `rgba(${rgb.join(",")},${opacity})`);
  glow.addColorStop(.3, `rgba(${rgb.join(",")},${opacity*.65})`);
  glow.addColorStop(1, `rgba(${rgb.join(",")},0)`);
  ctx.fillStyle = glow;
  ctx.beginPath(); ctx.arc(x, y, radius, 0, 2*Math.PI); ctx.fill();
}
function render() {
  if (!valid) return;
  canvases.forEach((canvas, panel) => {
    if (!canvas) return;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    const ratio = Math.min(window.devicePixelRatio, 2);
    if (canvas.width !== Math.round(width*ratio) || canvas.height !== Math.round(height*ratio)) {
      canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const project = projection(width, height);
    const points = data.nodes.map(node => project(node.position));
    if (panel === 1) {
      const offset = metric.value === "change" ? 2 : 0;
      data.edges.forEach((edge, index) => {
        if (channels.value !== "both" && channels.value !== edge.kind) return;
        const values = signal("messages", index);
        const maximum = scales.messages[`${edge.family}-${edge.kind}`][offset/2];
        const intensity = maximum === 0 ? 0 : Math.sqrt(values[offset+1]/maximum);
        if (intensity === 0) return;
        const relevant = focus.value < 0 || edge.source === focus.value || edge.target === focus.value;
        const opacity = intensity*(relevant ? .65 : .06);
        const rgb = color(values[offset]);
        ctx.strokeStyle = `rgba(${rgb.join(",")},${opacity*.15})`;
        ctx.lineWidth = .6;
        ctx.beginPath();
        for (let step=0; step<=12; step++) {
          const point = project(curve(edge, step/12));
          if (step === 0) ctx.moveTo(point.x, point.y);
          else ctx.lineTo(point.x, point.y);
        }
        ctx.stroke();
        const count = 2;
        for (let bead=0; bead<count; bead++) {
          const t = (time.value*.65+bead/count)%1;
          const point = project(curve(edge, t));
          const radius = (edge.kind === "prediction" ? 2.8 : 1.7)+intensity*2.3;
          dot(ctx, point.x, point.y, radius*2, rgb, opacity);
          if (edge.kind === "feedback") {
            const trailing = project(curve(edge, Math.max(0, t-.025)));
            dot(ctx, trailing.x, trailing.y, radius, rgb, opacity*.6);
          }
        }
      });
      if (inputReady.value && inputCanvas.value && inputFrame !== frame.value) {
        const [height, width] = data.settings.shape;
        const columns = data.settings.input_atlas.columns;
        inputCanvas.value.getContext("2d").drawImage(
          inputImage, (frame.value%columns)*width, Math.floor(frame.value/columns)*height,
          width, height, 0, 0, width, height,
        );
        inputFrame = frame.value;
      }
    }
    const order = points.map((point, index) => ({ ...point, index })).sort((a, b) => a.depth-b.depth);
    for (const point of order) {
      const node = data.nodes[point.index], values = signal("nodes", point.index);
      const intensity = scales.nodes[node.layer] === 0 ? 0 : Math.sqrt(values[3]/scales.nodes[node.layer]);
      const rgb = color(values[2]);
      const relevant = focus.value < 0 || focus.value === point.index;
      if (panel === 0 && intensity > 0) {
        dot(ctx, point.x, point.y, 12+intensity*15, rgb, intensity*(relevant ? .85 : .16));
        const phase = (time.value*.65)%1;
        ctx.strokeStyle = `rgba(${rgb.join(",")},${intensity*(1-phase)*(relevant ? .7 : .1)})`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(point.x, point.y, 4+phase*19, 0, 2*Math.PI); ctx.stroke();
      }
      ctx.fillStyle = panel === 0 ? "#b4c3d1" : "#405165";
      ctx.beginPath(); ctx.arc(point.x, point.y, node.layer === "hidden" ? 3 : 2, 0, 2*Math.PI); ctx.fill();
      if (point.index === focus.value) {
        ctx.strokeStyle = "#f3e0bb"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(point.x, point.y, 10, 0, 2*Math.PI); ctx.stroke();
      }
    }
    ctx.font = "11px sans-serif"; ctx.fillStyle = "#8899ac"; ctx.textAlign = "left";
    for (const layer of ["canvas", "hidden", "sensory", "evidence"]) {
      const positions = data.nodes.filter(node => node.layer === layer).map(node => node.position);
      const center = positions[0].map((_, axis) =>
        positions.reduce((sum, value) => sum+value[axis], 0)/positions.length,
      );
      const point = project(center);
      ctx.fillText(layer.toUpperCase(), 12, Math.max(15, Math.min(height-10, point.y)));
    }
  });
}
function tick(stamp) {
  if (previousStamp !== null && playing.value) {
    time.value += (stamp-previousStamp)/1000*Number(speed.value);
    if (time.value >= data.settings.duration) {
      if (loop.value) time.value %= data.settings.duration;
      else { time.value = data.settings.duration; playing.value = false; }
    }
  }
  previousStamp = stamp;
  render();
  animation = playing.value ? requestAnimationFrame(tick) : null;
}
function togglePlay() {
  if (!playing.value && time.value >= data.settings.duration) time.value = 0;
  playing.value = !playing.value;
}
function seek(value) {
  playing.value = false;
  time.value = Number(value);
  render();
}
function restart() {
  time.value = 0; previousStamp = null; render();
}
function resetCamera() {
  yaw.value = .42; pitch.value = .4; render();
}
function pointerDown(event) {
  event.currentTarget.setPointerCapture(event.pointerId);
  drag = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
}
function pointerMove(event) {
  if (!drag || drag.id !== event.pointerId) return;
  const dx = event.clientX-drag.x, dy = event.clientY-drag.y;
  drag.moved ||= Math.abs(dx)+Math.abs(dy) > 2;
  yaw.value += dx*.007;
  pitch.value = Math.max(-1.2, Math.min(1.2, pitch.value+dy*.007));
  drag.x = event.clientX; drag.y = event.clientY;
  render();
}
function pointerUp(event) {
  if (drag && !drag.moved) {
    const canvas = event.currentTarget, rect = canvas.getBoundingClientRect();
    const project = projection(rect.width, rect.height);
    const closest = data.nodes.map((node, index) => {
      const point = project(node.position);
      return { index, distance: Math.hypot(point.x-event.clientX+rect.left, point.y-event.clientY+rect.top) };
    }).sort((a, b) => a.distance-b.distance)[0];
    focus.value = closest.distance < 16 ? closest.index : -1;
  }
  drag = null;
  render();
}
watch(playing, active => {
  if (animation !== null) cancelAnimationFrame(animation);
  animation = null;
  previousStamp = null;
  if (active) animation = requestAnimationFrame(tick);
  else render();
});
watch([metric, channels, focus], render, { flush: "post" });
onMounted(() => {
  if (!valid) return;
  observer = new ResizeObserver(render);
  canvases.forEach(canvas => observer.observe(canvas));
  animation = requestAnimationFrame(tick);
  if (hasInput) {
    inputImage = new Image();
    inputImage.onload = () => {
      const [height, width] = data.settings.shape;
      const atlas = data.settings.input_atlas;
      if (inputImage.naturalWidth !== atlas.columns*width
        || inputImage.naturalHeight !== Math.ceil(atlas.frames/atlas.columns)*height
        || atlas.frames !== data.records.length) {
        inputError.value = "The recorded input image does not match this recording. Regenerate it with the command below.";
        return;
      }
      inputReady.value = true;
      render();
    };
    inputImage.onerror = () => {
      inputError.value = "The recorded input image failed to load. Regenerate it with the command below.";
    };
    inputImage.src = inputAtlasUrl;
  }
});
onBeforeUnmount(() => {
  if (animation !== null) cancelAnimationFrame(animation);
  observer?.disconnect();
  if (inputImage) { inputImage.onload = null; inputImage.onerror = null; }
});
</script>

<template>
  <p v-if="!valid" role="alert">This recording has an incompatible schema. Regenerate dual-capture.json using the command below.</p>
  <section v-else class="dual-view" aria-label="Paired continuous inference rendering">
    <div class="transport">
      <button @click="togglePlay">{{ playing ? "Pause" : "Play" }}</button>
      <button @click="restart">Restart</button>
      <label><input v-model="loop" type="checkbox"> Loop</label>
      <label>Speed <select v-model="speed"><option :value=".5">½×</option><option :value="1">1×</option><option :value="2">2×</option></select></label>
      <input class="scrubber" aria-label="Simulation time" type="range" min="0" :max="data.settings.duration"
        :step="1/data.settings.fps" :value="time" @input="seek($event.target.value)">
      <output>{{ time.toFixed(3) }} / {{ data.settings.duration.toFixed(1) }} s</output>
    </div>
    <div class="options">
      <label>Ripples <select v-model="metric"><option value="change">Message changes</option><option value="value">Message values</option></select></label>
      <label>Channels <select v-model="channels"><option value="both">Prediction + feedback</option><option value="prediction">Predictions</option><option value="feedback">Feedback</option></select></label>
      <label>Focus <select v-model="focus"><option :value="-1">All regions</option><option v-for="(node, index) in data.nodes" :key="node.id" :value="index">{{ node.label }}</option></select></label>
      <button @click="resetCamera">Reset camera</button>
    </div>
    <div class="pair">
      <figure v-for="(title, index) in ['Node pulses', 'Connection ripples']" :key="title">
        <figcaption><strong>{{ title }}</strong><span>{{ index === 0 ? 'Actual state change · quiet connections' : 'Directed messages · quiet node landmarks' }}</span></figcaption>
        <canvas :ref="element => canvases[index] = element" :data-frame="frame"
          :data-camera="`${yaw.toFixed(4)},${pitch.toFixed(4)}`" :aria-label="title" role="img"
          @pointerdown="pointerDown" @pointermove="pointerMove" @pointerup="pointerUp" @pointercancel="drag = null"></canvas>
      </figure>
    </div>
    <div class="input-row">
      <figure class="experiment-input">
        <figcaption><strong>Experiment input</strong><span>Recorded RGB stimulus · same simulation clock</span></figcaption>
        <p v-if="!hasInput" role="alert">This recording has no input images. Regenerate it with the command below.</p>
        <p v-else-if="inputError" role="alert">{{ inputError }}</p>
        <p v-else-if="!inputReady">Loading recorded input…</p>
        <canvas v-if="hasInput" v-show="inputReady" ref="inputCanvas" :width="data.settings.shape[1]"
          :height="data.settings.shape[0]" :data-frame="frame" aria-label="Experiment input" role="img"></canvas>
      </figure>
    </div>
    <p class="legend"><span class="positive">Warm: positive mean</span> · <span class="negative">Cool: negative mean</span> · Neutral: zero mean ·
      Predictions: smooth beads · Feedback: paired fine ripples.
      Brightness uses fixed per-layer/channel recording scales; carrier motion is illustrative, not a measured delay.</p>
    <p class="energy">
      Total objective {{ number(interpolate(current.energy[0], next.energy[0])) }} ·
      Hidden {{ number(interpolate(current.energy[1], next.energy[1])) }} ·
      Sensory {{ number(interpolate(current.energy[2], next.energy[2])) }}
    </p>
    <details class="inspection">
      <summary>Recorded signals{{ selectedNode ? ` · ${selectedNode.label}` : '' }}</summary>
      <p v-if="!selectedNode">Select a node in either view or use Focus to inspect its signals. Both views remain synchronized.</p>
      <template v-else>
        <p class="node-signal">Mean state {{ number(selectedSignal[0]) }} · RMS state change {{ number(selectedSignal[3]) }} ·
          RMS end-state gradient {{ number(interpolate(current.gradients[focus], next.gradients[focus])) }}</p>
        <div class="signal-table"><table>
          <thead><tr><th>Connection</th><th>Type</th><th>Mean message</th><th>RMS change</th></tr></thead>
          <tbody><tr v-for="edge in incident" :key="edge.index">
            <td>{{ data.nodes[edge.source].label }} → {{ data.nodes[edge.target].label }}</td>
            <td>{{ edge.kind }}</td><td>{{ number(signal("messages", edge.index)[0]) }}</td>
            <td>{{ number(signal("messages", edge.index)[3]) }}</td>
          </tr></tbody>
        </table></div>
      </template>
    </details>
  </section>
</template>

<style scoped>
:global(.experiment-page) { max-width:1320px; margin:0 auto; padding:28px 32px; }
.dual-view { margin:24px 0; }
.transport, .options { display:flex; flex-wrap:wrap; align-items:center; gap:12px; font-size:12px; margin:12px 0; }
button, select { border:1px solid #3b4756; border-radius:5px; background:#19212b; color:#e6edf4; padding:5px 8px; }
label { display:flex; align-items:center; gap:5px; }
.scrubber { flex:1; min-width:120px; accent-color:#82c7d3; }
output { font-variant-numeric:tabular-nums; min-width:100px; }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
figure { margin:0; min-width:0; border:1px solid #2b3644; border-radius:12px; overflow:hidden; background:radial-gradient(ellipse at center, #101d2b, #080e16); }
figcaption { display:flex; flex-direction:column; padding:15px 17px 0; gap:3px; font-size:14px; }
figcaption span { font-size:11px; color:#8798ab; }
canvas { display:block; width:100%; height:520px; touch-action:none; cursor:grab; }
canvas:active { cursor:grabbing; }
.input-row { display:flex; justify-content:center; margin-top:16px; }
.experiment-input { width:418px; max-width:100%; padding-bottom:16px; }
.experiment-input canvas { width:calc(100% - 32px); max-width:384px; height:auto; margin:14px 16px 0; image-rendering:pixelated; touch-action:auto; cursor:default; }
.experiment-input p { font-size:12px; padding:0 16px; }
.legend, .energy { font-size:12px; line-height:1.6; color:#98a8ba; }
.positive { color:#ffae91; } .negative { color:#76d3e9; }
.energy { font-variant-numeric:tabular-nums; }
.inspection { border-top:1px solid #2b3644; padding-top:12px; font-size:12px; }
.inspection summary { cursor:pointer; }
.signal-table { overflow-x:auto; max-height:340px; overflow-y:auto; }
.signal-table table { font-size:11px; white-space:nowrap; }
@media(max-width:800px) {
  :global(.experiment-page) { padding:24px 16px; }
  .pair { grid-template-columns:1fr; } canvas { height:480px; }
}
</style>
