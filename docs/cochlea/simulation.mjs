// Browser counterpart of audio_processing/cochlea.py; checked against it in tests/docs.
export const SAMPLE_RATE = 22050;
export const NODES = 16;
export const DURATION = 0.16;
export const BURST_END = 0.08;
export const AMPLITUDE = 0.35;
export const FREQUENCIES = Object.freeze(
  Array.from({ length: NODES }, (_, i) => 80 * (8000 / 80) ** (i / (NODES - 1))),
);

export function stimulus(frequency, burst = true) {
  if (!Number.isFinite(frequency) || frequency <= 0 || frequency >= SAMPLE_RATE / 2) {
    throw new RangeError("Stimulus frequency must be positive and below Nyquist.");
  }
  return Float64Array.from({ length: Math.round(DURATION * SAMPLE_RATE) }, (_, n) => {
    const t = n / SAMPLE_RATE;
    const envelope = Math.min(1, t / 0.002) * (burst ? Math.max(0, Math.min(1, (BURST_END - t) / 0.002)) : 1);
    return AMPLITUDE * envelope * Math.sin(2 * Math.PI * frequency * t);
  });
}

export function simulate(samples, { qualityFactor = 16, correctionRate = 0.5, initialState } = {}) {
  if (!samples.length || !Array.from(samples).every(Number.isFinite)) {
    throw new RangeError("Simulation requires a nonempty finite mono waveform.");
  }
  if (!Number.isFinite(qualityFactor) || qualityFactor <= 0.5 ||
      !Number.isFinite(correctionRate) || correctionRate < 0 || correctionRate > 1) {
    throw new RangeError("Use an underdamped Q > 0.5 and a correction rate in [0, 1].");
  }
  const state = initialState ?? new Float64Array(2 * NODES);
  if (state.length !== 2 * NODES || !Array.from(state).every(Number.isFinite)) {
    throw new RangeError("Initial state requires 32 finite position/quadrature coordinates.");
  }
  const x = FREQUENCIES.map((_, i) => state[2 * i]);
  const q = FREQUENCIES.map((_, i) => state[2 * i + 1]);
  const coefficients = FREQUENCIES.map(frequency => {
    const omega = 2 * Math.PI * frequency;
    const gamma = omega / (2 * qualityFactor);
    const nu = Math.sqrt(omega ** 2 - gamma ** 2);
    const radius = Math.exp(-gamma / SAMPLE_RATE);
    return { cosine: radius * Math.cos(nu / SAMPLE_RATE), sine: radius * Math.sin(nu / SAMPLE_RATE) };
  });
  const raw = Float64Array.from(samples);
  const prediction = new Float64Array(raw.length);
  const error = new Float64Array(raw.length);
  const beforeX = new Float64Array(raw.length * NODES);
  const beforeQ = new Float64Array(raw.length * NODES);
  const afterX = new Float64Array(raw.length * NODES);
  const afterQ = new Float64Array(raw.length * NODES);
  let maximumRadius = 0;
  for (let n = 0; n < raw.length; n++) {
    for (let i = 0; i < NODES; i++) {
      const { cosine, sine } = coefficients[i];
      const position = cosine * x[i] + sine * q[i];
      q[i] = cosine * q[i] - sine * x[i];
      x[i] = position;
      beforeX[n * NODES + i] = x[i];
      beforeQ[n * NODES + i] = q[i];
      prediction[n] += x[i];
    }
    error[n] = raw[n] - prediction[n];
    for (let i = 0; i < NODES; i++) {
      x[i] += correctionRate / NODES * error[n];
      afterX[n * NODES + i] = x[i];
      afterQ[n * NODES + i] = q[i];
      maximumRadius = Math.max(maximumRadius, Math.hypot(x[i], q[i]));
    }
  }
  return { raw, prediction, error, beforeX, beforeQ, afterX, afterQ, maximumRadius };
}

export function activity(trace, hop = 32) {
  if (!Number.isInteger(hop) || hop < 1) throw new RangeError("Activity hop must be a positive integer.");
  const columns = Math.floor(trace.raw.length / hop);
  return Array.from({ length: NODES }, (_, i) => Array.from({ length: columns }, (_, k) => {
    let squared = 0;
    for (let n = k * hop; n < (k + 1) * hop; n++) {
      squared += trace.afterX[n * NODES + i] ** 2 + trace.afterQ[n * NODES + i] ** 2;
    }
    return Math.sqrt(squared / hop);
  }));
}
