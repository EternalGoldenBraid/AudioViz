import json
from pathlib import Path
import subprocess

import numpy as np

from audioviz.audio_processing.cochlea import Cochlea, CochleaConfig


def test_browser_trajectory_matches_python_inference_and_free_dynamics():
    module = Path(__file__).resolve().parents[2] / "docs/cochlea/simulation.mjs"
    cases = [
        {"qualityFactor": 16, "correctionRate": .5},
        {"qualityFactor": 2, "correctionRate": 1},
        {"qualityFactor": 32, "correctionRate": 0, "initialState": [0.2, -0.1] * 16},
    ]
    samples = np.random.default_rng(11).normal(scale=.2, size=257)
    script = f"""
        import {{ simulate, activity, stimulus, FREQUENCIES }} from {json.dumps(module.as_uri())};
        const cases = {json.dumps(cases)};
        const samples = {json.dumps(samples.tolist())};
        const runs = cases.map(options => {{
            const t = simulate(samples, options);
            return Object.fromEntries(Object.entries(t).map(([k,v]) => [k, ArrayBuffer.isView(v) ? [...v] : v]));
        }});
        const burst = stimulus(371);
        const full = simulate(burst);
        const changed = Float64Array.from(burst, (v,n) => n > 600 ? -v : v);
        const future = simulate(changed);
        if (full.prediction.slice(0,601).some((v,n) => v !== future.prediction[n])) throw Error('Not causal');
        for (const options of [{{qualityFactor: .5}}, {{correctionRate: 2}}]) {{
            let rejected = false;
            try {{ simulate(samples, options); }} catch (e) {{ rejected = e instanceof RangeError; }}
            if (!rejected) throw Error('Invalid parameters accepted');
        }}
        console.log(JSON.stringify({{runs, burst:[...burst], activity:activity(full), frequencies:FREQUENCIES}}));
    """
    result = json.loads(subprocess.run(
        ["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True,
    ).stdout)
    for options, actual in zip(cases, result["runs"]):
        bank = Cochlea(22050, CochleaConfig(
            quality_factor=options["qualityFactor"], correction_rate=options["correctionRate"],
        ))
        if "initialState" in options:
            bank.state[:] = np.array(options["initialState"]).reshape(16, 2)
        for n, sample in enumerate(samples):
            prior_x = bank._cosine * bank.state[:, 0] + bank._sine * bank.state[:, 1]
            prior_q = bank._cosine * bank.state[:, 1] - bank._sine * bank.state[:, 0]
            block = bank.process(np.array([sample]))
            segment = slice(n * 16, (n + 1) * 16)
            np.testing.assert_allclose(actual["beforeX"][segment], prior_x, atol=2e-12)
            np.testing.assert_allclose(actual["beforeQ"][segment], prior_q, atol=2e-12)
            np.testing.assert_allclose(actual["afterX"][segment], bank.state[:, 0], atol=2e-12)
            np.testing.assert_allclose(actual["afterQ"][segment], bank.state[:, 1], atol=2e-12)
            np.testing.assert_allclose(actual["prediction"][n], block.prediction[0], atol=2e-12)
            np.testing.assert_allclose(actual["error"][n], block.error[0], atol=2e-12)
        np.testing.assert_allclose(np.array(actual["prediction"]) + actual["error"], samples, atol=1e-15)
    burst = np.array(result["burst"])
    assert np.max(np.abs(burst)) <= .35
    assert np.all(burst[1764:] == 0)
    bank = Cochlea(22050, history_hop=32)
    bank.process(burst)
    np.testing.assert_allclose(result["frequencies"], bank.frequencies, rtol=1e-14)
    np.testing.assert_allclose(result["activity"], bank.activity_history().values, atol=2e-12)
