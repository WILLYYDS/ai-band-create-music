import numpy as np

from app.services.waveforms import summarize_waveform


def test_summarize_waveform_preserves_relative_energy() -> None:
    samples = np.concatenate((np.full(100, 0.25), np.full(100, 1.0))).astype(np.float32)

    assert summarize_waveform(samples, 2) == [0.25, 1.0]
    assert summarize_waveform(np.zeros(8, dtype=np.float32), 4) == [0.0] * 4
