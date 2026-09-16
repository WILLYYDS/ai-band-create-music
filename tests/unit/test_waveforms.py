import inspect

import numpy as np

from app.services.waveforms import (
    WAVEFORM_BIN_COUNT,
    extract_waveform,
    merge_waveform_sets,
    summarize_waveform,
)


def test_summarize_waveform_preserves_relative_energy() -> None:
    samples = np.concatenate((np.full(100, 0.25), np.full(100, 1.0))).astype(np.float32)

    assert summarize_waveform(samples, 2) == [0.25, 1.0]
    assert summarize_waveform(np.zeros(8, dtype=np.float32), 4) == [0.0] * 4


def test_summarize_waveform_defaults_to_the_editor_bin_count() -> None:
    samples = np.ones(200_000, dtype=np.float32)

    assert WAVEFORM_BIN_COUNT == 640
    assert len(summarize_waveform(samples)) == WAVEFORM_BIN_COUNT


def test_both_entry_points_share_one_default_bin_count() -> None:
    # 同一模块内的两个默认值必须一致，否则 "full" 与分轨的 x 轴长度不同。
    assert (
        inspect.signature(summarize_waveform).parameters["bin_count"].default
        == inspect.signature(extract_waveform).parameters["bin_count"].default
        == WAVEFORM_BIN_COUNT
    )


def test_merge_waveform_sets_replaces_legacy_bins_with_fresh_ones() -> None:
    legacy = {"full": [0.25] * 64}
    fresh = {"full": [0.25] * 640, "vocal": [0.5] * 640, "drums": [0.5] * 640}

    assert merge_waveform_sets(legacy, fresh) == fresh


def test_merge_waveform_sets_keeps_stored_bins_when_extraction_failed() -> None:
    legacy = {"full": [0.25] * 64}

    assert merge_waveform_sets(legacy, {}) == legacy
    assert merge_waveform_sets(None, {}) == {}
    assert merge_waveform_sets("not a mapping", {}) == {}
