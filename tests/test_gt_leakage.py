"""Static guardrails against GT leakage in deployment endpoints."""

from __future__ import annotations

from pathlib import Path

from xfp.data.pipeline import infer_patient_id


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "src" / "xfp" / "fingerprint" / "runner.py"
RUN_FINGERPRINT_SCRIPT = ROOT / "scripts" / "run_fingerprint.py"


def test_runner_defines_endpoint_modes() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "ENDPOINT_MODES" in text
    assert "upper_bound_gt" in text
    assert "predicted_mask" in text
    assert "mask_free" in text


def test_gt_mask_loaded_only_for_upper_bound_mode() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8")
    expected = 'gt_mask = data["mask"].astype(np.uint8) if endpoint_mode == "upper_bound_gt" else None'
    assert expected in text


def test_mask_free_feature_reducer_exists() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8")
    assert "def _reduce_attribution_mask_free(" in text
    assert "for feature_key in MASK_DEPENDENT_FEATURES:" in text


def test_cli_exposes_endpoint_mode() -> None:
    text = RUN_FINGERPRINT_SCRIPT.read_text(encoding="utf-8")
    assert "--endpoint-mode" in text
    assert "choices=[\"upper_bound_gt\", \"predicted_mask\", \"mask_free\"]" in text


def test_infer_patient_id_patterns() -> None:
    assert infer_patient_id("shenzhen", "CHNCXR_0110_1") == "CHNCXR_0110"
    assert infer_patient_id("montgomery", "MCUCXR_0001_0") == "MCUCXR_0001"
    assert infer_patient_id("nih_chestxray14", "00000001_001") == "00000001"
    assert infer_patient_id("jsrt", "JPCLN001") == "JPCLN001"
