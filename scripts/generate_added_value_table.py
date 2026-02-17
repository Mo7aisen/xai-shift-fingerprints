#!/usr/bin/env python3
"""Generate added-value comparison table for fingerprinting vs baselines."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SEED = 2025
N_BOOT = 1000


def stratified_boot_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    b0 = rng.choice(idx0, size=idx0.size, replace=True)
    b1 = rng.choice(idx1, size=idx1.size, replace=True)
    return np.concatenate([b0, b1])


def auc_with_ci(y: np.ndarray, score: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    point = float(roc_auc_score(y, score))
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = stratified_boot_indices(y, rng)
        boots[i] = roc_auc_score(y[idx], score[idx])
    low, high = np.quantile(boots, [0.025, 0.975])
    return point, float(low), float(high), boots


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running = max(running, adj)
        adjusted[name] = running
    return adjusted


def sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_pvalue(p: float) -> str:
    if p <= 0:
        return "p<1e-300"
    if p < 1e-4:
        return f"p={p:.2e}"
    return f"p={p:.4f}"


def build_dataframe() -> pd.DataFrame:
    jsrt = pd.read_parquet(ROOT / "data/fingerprints/jsrt_to_nih/jsrt.parquet")
    jsrt = jsrt[["sample_id", "dice", "attribution_abs_sum", "border_abs_sum", "coverage_auc", "hist_entropy"]].copy()
    jsrt["dataset"] = "jsrt"

    nih = pd.read_parquet(ROOT / "data/fingerprints/jsrt_to_nih/nih_chestxray14.parquet")
    nih = nih[["sample_id", "dice", "attribution_abs_sum", "border_abs_sum", "coverage_auc", "hist_entropy"]].copy()
    nih["dataset"] = "nih_chestxray14"

    fp = pd.concat([jsrt, nih], ignore_index=True)

    energy = pd.read_csv(ROOT / "results/baselines/energy_ood_scores.csv")
    energy = energy[["sample_id", "dataset", "msp_score"]]

    resnet = pd.read_csv(ROOT / "results/baselines/resnet_ood_scores.csv")
    resnet = resnet[["sample_id", "dataset", "score"]].rename(columns={"score": "resnet_score"})

    df = fp.merge(energy, on=["sample_id", "dataset"], how="inner")
    df = df.merge(resnet, on=["sample_id", "dataset"], how="inner")

    # Balanced JSRT vs NIH slice to avoid severe class imbalance bias.
    rng = np.random.default_rng(SEED)
    id_df = df[df["dataset"] == "jsrt"].copy()
    ood_df = df[df["dataset"] == "nih_chestxray14"].copy()
    ood_df = ood_df.sample(n=len(id_df), random_state=SEED)
    out = pd.concat([id_df, ood_df], ignore_index=True)

    # Fingerprint anomaly score: average absolute z-score against ID reference statistics.
    feat_cols = ["attribution_abs_sum", "border_abs_sum", "coverage_auc", "hist_entropy"]
    mu = id_df[feat_cols].mean()
    sd = id_df[feat_cols].std(ddof=0).replace(0.0, 1.0)
    z = (out[feat_cols] - mu) / sd
    out["fingerprint_score"] = z.abs().mean(axis=1)

    out["dice_score"] = 1.0 - out["dice"]
    out["confidence_score"] = -out["msp_score"]
    out["label_ood"] = (out["dataset"] == "nih_chestxray14").astype(int)
    return out


def main() -> None:
    out_dir = ROOT / "reports" / "added_value"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataframe()
    y = df["label_ood"].to_numpy(dtype=int)

    method_scores = {
        "Fingerprinting (our method)": df["fingerprint_score"].to_numpy(dtype=float),
        "Dice score alone": df["dice_score"].to_numpy(dtype=float),
        "Confidence thresholding": df["confidence_score"].to_numpy(dtype=float),
        "ResNet-OOD baseline": df["resnet_score"].to_numpy(dtype=float),
    }

    auc_stats: dict[str, tuple[float, float, float, np.ndarray]] = {}
    for method, score in method_scores.items():
        auc_stats[method] = auc_with_ci(y, score)

    # Bootstrap difference test against fingerprinting.
    fp_boot = auc_stats["Fingerprinting (our method)"][3]
    raw_p = {}
    for method in ["Dice score alone", "Confidence thresholding", "ResNet-OOD baseline"]:
        diff = fp_boot - auc_stats[method][3]
        p = 2.0 * min(float(np.mean(diff <= 0)), float(np.mean(diff >= 0)))
        p = max(p, 1.0 / (N_BOOT + 1.0))
        raw_p[method] = max(min(p, 1.0), 0.0)
    holm_p = holm_bonferroni(raw_p)

    rows = []
    for method in [
        "Fingerprinting (our method)",
        "Dice score alone",
        "Confidence thresholding",
        "ResNet-OOD baseline",
    ]:
        auc, lo, hi, _ = auc_stats[method]
        marker = ""
        if method in holm_p:
            marker = sig_marker(holm_p[method])

        r, p = pearsonr(method_scores[method], df["dice"].to_numpy(dtype=float))
        corr_txt = f"rho={r:.3f}, {format_pvalue(float(p))}"

        spatial = "Qualitative (attribution concentrates near failure regions)" if method.startswith("Fingerprint") else "N/A"

        rows.append(
            {
                "Method": method,
                "OOD Detectability (AUROC ± 95% CI)": f"{auc:.3f} [{lo:.3f}-{hi:.3f}]{marker}",
                "Error Correlation (Pearson r with Dice)": corr_txt,
                "Spatial Interpretability Score": spatial,
                "holm_p_vs_fingerprinting": holm_p.get(method, np.nan),
                "raw_p_vs_fingerprinting": raw_p.get(method, np.nan),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "added_value_comparison.csv", index=False)

    # TeX table body
    tex_rows = []
    for _, row in out_df.iterrows():
        tex_rows.append(
            f"{row['Method']} & {row['OOD Detectability (AUROC ± 95% CI)']} & {row['Error Correlation (Pearson r with Dice)']} & {row['Spatial Interpretability Score']} \\\\"
        )

    tex = "\n".join(tex_rows) + "\n"
    for rel in [
        ROOT / "manuscript" / "tables" / "added_value_comparison.tex",
        ROOT / "final_elsevier_submission" / "tables" / "added_value_comparison.tex",
        ROOT / "submission_medical_image_analysis" / "tables" / "added_value_comparison.tex",
    ]:
        rel.parent.mkdir(parents=True, exist_ok=True)
        rel.write_text(tex, encoding="utf-8")

    notes = [
        "# Added-Value Statistics",
        "",
        "- Task: balanced JSRT vs NIH OOD discrimination (n_id=n_ood=247).",
        "- CI method: stratified bootstrap (n=1000, seed=2025).",
        "- Significance markers compare each baseline against fingerprinting using bootstrap AUROC differences and Holm-Bonferroni correction.",
        "",
        "## Holm-corrected p-values vs Fingerprinting",
    ]
    for method in ["Dice score alone", "Confidence thresholding", "ResNet-OOD baseline"]:
        notes.append(f"- {method}: raw {format_pvalue(raw_p[method])}, holm {format_pvalue(holm_p[method])}")

    (out_dir / "added_value_stats.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"[added-value] wrote {out_dir / 'added_value_comparison.csv'}")


if __name__ == "__main__":
    main()
