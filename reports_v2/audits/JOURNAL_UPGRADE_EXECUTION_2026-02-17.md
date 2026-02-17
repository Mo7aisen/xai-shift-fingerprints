# Journal Upgrade Execution (2026-02-17)

## Scope
- Target: journal-strengthening pass for external domain validation and baseline comparisons.
- Experiment chain: `jsrt_to_shenzhen`.
- Seeds: `42 43 44 45 46`.
- Endpoints: `predicted_mask`, `mask_free`.

## Submitted Slurm Jobs
- Gate-3 (full seeds + statistical report): `2668` (`RUNNING` at submission time)
- Gate-4 (IG quality + robustness): `2669` (`afterok:2668`)
- Gate-5 (clinical relevance + determinism + final decision): `2670` (`afterok:2669`)

## Output Tagging (no overwrite of Montgomery artifacts)
- Output tag: `jsrt_to_shenzhen_2026-02-17`
- Gate-3 outputs:
  - `reports_v2/audits/GATE3_FULL_SEEDS_STATS_jsrt_to_shenzhen_2026-02-17.csv`
  - `reports_v2/audits/GATE3_FULL_SEEDS_SUMMARY_jsrt_to_shenzhen_2026-02-17.json`
  - `reports_v2/audits/GATE3_FULL_SEEDS_STATS_jsrt_to_shenzhen_2026-02-17.md`
- Gate-4 outputs:
  - `reports_v2/audits/GATE4_IG_ROBUSTNESS_jsrt_to_shenzhen_2026-02-17.json`
  - `reports_v2/audits/GATE4_IG_ROBUSTNESS_jsrt_to_shenzhen_2026-02-17.md`
- Gate-5 outputs:
  - `reports_v2/audits/GATE5_CLINICAL_SUMMARY_jsrt_to_shenzhen_2026-02-17.json`
  - `reports_v2/audits/GATE5_BITWISE_DETERMINISM_SUMMARY_jsrt_to_shenzhen_2026-02-17.json`
  - `reports_v2/audits/GATE5_FINAL_SUMMARY_jsrt_to_shenzhen_2026-02-17.json`

## Baseline Job (pending due user submit-limit at launch time)
Submit after one queued job clears:

```bash
cd /storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full
sbatch --dependency=afterok:2670 \
  --export=ALL,XFP_EXPERIMENT=jsrt_to_shenzhen,XFP_SUBSET=full \
  scripts/run_journal_baselines_slurm.sh
```

Expected baseline artifacts:
- `results/baselines/jsrt_to_shenzhen_full/journal_ood_baselines_jsrt_to_shenzhen_full.csv`
- `results/baselines/jsrt_to_shenzhen_full/journal_ood_baselines_jsrt_to_shenzhen_full.md`

Automated watcher option (already started on cluster for dependency `2670`):

```bash
nohup scripts/submit_baseline_when_slot_free.sh 2670 jsrt_to_shenzhen full >/dev/null 2>&1 &
```

## Paper Table Builder
After Gate-3/5 and baselines complete:

```bash
python scripts/build_journal_results_table.py \
  --experiment jsrt_to_shenzhen \
  --gate3-csv reports_v2/audits/GATE3_FULL_SEEDS_STATS_jsrt_to_shenzhen_2026-02-17.csv \
  --gate5-json reports_v2/audits/GATE5_CLINICAL_SUMMARY_jsrt_to_shenzhen_2026-02-17.json \
  --baseline-csv results/baselines/jsrt_to_shenzhen_full/journal_ood_baselines_jsrt_to_shenzhen_full.csv \
  --out-csv reports_v2/audits/JOURNAL_MAIN_RESULTS_jsrt_to_shenzhen_2026-02-17.csv \
  --out-md reports_v2/audits/JOURNAL_MAIN_RESULTS_jsrt_to_shenzhen_2026-02-17.md
```
