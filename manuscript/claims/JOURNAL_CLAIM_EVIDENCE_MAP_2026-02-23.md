# Journal Claim-to-Evidence Map

- Date tag: `2026-02-23`
- Overall status: `IN_PROGRESS`

## Summary

- PROVEN: `4`
- PARTIAL: `1`
- PENDING: `1`
- BLOCKED: `0`

## Claim Table

| Claim ID | Status | Claim | Rationale |
|---|---|---|---|
| C1 | PARTIAL | Selection governance was frozen before P1.5 completion (top-5 set, backup candidate, no post-hoc tuning guard). | Verifier status=WARN, PASS=44, FAIL=1, PENDING=0. |
| C2 | PROVEN | Hardshift pilot screening ranking and frozen P1.5 candidate selection set are documented and manuscript-ready. | Pilot table=GENERATED, pilot figure=GENERATED. |
| C3 | PROVEN | P1.5 survivor decision evidence (mask_free + Dice-drop constraints; predicted_mask diagnostic) is available for journal reporting. | P1.5 CSV is available and exported into both table and figure assets. |
| C4 | PROVEN | Baseline-comparison and journal-summary manuscript assets are automated, generated, and traceable to source audit CSVs. | journal_main_table=GENERATED, journal_main_figure=GENERATED, baseline_tables=(GENERATED,GENERATED), baseline_figure=GENERATED. |
| C5 | PROVEN | The journal asset pipeline (verify + tables + figures + claims + stats) is rerunnable and auditable from a single command. | asset_pipeline=IN_PROGRESS, table_pipeline=IN_PROGRESS counts={'GENERATED': 6, 'PENDING': 2, 'FAILED': 0, 'SKIPPED': 0}, figure_pipeline=IN_PROGRESS counts={'GENERATED': 5, 'PENDING': 2, 'FAILED': 0, 'SKIPPED': 0}, stats_pipeline=IN_PROGRESS counts={'GENERATED': 4, 'PENDING': 2, 'FAILED': 0, 'SKIPPED': 0}. |
| C6 | PENDING | Journal-level scientific claim (full validation on the selected survivor with ablations/robustness/failure analysis) is supported. | This requires P2/P3 outputs and final analysis artifacts; current work only establishes the foundation and automation. |

## C1: PARTIAL

Selection governance was frozen before P1.5 completion (top-5 set, backup candidate, no post-hoc tuning guard).

- Rationale: Verifier status=WARN, PASS=44, FAIL=1, PENDING=0.

Evidence paths:
- `configs/protocol_hardshift_journal_p2_p3_prelock_v0_1.yaml`
- `reports_v2/audits/HARDSHIFT_JOURNAL_TRACK_PROTOCOL_FREEZE_2026-02-23.md`
- `reports_v2/audits/HARDSHIFT_JOURNAL_TRACK_VERIFY_2026-02-23.json`
- `reports_v2/audits/HARDSHIFT_JOURNAL_TRACK_VERIFY_2026-02-23.md`

Notes:
- This claim is about governance/protocol integrity, not final model performance.
- P1.5 output pending does not invalidate the freeze claim.

## C2: PROVEN

Hardshift pilot screening ranking and frozen P1.5 candidate selection set are documented and manuscript-ready.

- Rationale: Pilot table=GENERATED, pilot figure=GENERATED.

Evidence paths:
- `reports_v2/audits/HARDSHIFT_PILOT_SCREEN_SHENZHEN_2026-02-23.csv`
- `manuscript/tables/generated/hardshift_pilot_screen_shenzhen_2026_02_23.md`
- `manuscript/tables/generated/hardshift_pilot_screen_shenzhen_2026_02_23.tex`
- `manuscript/figures/generated/hardshift_pilot_screen_ranked_resnet_auroc.png`
- `manuscript/figures/generated/hardshift_pilot_screen_ranked_resnet_auroc.pdf`

Notes:
- Top-5 and rank-6 backup highlighting are encoded in the figure config and the verifier.

## C3: PROVEN

P1.5 survivor decision evidence (mask_free + Dice-drop constraints; predicted_mask diagnostic) is available for journal reporting.

- Rationale: P1.5 CSV is available and exported into both table and figure assets.

Evidence paths:
- `reports_v2/audits/HARDSHIFT_P15_SEED42_FINGERPRINT_DICE_SCREEN_SHENZHEN_2026-02-23.csv`
- `manuscript/tables/generated/hardshift_p15_dice_screen_shenzhen_2026_02_23.md`
- `manuscript/tables/generated/hardshift_p15_dice_screen_shenzhen_2026_02_23.tex`
- `manuscript/figures/generated/hardshift_p15_survivor_screen_panel.png`
- `manuscript/figures/generated/hardshift_p15_survivor_screen_panel.pdf`

Notes:
- This claim remains pending until the P1.5 audit CSV is present in the workspace.
- The scaffolded table/figure stubs already reserve manuscript slots and captions.

## C4: PROVEN

Baseline-comparison and journal-summary manuscript assets are automated, generated, and traceable to source audit CSVs.

- Rationale: journal_main_table=GENERATED, journal_main_figure=GENERATED, baseline_tables=(GENERATED,GENERATED), baseline_figure=GENERATED.

Evidence paths:
- `manuscript/tables/generated/journal_main_results_jsrt_to_montgomery_2026_02_23.tex`
- `manuscript/tables/generated/baseline_crossdataset_jsrt_to_montgomery_2026_02_23.tex`
- `manuscript/tables/generated/baseline_crossdataset_jsrt_to_shenzhen_2026_02_23.tex`
- `manuscript/figures/generated/journal_main_results_jsrt_to_montgomery_endpoint_panel.pdf`
- `manuscript/figures/generated/baseline_crossdataset_auroc_by_method.pdf`
- `configs/journal_table_export_specs_v1.json`
- `configs/journal_figure_export_specs_v1.json`

Notes:
- Future narrowing of the final manuscript comparison set should preserve the same source audit paths and scripted exports.

## C5: PROVEN

The journal asset pipeline (verify + tables + figures + claims + stats) is rerunnable and auditable from a single command.

- Rationale: asset_pipeline=IN_PROGRESS, table_pipeline=IN_PROGRESS counts={'GENERATED': 6, 'PENDING': 2, 'FAILED': 0, 'SKIPPED': 0}, figure_pipeline=IN_PROGRESS counts={'GENERATED': 5, 'PENDING': 2, 'FAILED': 0, 'SKIPPED': 0}, stats_pipeline=IN_PROGRESS counts={'GENERATED': 4, 'PENDING': 2, 'FAILED': 0, 'SKIPPED': 0}.

Evidence paths:
- `scripts/run_journal_asset_pipeline.py`
- `reports_v2/audits/JOURNAL_ASSET_PIPELINE_STATUS_2026-02-23.json`
- `reports_v2/audits/JOURNAL_ASSET_PIPELINE_STATUS_2026-02-23.md`
- `reports_v2/manifests/journal_asset_pipeline_2026-02-23.sha256`
- `reports_v2/manifests/journal_tables_generated_2026-02-23.sha256`
- `reports_v2/manifests/journal_figures_generated_2026-02-23.sha256`
- `scripts/export_journal_stats.py`
- `configs/journal_stats_export_specs_v1.json`
- `manuscript/stats/generated/STATS_EXPORT_INDEX_2026-02-23.json`
- `reports_v2/manifests/journal_stats_generated_2026-02-23.sha256`

Notes:
- IN_PROGRESS is expected pre-P1.5 completion because pending placeholders are intentionally preserved.

## C6: PENDING

Journal-level scientific claim (full validation on the selected survivor with ablations/robustness/failure analysis) is supported.

- Rationale: This requires P2/P3 outputs and final analysis artifacts; current work only establishes the foundation and automation.

Evidence paths:
- `manuscript/tables/generated/p2_survivor_results_template.tex`
- `manuscript/tables/generated/p3_ablation_results_template.tex`
- `manuscript/figures/generated/hardshift_p2_survivor_results_panel.pdf`
- `manuscript/figures/generated/hardshift_p3_ablation_summary.pdf`

Notes:
- Keep this explicitly pending to avoid overclaiming before P2/P3 completion.
