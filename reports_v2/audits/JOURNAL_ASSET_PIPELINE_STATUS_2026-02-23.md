# Journal Asset Pipeline Status

- Date tag: `2026-02-23`
- Overall status: `IN_PROGRESS`
- Strict mode: `False`

## Step Summary

| Step | Status | Return Code | Output JSON |
|---|---|---:|---|
| verify_hardshift_phase1_journal_track | WARN | 0 | reports_v2/audits/HARDSHIFT_JOURNAL_TRACK_VERIFY_2026-02-23.json |
| export_journal_tables | IN_PROGRESS | 0 | manuscript/tables/generated/TABLE_EXPORT_INDEX_2026-02-23.json |
| export_journal_figures | IN_PROGRESS | 0 | manuscript/figures/generated/FIGURE_EXPORT_INDEX_2026-02-23.json |
| export_journal_stats | IN_PROGRESS | 0 | manuscript/stats/generated/STATS_EXPORT_INDEX_2026-02-23.json |
| build_journal_claim_evidence_map | IN_PROGRESS | 0 | manuscript/claims/JOURNAL_CLAIM_EVIDENCE_MAP_2026-02-23.json |
| build_journal_submission_skeleton | PASS | 0 | manuscript/journal_submission/auto/SKELETON_INDEX_2026-02-23.json |

## Subreports

- Verify: `WARN` (reports_v2/audits/HARDSHIFT_JOURNAL_TRACK_VERIFY_2026-02-23.json)
- Tables: `IN_PROGRESS` (manuscript/tables/generated/TABLE_EXPORT_INDEX_2026-02-23.json)
- Figures: `IN_PROGRESS` (manuscript/figures/generated/FIGURE_EXPORT_INDEX_2026-02-23.json)
- Claims map: `IN_PROGRESS` (manuscript/claims/JOURNAL_CLAIM_EVIDENCE_MAP_2026-02-23.json)
- Stats: `IN_PROGRESS` (manuscript/stats/generated/STATS_EXPORT_INDEX_2026-02-23.json)
- Skeleton: `PASS` (manuscript/journal_submission/auto/SKELETON_INDEX_2026-02-23.json)

## Logs (Truncated)

### verify_hardshift_phase1_journal_track

`Command`

```bash
/home/ubuntu/xai-env/bin/python scripts/verify_hardshift_phase1_journal_track.py --date-tag 2026-02-23
```

`stdout`

```text
Wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/HARDSHIFT_JOURNAL_TRACK_VERIFY_2026-02-23.json
Wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/audits/HARDSHIFT_JOURNAL_TRACK_VERIFY_2026-02-23.md
Wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/manifests/hardshift_journal_track_phase1_2026-02-23.sha256
Overall status: WARN
```

### export_journal_tables

`Command`

```bash
/home/ubuntu/xai-env/bin/python scripts/export_journal_tables.py --date-tag 2026-02-23
```

`stdout`

```text
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/tables/generated/TABLE_EXPORT_INDEX_2026-02-23.json
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/tables/generated/TABLE_EXPORT_INDEX_2026-02-23.md
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/manifests/journal_tables_generated_2026-02-23.sha256
[INFO] overall_status=IN_PROGRESS generated=6 pending=2 failed=0
```

### export_journal_figures

`Command`

```bash
/home/ubuntu/xai-env/bin/python scripts/export_journal_figures.py --date-tag 2026-02-23
```

`stdout`

```text
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/figures/generated/FIGURE_EXPORT_INDEX_2026-02-23.json
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/figures/generated/FIGURE_EXPORT_INDEX_2026-02-23.md
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/manifests/journal_figures_generated_2026-02-23.sha256
[INFO] overall_status=IN_PROGRESS generated=5 pending=2 failed=0
```

### export_journal_stats

`Command`

```bash
/home/ubuntu/xai-env/bin/python scripts/export_journal_stats.py --date-tag 2026-02-23
```

`stdout`

```text
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/stats/generated/STATS_EXPORT_INDEX_2026-02-23.json
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/stats/generated/STATS_EXPORT_INDEX_2026-02-23.md
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/manifests/journal_stats_generated_2026-02-23.sha256
[INFO] overall_status=IN_PROGRESS generated=4 pending=2 failed=0
```

### build_journal_claim_evidence_map

`Command`

```bash
/home/ubuntu/xai-env/bin/python scripts/build_journal_claim_evidence_map.py --date-tag 2026-02-23
```

`stdout`

```text
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/claims/JOURNAL_CLAIM_EVIDENCE_MAP_2026-02-23.json
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/claims/JOURNAL_CLAIM_EVIDENCE_MAP_2026-02-23.md
[INFO] overall_status=IN_PROGRESS counts={'PROVEN': 4, 'PARTIAL': 1, 'PENDING': 1, 'BLOCKED': 0}
```

### build_journal_submission_skeleton

`Command`

```bash
/home/ubuntu/xai-env/bin/python scripts/build_journal_submission_skeleton.py --date-tag 2026-02-23
```

`stdout`

```text
[INFO] wrote skeleton in /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/journal_submission
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/journal_submission/auto/SKELETON_INDEX_2026-02-23.json
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/manuscript/journal_submission/auto/SKELETON_INDEX_2026-02-23.md
[INFO] wrote /home/ubuntu/xai_shift_fingerprints_reproduction_20251225_full/reports_v2/manifests/journal_submission_skeleton_2026-02-23.sha256
[INFO] manifest_sha256=91365d0b48cb14af6eb9e24985a501e8cd591fb08ce811e606f6094494e917fc
[INFO] asset_pipeline_status=IN_PROGRESS
```

## Manifest

- Manifest: `reports_v2/manifests/journal_asset_pipeline_2026-02-23.sha256`
- Manifest SHA256: `5b7ca9d4591c265b5ee22ab1ca30dec612ab709de4d5214983a7a2ce35ee131a`
- Files hashed: `20`

## Notes

- `IN_PROGRESS` is expected while P1.5 outputs are still missing or unsynced.
- Re-run this orchestration script after P1.5/P2/P3 outputs are available to refresh all audits/tables/figures/stats/claims/skeleton in one command.
