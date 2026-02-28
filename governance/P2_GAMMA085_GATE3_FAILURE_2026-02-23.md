# P2 gamma085 Gate-3 Failure Record

- Date of decision: `2026-02-23`
- Candidate: `shenzhen_to_shenzhen_pp_gamma085`
- Evaluation job: Slurm `2740` (`xfp_gate3_reconstruct`)
- Status: `Candidate rejected`

## Outcome

`shenzhen_to_shenzhen_pp_gamma085` failed P2 Gate-3 validation and is rejected from further progression.

The job completed successfully at the infrastructure level (`COMPLETED`, `ExitCode=0:0`), but the statistical gate failed. Per [reports_v2/audits/GATE3_FULL_SEEDS_STATS_hardshift_p2_gamma085_2026-02-23.md](../reports_v2/audits/GATE3_FULL_SEEDS_STATS_hardshift_p2_gamma085_2026-02-23.md), the overall result was `NO-GO / CONDITIONAL`.

## Failure Evidence

### Required gate criterion failure

- Required threshold: `Post-hoc power mean > 0.80` for both endpoints
- Observed:
  - `predicted_mask`: `0.2274`
  - `mask_free`: `0.2663`
- Result: `FAIL` for both endpoints

### Discriminative weakness

- AUROC was near chance and not operationally acceptable:
  - `predicted_mask`: `0.5221`
  - `mask_free`: `0.5233`
- AUPR remained similarly weak:
  - `predicted_mask`: `0.5296`
  - `mask_free`: `0.5324`

### FPR95 weakness

- FPR95 remained unacceptably high:
  - `predicted_mask`: `0.9329`
  - `mask_free`: `0.9399`

### Non-failing checks that do not rescue the candidate

- AUROC seed-CI width gate: `PASS`
- Feature-stability Jaccard mean gate: `PASS`

These two passes do not override the failed power criterion and weak discrimination profile.

## Governance Disposition

- Do not modify `gamma085`
- Do not relax or reinterpret thresholds
- Do not launch P3 for `gamma085`
- Record this candidate as failed at P2 and remove it from the active promotion path

## Evidence Files

- [GATE3_FULL_SEEDS_STATS_hardshift_p2_gamma085_2026-02-23.md](../reports_v2/audits/GATE3_FULL_SEEDS_STATS_hardshift_p2_gamma085_2026-02-23.md)
- [GATE3_FULL_SEEDS_SUMMARY_hardshift_p2_gamma085_2026-02-23.json](../reports_v2/audits/GATE3_FULL_SEEDS_SUMMARY_hardshift_p2_gamma085_2026-02-23.json)
- [GATE3_FULL_SEEDS_STATS_hardshift_p2_gamma085_2026-02-23.csv](../reports_v2/audits/GATE3_FULL_SEEDS_STATS_hardshift_p2_gamma085_2026-02-23.csv)
- [GATE3_RECONSTRUCT_HASH_MATCH_hardshift_p2_gamma085_2026-02-23_2740.csv](../reports_v2/audits/GATE3_RECONSTRUCT_HASH_MATCH_hardshift_p2_gamma085_2026-02-23_2740.csv)
