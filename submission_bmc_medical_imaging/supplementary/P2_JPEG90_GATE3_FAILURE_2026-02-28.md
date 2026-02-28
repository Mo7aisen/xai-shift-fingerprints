# P2 jpeg90 Gate-3 Failure Record

- Date of decision: `2026-02-28`
- Candidate: `shenzhen_to_shenzhen_pp_jpeg90`
- Evaluation job: Slurm `2798` (`xfp_gate3_reconstruct`)
- Status: `Candidate rejected`
- Governance state after decision: `v0.2 protocol exhausted`

## Outcome

`shenzhen_to_shenzhen_pp_jpeg90` failed P2 Gate-3 validation and is rejected from further progression.

The job completed normally at the infrastructure level (`COMPLETED`, `ExitCode=0:0`, wall time `00:58:11`), but the statistical gate failed. Per [GATE3_FULL_SEEDS_STATS_hardshift_p2_jpeg90_2026-02-23.md](../reports_v2/audits/GATE3_FULL_SEEDS_STATS_hardshift_p2_jpeg90_2026-02-23.md), the overall result was `NO-GO / CONDITIONAL`.

## Failure Evidence

### Required gate criterion failure

- Required threshold: `Post-hoc power mean > 0.80` for both endpoints
- Observed:
  - `predicted_mask`: `0.0400`
  - `mask_free`: `0.0796`
- Result: `FAIL` for both endpoints

### Near-chance performance

- AUROC remained effectively near chance:
  - `predicted_mask`: `0.5044`
  - `mask_free`: `0.5123`
- AUPR remained weak:
  - `predicted_mask`: `0.5089`
  - `mask_free`: `0.5178`

### FPR95 weakness

- FPR95 remained extremely high:
  - `predicted_mask`: `0.9717`
  - `mask_free`: `0.9611`

### Non-failing checks that do not rescue the candidate

- AUROC seed-CI width gate: `PASS`
- Feature-stability Jaccard mean gate: `PASS`

These checks do not override the failed power criterion or the near-chance discrimination profile.

## Governance Disposition

- Do not launch P3 for `jpeg90`
- Close `jpeg90` as a failed P2 candidate
- Record v0.2 candidate list as exhausted (`gamma085` failed, `jpeg90` failed)
- Treat this as a negative result in the Shenzhen hardshift setting under the current method and gate definition

## Evidence Files

- [GATE3_FULL_SEEDS_STATS_hardshift_p2_jpeg90_2026-02-23.md](../reports_v2/audits/GATE3_FULL_SEEDS_STATS_hardshift_p2_jpeg90_2026-02-23.md)
- [GATE3_FULL_SEEDS_SUMMARY_hardshift_p2_jpeg90_2026-02-23.json](../reports_v2/audits/GATE3_FULL_SEEDS_SUMMARY_hardshift_p2_jpeg90_2026-02-23.json)
- [GATE3_FULL_SEEDS_STATS_hardshift_p2_jpeg90_2026-02-23.csv](../reports_v2/audits/GATE3_FULL_SEEDS_STATS_hardshift_p2_jpeg90_2026-02-23.csv)
- [GATE3_RECONSTRUCT_HASH_MATCH_hardshift_p2_jpeg90_2026-02-23_2798.csv](../reports_v2/audits/GATE3_RECONSTRUCT_HASH_MATCH_hardshift_p2_jpeg90_2026-02-23_2798.csv)
