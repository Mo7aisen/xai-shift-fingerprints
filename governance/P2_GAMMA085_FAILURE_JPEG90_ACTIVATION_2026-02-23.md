# P2 gamma085 Failure and jpeg90 Activation Memo

- Date of decision: `2026-02-23`
- Primary candidate failed: `shenzhen_to_shenzhen_pp_gamma085`
- Backup candidate activated: `shenzhen_to_shenzhen_pp_jpeg90`
- Governance mode: `sequential_primary_first`

## Decision

The primary P2 candidate, `shenzhen_to_shenzhen_pp_gamma085`, failed Gate-3 validation and is rejected. The backup candidate, `shenzhen_to_shenzhen_pp_jpeg90` (v0.2 rank-2), is activated as the next sequential P2 candidate under the pre-locked governance rule.

This activation preserves the v0.2 selection order and does not alter thresholds, criteria, or the failure interpretation of `gamma085`.

## Failure Evidence Summary (gamma085)

- Gate-level result: `NO-GO / CONDITIONAL`
- Required power threshold: `> 0.80`
- Observed power:
  - `predicted_mask`: `0.2274`
  - `mask_free`: `0.2663`
- Near-chance AUROC:
  - `predicted_mask`: `0.5221`
  - `mask_free`: `0.5233`
- FPR95 weakness:
  - `predicted_mask`: `0.9329`
  - `mask_free`: `0.9399`

Reference failure record: [P2_GAMMA085_GATE3_FAILURE_2026-02-23.md](./P2_GAMMA085_GATE3_FAILURE_2026-02-23.md)

## Backup Activation Rationale (jpeg90)

- `jpeg90` is the designated backup candidate in the P2 governance plan
- `jpeg90` is the v0.2 rank-2 candidate already frozen for this contingency
- Metadata and pilot artifacts were already prepared before activation
- Sequential backup activation is the intended path after a primary P2 fail

Supporting references:

- [P2_PASS_FAIL_CHECKLIST_2026-02-23.md](./P2_PASS_FAIL_CHECKLIST_2026-02-23.md)
- [HARDSHIFT_P2_P3_VALIDATION_GOVERNANCE_NOTE_2026-02-23.md](./HARDSHIFT_P2_P3_VALIDATION_GOVERNANCE_NOTE_2026-02-23.md)
- [HARDSHIFT_P2_GAMMA085_PRELAUNCH_VERIFICATION_2026-02-23.md](../reports_v2/audits/HARDSHIFT_P2_GAMMA085_PRELAUNCH_VERIFICATION_2026-02-23.md)
- [HARDSHIFT_PILOT_METADATA_PREP_2026-02-23.md](../reports_v2/audits/HARDSHIFT_PILOT_METADATA_PREP_2026-02-23.md)

## Launch Package for jpeg90

- Launcher: `scripts/launch_hardshift_p2_jpeg90_sequential.sh`
- Config: `configs/hardshift_p2_jpeg90.yaml`
- Execution model: sequential P2 only
- Scope: launch `jpeg90` P2 only; no P3 is authorized by this memo

## Next Steps

1. Keep `gamma085` closed as a failed candidate.
2. Use the prepared `jpeg90` launch package for the next P2 run.
3. Execute `jpeg90` under the same Gate-3 style multi-seed protocol.
4. Make a new P2 decision only after `jpeg90` outputs are complete and reviewed.

## Constraints Reaffirmed

- Do not modify `gamma085`
- Do not change thresholds
- Do not run P3 on `gamma085`
- Do not interpret `gamma085` as a conditional pass
