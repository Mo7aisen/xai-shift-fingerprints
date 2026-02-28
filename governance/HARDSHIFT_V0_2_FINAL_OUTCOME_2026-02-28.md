# Hardshift v0.2 Final Outcome

- Date: `2026-02-28`
- Scope: Shenzhen hardshift P1.5 / P2 candidate progression under v0.2 ranked governance
- Final result: `0/2 survivors`
- Scientific disposition: `honest negative result`

## Executive Conclusion

The v0.2 hardshift candidate path is exhausted. Both promoted candidates failed P2 Gate-3 validation under the predeclared thresholds, so there is no defensible basis to advance any v0.2 candidate to P3.

This is a negative result for the current fingerprint method in the Shenzhen hardshift setting, not a threshold artifact and not a procedural failure.

## Candidate Outcomes

### 1. Primary candidate: `shenzhen_to_shenzhen_pp_gamma085`

- P2 result: `FAIL`
- Gate-level status: `NO-GO / CONDITIONAL`
- Critical failure:
  - `predicted_mask` power: `0.2274`
  - `mask_free` power: `0.2663`
- Performance profile:
  - AUROC near chance (`0.5221`, `0.5233`)
  - FPR95 weak (`0.9329`, `0.9399`)

Reference: [P2_GAMMA085_GATE3_FAILURE_2026-02-23.md](./P2_GAMMA085_GATE3_FAILURE_2026-02-23.md)

### 2. Backup candidate: `shenzhen_to_shenzhen_pp_jpeg90`

- P2 result: `FAIL`
- Gate-level status: `NO-GO / CONDITIONAL`
- Critical failure:
  - `predicted_mask` power: `0.0400`
  - `mask_free` power: `0.0796`
- Performance profile:
  - AUROC near chance (`0.5044`, `0.5123`)
  - FPR95 weak (`0.9717`, `0.9611`)

Reference: [P2_JPEG90_GATE3_FAILURE_2026-02-28.md](./P2_JPEG90_GATE3_FAILURE_2026-02-28.md)

## Interpretation

The method currently shows:

- stable feature rankings across seeds (`Jaccard = 1.0`)
- narrow seed-level AUROC spread (`seed-CI width = 0.0000`)
- but weak discrimination and extremely low power in the Shenzhen hardshift setting

This combination indicates that the procedure is reproducibly weak rather than noisily promising. Repeating the same strategy on lower-ranked v0.2 candidates is not scientifically compelling under the current evidence.

## Governance Outcome

- v0.2 protocol is exhausted: `0/2 survivors`
- No P3 launch is authorized
- No threshold lowering is justified post hoc
- The correct record is an honest negative result for this hardshift branch

## Decision Options

### A. Halt P1.5 branch and document the negative result

- Scientifically strongest option
- Preserve integrity of the predeclared gate
- Keep the paper anchored on validated results rather than forcing a weak extension

### B. Expand to v0.3 (ranks 3-10)

- Possible, but weakly justified
- Lower-ranked candidates already have less favorable screening evidence
- High compute cost with low prior probability of rescue

### C. Redesign the fingerprint method

- Plausible if the goal is to salvage hardshift detection
- Requires a new methodological hypothesis, not just more compute
- Appropriate only as a fresh branch, not a continuation of v0.2

### D. Lower Gate-3 threshold

- Not recommended
- This would introduce post-hoc bias and weaken the validity of the claim

## Recommended Decision

Choose `A`: halt the current P1.5 hardshift promotion branch and document the negative result.

The data support an honest limitation statement more strongly than a continuation claim. If hardshift remains strategically important, any future work should proceed as a redesigned method branch (`C`), not as a retrospective rescue of v0.2.

## Manuscript Direction if Halted

- Use Phase 1 Extended as the primary validated result set
- Report P1.5 as attempted and unsuccessful under predeclared validation
- Add a limitations/discussion section stating that the current fingerprint method did not achieve adequate power in the Shenzhen hardshift setting
- Do not present P1.5 as a near-miss or hidden success
