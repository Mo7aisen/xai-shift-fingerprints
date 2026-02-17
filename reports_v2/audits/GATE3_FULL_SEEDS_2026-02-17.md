# Gate 3 Full Seeds Audit (Official Batch)

Date: 2026-02-17
Batch tag: `gate3_official_20260217`
Experiment: `jsrt_to_montgomery`
Subset: `full`
Seeds: `[42, 43, 44, 45, 46]`

## Traceability Lock

- Commit hash (in every run log/registry row): `10aed312b644`
- Snapshot hash (in every run log/registry row): `13207d48d107223aa8f23ad36e0bb702688a6224b42a048964a32303705e11d4`
- Registry scope marker: `batch_tag=gate3_official_20260217`

Status: `PASS`

## Official Jobs

| Seed | Job ID | State | Elapsed | Start (UTC) | End (UTC) | Peak VRAM (MiB) | Post-run tests |
|---|---:|---|---|---|---|---:|---|
| 42 | 2651 | COMPLETED | 00:05:17 | 2026-02-17T12:22:42 | 2026-02-17T12:27:59 | 3618 | 11 passed |
| 43 | 2652 | COMPLETED | 00:05:15 | 2026-02-17T12:27:59 | 2026-02-17T12:33:14 | 3618 | 11 passed |
| 44 | 2653 | COMPLETED | 00:05:15 | 2026-02-17T12:33:14 | 2026-02-17T12:38:29 | 3624 | 11 passed |
| 45 | 2654 | COMPLETED | 00:05:16 | 2026-02-17T12:38:29 | 2026-02-17T12:43:45 | 3618 | 11 passed |
| 46 | 2655 | COMPLETED | 00:05:12 | 2026-02-17T12:43:46 | 2026-02-17T12:48:58 | 3618 | 11 passed |

## Aggregate Stability

- Seed coverage (`42..46`): `PASS`
- Completion rate: `5/5` (`PASS`)
- Runtime mean: `00:05:15`
- Runtime range: `00:05:12` to `00:05:17`
- Peak VRAM mean: `3619.2 MiB`
- Peak VRAM range: `3618` to `3624 MiB`
- Endpoint execution: `predicted_mask + mask_free` completed in all runs (`PASS`)

## Protocol Gate 3 Decision

### Gate 3 infrastructure/reproducibility criteria

- 5-seed official batch executed: `PASS`
- Patient-level + leakage guards remained active (post-run test suite): `PASS`
- Traceability (commit + snapshot hash in logs/registry): `PASS`

### Gate 3 statistical criteria

- Statistical analysis artifact: `reports_v2/audits/GATE3_FULL_SEEDS_STATS_2026-02-17.md`
- Per-seed metrics table: `reports_v2/audits/GATE3_FULL_SEEDS_STATS.csv`
- Summary JSON: `reports_v2/audits/GATE3_FULL_SEEDS_SUMMARY.json`

Endpoint `predicted_mask`:
- AUROC mean: `0.8964`
- AUROC seed-CI width: `0.0000` (`PASS`, threshold `< 0.06`)
- Feature stability Jaccard mean: `1.0000` (`PASS`, threshold `> 0.75`)
- Power mean: `1.0000` (`PASS`, threshold `> 0.80`)

Endpoint `mask_free`:
- AUROC mean: `0.8560`
- AUROC seed-CI width: `0.0000` (`PASS`, threshold `< 0.06`)
- Feature stability Jaccard mean: `1.0000` (`PASS`, threshold `> 0.75`)
- Power mean: `1.0000` (`PASS`, threshold `> 0.80`)

Statistical gate result: `PASS`

### Bitwise Replay Check (Strict Reproducibility Note)

- Replay hash table: `reports_v2/audits/GATE3_RECONSTRUCT_HASH_MATCH_2659.csv`
- Official-vs-replay output hashes matched: `0/10` (`FAIL`)
- Interpretation: pipeline is not bitwise deterministic across reruns (same seed/config), likely due GPU-level nondeterminism.
- Impact: statistical Gate-3 acceptance remains valid; however, byte-level reproducibility hardening is required before final publication package freeze.

## Final Gate 3 Status

`FULL PASS (with reproducibility warning)`

Interpretation:
- Engineering and statistical Gate-3 criteria are satisfied (`FULL PASS`).
- A strict byte-level replay mismatch was observed and is now a tracked risk for Gate-4/packaging hardening.
