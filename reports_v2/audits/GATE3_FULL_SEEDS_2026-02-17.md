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

- CI width threshold for AUROC (`< 0.06`): `PENDING` (not computed by constrained launcher)
- Feature stability Jaccard threshold (`> 0.75`): `PENDING`
- Power analysis threshold (`> 0.8`): `PENDING`

## Final Gate 3 Status

`CONDITIONAL PASS`

Interpretation:
- Engineering reliability + reproducibility objectives for Gate 3 are satisfied.
- Statistical Gate 3 acceptance remains pending explicit AUROC/CI/stability/power computation on the 5-seed outputs.
