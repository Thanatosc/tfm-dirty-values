# Dataset Manifest

Status: **Public release manifest for the frozen Kill Test and 11-record full panel**  
Distribution boundary: **raw OpenML records and processed feature matrices are not redistributed**

## Kill Test panel

These records were already validated by the source `tfm-missingness` pipeline under the same row, feature, class, missingness, and license constraints. Nine source split archives were migrated byte-identically into this project and re-signed under the new configuration. This avoids modifying the source cache after its Windows ACL rejected direct sklearn access.

| OpenML ID | Fixed record | Rows | Features | Categorical predictors | Native missing | License | Role |
|---:|---|---:|---:|---:|---:|---|---|
| 31 | credit-g v1 | 1,000 | 20 | 13 | 0 | Public | mixed-type anchor |
| 44 | spambase v1 | 4,601 | 57 | 0 | 0 | Public | numeric generalization |
| 1049 | pc4 v1 | 1,458 | 37 | 0 | 0 | Public | numeric generalization |

Splits are stratified 70/30 with seeds `1907`, `2027`, and `4049`. `credit-g` overlaps the closest missingness paper's dataset family; all three overlap the author's earlier missingness pipeline. Both facts must be disclosed. This Kill Test does not include C5 categorical corruption, so one mixed-type anchor is sufficient at this gate.

The local split manifest contains nine splits:

- Config SHA-256: `11b16b5771fe8d2496d5c71dfa70831dbcdad6fa84b82c90852ffc873edf87e7`
- Content manifest SHA-256: `489f565d6bdc6b216b6e2fe1940c9e43f4f21930bca9fea7b37b65a8b366682a`
- Manifest file SHA-256: `DDF21AF9507DDDF7B919175FA3CCE24D5AF3EC12E218914F8421E98ECC93F42F`
- Every local NPZ matches the SHA-256 of its frozen source NPZ.

## Full-study panel

The full panel is frozen in `data/full-panel.json` (SHA-256 `4AD3152CA6E3E5A32E2785DA7254919784FF780AC55690EE823632735368B729`). It contains 11 unique underlying records.

| OpenML ID | Record/version | Rows | Predictors | Categorical | Native missing | C5 |
|---:|---|---:|---:|---:|---:|:---:|
| 3 | kr-vs-kp v1 | 3,196 | 36 | 36 | 0 | yes |
| 24 | mushroom v1 | 8,124 | 22 | 22 | 0.0139 | yes |
| 31 | credit-g v1 | 1,000 | 20 | 13 | 0 | yes |
| 44 | spambase v1 | 4,601 | 57 | 0 | 0 | no |
| 802 | pbcseq v2 | 1,945 | 18 | 6 | 0.0324 | yes |
| 1049 | pc4 v1 | 1,458 | 37 | 0 | 0 | no |
| 1487 | ozone-level-8hr v1 | 2,534 | 72 | 0 | 0 | no |
| 1504 | steel-plates-fault v1 | 1,941 | 33 | 0 | 0 | no |
| 42193 | compas-two-years v4 | 5,278 | 13 | 6 | 0 | yes |
| 44158 | KDDCup09_upselling v4 | 5,032 | 45 | 11 | 0 | yes |
| 44160 | rl v4 | 4,970 | 12 | 7 | 0 | yes |

The seven-record C5 panel is OpenML `3, 24, 31, 802, 42193, 44158, 44160`. Each has at least three categorical predictors.

The full study uses five stratified 70/30 splits per record, with seeds `1907`, `2027`, `4049`, `6029`, and `8089`. The following artifacts were frozen before authorization:

- 55 dataset-seed splits with file-level hashes for all 55 NPZ arrays and 55 JSON metadata files.
- 275 clean tasks, 3,600 numeric-corruption tasks, and 700 categorical-corruption tasks.
- 4,575 unique tasks in `artifacts/phase2/full-study-task-manifest.json`.

Frozen Phase 2 identities:

- Config SHA-256: `F8786382B04E9658DAF928041F3815D01039F825633C019275314830F600E159`
- Split-manifest file SHA-256: `AD026F62A536B50A5BE6BCE4E4CEEE5B21A413BC5A55A88DB5AF9D4831D12F87`
- Split-content identity: `1e8226040d3e3cd09172809c915636e302c5ba20a36a7c57781d22020a9e8bdc`
- Split-file identity: `f76595077fd7e7fe4a08b70f91d5d3cc17075603edf9cc0cae17d864f5a8a855`
- Task-manifest identity: `b2bb293fedbb497534311b7bb353e10de2cc4d6a4525e30a292717fcf581cc8c`
- Task-manifest file SHA-256: `B27FC5415BDE5F5F54C8864FCD9E4158555ACBC60CCD37E90D38882E5ADF79DD`

## Citation verification

All records have an active/public cached status and `Public` license. The eight citation strings missing from the earlier cached OpenML metadata were refreshed on `2026-08-14` from official OpenML lineage and checked against Crossref or OpenAlex where a scholarly identifier was available. The verification record is `research/dataset-citation-verification.md`. Repository-only records remain tied to their authoritative source pages. The COMPAS record is a robustness benchmark only and cannot support policy or fairness conclusions.

No full-study model outcome may be accessed until the split manifest, task manifest, Phase 1 `GO` decision, validation report, panel hash, and model checkpoints all pass the Phase 2 preflight and are bound into a separate human authorization artifact.
