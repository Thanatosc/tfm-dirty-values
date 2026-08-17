# Frozen study identities

These identities refer to the formal private execution before publication-path
normalization.

| Artifact | SHA-256 |
|---|---|
| Phase 1 configuration | `11B16B5771FE8D2496D5C71DFA70831DBCDAD6FA84B82C90852FFC873EDF87E7` |
| Phase 1 task manifest identity | `7226D32BA2EEA35BB33CF6E3BE5E64A08B53B33B237252FA5433992FF55C5034` |
| Phase 2 configuration | `F8786382B04E9658DAF928041F3815D01039F825633C019275314830F600E159` |
| Phase 2 task manifest file | `B27FC5415BDE5F5F54C8864FCD9E4158555ACBC60CCD37E90D38882E5ADF79DD` |
| Phase 2 task manifest identity | `B2BB293FEDBB497534311B7BB353E10DE2CC4D6A4525E30A292717FCF581CC8C` |
| Phase 2 split manifest file | `AD026F62A536B50A5BE6BCE4E4CEEE5B21A413BC5A55A88DB5AF9D4831D12F87` |
| Authorized runtime code identity | `9325544B8ADB82EAA4DDBC99C12B01364315EDEADDF17E7FE68D7E41EFE8700D` |
| Persisted output collection identity | `1F3AFE5373A6B35F4FBA0D0CEB7E0900CFFD9B58B85FED00205AE7E9AAC08E8A` |
| Frozen aggregate analysis | `7D9389384762BC04BD352F0540F225D05D92B8F60B99D309232ADC0D9EBBFE30` |

The public SQLite files differ bytewise from the formal databases only because
machine-local paths and worker identifiers were sanitized. Task identities,
statuses, metrics, timings, attempts, and interruption counts are unchanged.
