# Phase 2 Analysis Compatibility Amendment

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run-to-validate bridge
- Origin Date: 2026-08-17
- Verification Status: REGISTERED_BEFORE_ANALYSIS_RESULTS
- Version Label: phase2_analysis_compat_v1

## Trigger

The frozen `analyze-full` entry point passed the 4,575-task completeness gate and then
stopped with `KeyError: 'ece'`. No hypothesis record, condition summary, p-value, or
bootstrap interval was returned before this amendment was registered.

The experiment writer persisted the preregistered 10-bin equal-width expected
calibration error under the explicit key `ece_10_equal_width`. The frozen analysis reader
requested the generic alias `ece`. All 4,575 atomic task outputs contain
`ece_10_equal_width`; none contains `ece`, and none lacks the stored ECE value.

## Mechanical correction

For analysis reads only, map:

```text
ece = ece_10_equal_width
```

The compatibility adapter must not change the SQLite database, atomic task JSON files,
metric values, task panel, dataset/seed aggregation, hypotheses, eligibility panels,
sign-flip tests, Holm families, bootstrap count, bootstrap seed, or interpretation rules.
It must fail if both keys exist with different values or if neither key exists.

The original authorized execution surface remains byte-unchanged. The adapter is outside
the runtime-code manifest and first validates the original Phase 2 authorization before
reading results.

## Frozen evidence

- Completed outputs inspected: `4,575`
- Outputs with `ece_10_equal_width`: `4,575`
- Outputs with `ece`: `0`
- Outputs with neither key: `0`
- Completed SQLite SHA-256: `7B9ADA6E20D91529FECE380AA062C67617AB3E09224A4FD80BB95E1B3A701B52`
- Authorized runtime identity: `9325544b8adb82eaa4ddbc99c12b01364315edeaddf17e7fe68d7e41efe8700d`
- Compatibility adapter: `scripts/full_analysis_compat.py`
- Compatibility adapter SHA-256: `4141C84D7593551176D5510FF0C262AC70D84D5192C754870DE40CD2C4C90F65`

## Classification

This is a method-neutral schema compatibility amendment, not an outcome-dependent
analytic change. It restores the already frozen ECE definition and does not add, remove,
or select an analysis based on observed results.
