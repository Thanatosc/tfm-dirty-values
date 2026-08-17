# Model artifacts

Model checkpoints are intentionally not redistributed. Obtain them from the
model publishers, place them at the repository-relative paths below, and verify
their SHA-256 values before running an experiment.

| Model | Official source | Required path | SHA-256 |
|---|---|---|---|
| TabPFN v2 classifier | https://huggingface.co/Prior-Labs/TabPFN-v2-clf | `models/tabpfn-v2/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` | `CF8C519C01EAF1613EE91239006D57B1C806FF5F23AC1AEB1315BA1015210E49` |
| TabICL v2 classifier | https://huggingface.co/jingang/TabICL | `models/tabiclv2/tabicl-classifier-v2-20260212.ckpt` | `BDC7DBD5E4FF21F8F0456FCF90C6B7CDF72DBEA960F2D05B19BEC19F9B3D4ED0` |

PowerShell verification:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath models\tabpfn-v2\tabpfn-v2-classifier-finetuned-zk73skhh.ckpt
Get-FileHash -Algorithm SHA256 -LiteralPath models\tabiclv2\tabicl-classifier-v2-20260212.ckpt
```
