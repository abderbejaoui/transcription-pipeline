# Correction Pipeline Evaluation — `baseline_hardened`

**Date:** 2026-06-01 23:04:35  
**Elapsed:** 79.2s  
**Records:** 216 (0 errors)  

---

## Summary

| Metric | Value |
|--------|-------|
| **Records evaluated** | 216 |
| **Mean WER (raw → gold)** | 0.3304 |
| **Mean WER (corrected → gold)** | 0.2913 |
| **WER reduction (Δ)** | **+0.0391** |
| **Correction precision** | 0.5426 (51/94) |
| **Correction recall** | 0.2867 (41/143) |
| **F1 score** | 0.3752 |
| **Do-no-harm rate (all records)** | 0.8750 (189/216) |
| **Do-no-harm rate (clean only)** | 0.7822 (79/101) |
| **Harmful changes (total)** | 30 (in 27 records) |
| **Missed errors — flagged for HITL** | 57 |
| **Missed errors — silently passed** | 45 |
| **Total flags (HITL)** | 147 |
| **Avg flags/record** | 0.68 |
| **Avg corrections applied/record** | 0.44 |

### WER by Language

| Language | N | WER (raw) | WER (corrected) | Δ |
|----------|---|-----------|-----------------|----|
| ar | 104 | 0.5207 | 0.5101 | +0.0107 |
| en | 97 | 0.1406 | 0.0682 | +0.0724 |
| mixed | 15 | 0.2390 | 0.2179 | +0.0211 |

### WER by Difficulty

| Difficulty | N | WER (raw) | WER (corrected) | Δ |
|------------|---|-----------|-----------------|----|
| easy | 122 | 0.1622 | 0.1845 | -0.0223 |
| hard | 33 | 0.5073 | 0.3886 | +0.1187 |
| medium | 61 | 0.5713 | 0.4524 | +0.1189 |

---

## Per-Record Details

| ID | Lang | Diff | Contains Error? | WER raw | WER corr | Δ | Changes | Flags | Do-no-harm |
|----|------|------|----------------|---------|----------|----|---------|-------|------------|
| eval_0000 | en | medium | Yes | 0.075 | 0.094 | -0.019 | 1 | 1 | ❌ |
| eval_0001 | en | medium | Yes | 0.093 | 0.116 | -0.023 | 1 | 2 | ❌ |
| eval_0002 | en | hard | Yes | 0.065 | 0.087 | -0.022 | 1 | 3 | ✅ |
| eval_0003 | en | medium | Yes | 0.067 | 0.067 | +0.000 | 0 | 1 | ✅ |
| eval_0004 | en | hard | Yes | 0.256 | 0.256 | +0.000 | 0 | 0 | ✅ |
| eval_0005 | en | medium | Yes | 0.098 | 0.073 | +0.024 | 1 | 1 | ✅ |
| eval_0006 | en | hard | Yes | 0.143 | 0.114 | +0.029 | 1 | 2 | ✅ |
| eval_0007 | en | medium | Yes | 0.098 | 0.098 | +0.000 | 0 | 1 | ✅ |
| eval_0008 | en | hard | Yes | 0.143 | 0.143 | +0.000 | 0 | 1 | ✅ |
| eval_0009 | en | medium | Yes | 0.118 | 0.118 | +0.000 | 0 | 2 | ✅ |
| eval_0010 | en | medium | Yes | 0.129 | 0.129 | +0.000 | 0 | 0 | ✅ |
| eval_0011 | en | medium | Yes | 0.125 | 0.125 | +0.000 | 0 | 2 | ✅ |
| eval_0012 | en | hard | Yes | 0.138 | 0.138 | +0.000 | 0 | 0 | ✅ |
| eval_0013 | en | hard | Yes | 0.125 | 0.125 | +0.000 | 0 | 1 | ✅ |
| eval_0014 | en | hard | Yes | 0.029 | 0.029 | +0.000 | 0 | 0 | ✅ |
| eval_0015 | en | hard | Yes | 0.094 | 0.094 | +0.000 | 0 | 0 | ✅ |
| eval_0016 | en | hard | Yes | 0.088 | 0.088 | +0.000 | 2 | 3 | ❌ |
| eval_0017 | en | medium | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0018 | en | medium | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0019 | en | hard | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0020 | en | hard | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0021 | en | hard | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0022 | ar | medium | Yes | 0.429 | 0.000 | +0.429 | 3 | 3 | ✅ |
| eval_0023 | en | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0024 | en | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0025 | en | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0026 | en | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0027 | ar | medium | Yes | 0.750 | 0.500 | +0.250 | 1 | 1 | ✅ |
| eval_0028 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0029 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 1 | 2 | ✅ |
| eval_0030 | en | medium | Yes | 0.250 | 0.250 | +0.000 | 0 | 1 | ✅ |
| eval_0031 | en | medium | Yes | 0.333 | 0.000 | +0.333 | 1 | 1 | ✅ |
| eval_0032 | en | medium | Yes | 0.667 | 0.333 | +0.333 | 1 | 1 | ✅ |
| eval_0033 | ar | medium | Yes | 1.000 | 0.333 | +0.667 | 1 | 2 | ✅ |
| eval_0034 | ar | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0035 | ar | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0036 | ar | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0037 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 1 | 1 | ✅ |
| eval_0038 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0039 | ar | medium | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0040 | ar | medium | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0041 | ar | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0042 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0043 | ar | medium | Yes | 0.667 | 0.333 | +0.333 | 1 | 1 | ✅ |
| eval_0044 | ar | medium | Yes | 1.000 | 0.500 | +0.500 | 1 | 1 | ✅ |
| eval_0045 | ar | medium | Yes | 1.000 | 0.667 | +0.333 | 1 | 1 | ✅ |
| eval_0046 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0047 | ar | hard | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0048 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0049 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0050 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0051 | ar | hard | Yes | 1.000 | 0.500 | +0.500 | 1 | 2 | ✅ |
| eval_0052 | ar | medium | Yes | 2.000 | 2.000 | +0.000 | 0 | 0 | ✅ |
| eval_0053 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0054 | ar | medium | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0055 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 1 | 1 | ✅ |
| eval_0056 | ar | hard | Yes | 1.000 | 0.333 | +0.667 | 1 | 2 | ✅ |
| eval_0057 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 1 | 2 | ✅ |
| eval_0058 | ar | hard | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0059 | ar | hard | Yes | 1.000 | 0.500 | +0.500 | 2 | 3 | ✅ |
| eval_0060 | ar | hard | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0061 | ar | hard | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0062 | ar | hard | Yes | 1.500 | 1.000 | +0.500 | 1 | 1 | ✅ |
| eval_0063 | ar | hard | Yes | 0.833 | 0.833 | +0.000 | 1 | 2 | ✅ |
| eval_0064 | ar | hard | Yes | 1.000 | 0.800 | +0.200 | 3 | 3 | ✅ |
| eval_0065 | ar | hard | Yes | 1.143 | 1.143 | +0.000 | 0 | 1 | ✅ |
| eval_0066 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0067 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0068 | ar | medium | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0069 | ar | hard | Yes | 1.000 | 0.500 | +0.500 | 1 | 2 | ✅ |
| eval_0070 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0071 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0072 | ar | medium | Yes | 0.500 | 0.500 | +0.000 | 0 | 1 | ✅ |
| eval_0073 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0074 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0075 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0076 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0077 | ar | easy | Yes | 0.333 | 0.333 | +0.000 | 0 | 0 | ✅ |
| eval_0078 | ar | medium | Yes | 0.500 | 0.500 | +0.000 | 1 | 1 | ✅ |
| eval_0079 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0080 | ar | easy | Yes | 0.500 | 0.500 | +0.000 | 0 | 1 | ✅ |
| eval_0081 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0082 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 1 | 1 | ✅ |
| eval_0083 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0084 | ar | easy | No | 0.000 | 0.500 | -0.500 | 1 | 1 | ❌ |
| eval_0085 | ar | medium | Yes | 0.500 | 2.000 | -1.500 | 1 | 1 | ✅ |
| eval_0086 | ar | medium | Yes | 0.500 | 0.500 | +0.000 | 1 | 1 | ✅ |
| eval_0087 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0088 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0089 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0090 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0091 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0092 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0093 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0094 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0095 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0096 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0097 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0098 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0099 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 1 | ✅ |
| eval_0100 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0101 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0102 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0103 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0104 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0105 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 1 | ✅ |
| eval_0106 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0107 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0108 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0109 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0110 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0111 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0112 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0113 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0114 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 1 | ✅ |
| eval_0115 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0116 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0117 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0118 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0119 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0120 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0121 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0122 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0123 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0124 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0125 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0126 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0127 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0128 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0129 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0130 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0131 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0132 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0133 | ar | easy | No | 0.000 | 1.667 | -1.667 | 2 | 2 | ❌ |
| eval_0134 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0135 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0136 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0137 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0138 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0139 | ar | easy | No | 0.000 | 0.143 | -0.143 | 1 | 2 | ❌ |
| eval_0140 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0141 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0142 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0143 | ar | easy | No | 0.000 | 0.200 | -0.200 | 1 | 1 | ❌ |
| eval_0144 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0145 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0146 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 2 | ❌ |
| eval_0147 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 1 | ✅ |
| eval_0148 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0149 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0150 | ar | easy | No | 0.000 | 0.200 | -0.200 | 1 | 1 | ❌ |
| eval_0151 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0152 | ar | easy | No | 0.000 | 0.143 | -0.143 | 1 | 1 | ❌ |
| eval_0153 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0154 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 1 | ❌ |
| eval_0155 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0156 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0157 | ar | easy | No | 0.000 | 0.667 | -0.667 | 1 | 2 | ❌ |
| eval_0158 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0159 | ar | easy | No | 0.000 | 1.000 | -1.000 | 1 | 1 | ❌ |
| eval_0160 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0161 | ar | easy | No | 0.000 | 0.600 | -0.600 | 1 | 1 | ❌ |
| eval_0162 | ar | easy | No | 0.000 | 0.600 | -0.600 | 2 | 3 | ❌ |
| eval_0163 | ar | easy | No | 0.000 | 0.600 | -0.600 | 1 | 1 | ❌ |
| eval_0164 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0165 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0166 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0167 | ar | easy | No | 0.000 | 0.500 | -0.500 | 1 | 1 | ❌ |
| eval_0168 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0169 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 1 | ❌ |
| eval_0170 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0171 | ar | easy | No | 0.000 | 0.250 | -0.250 | 1 | 1 | ❌ |
| eval_0172 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0173 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0174 | ar | easy | No | 0.000 | 1.500 | -1.500 | 1 | 1 | ❌ |
| eval_0175 | ar | easy | No | 0.000 | 0.750 | -0.750 | 1 | 1 | ❌ |
| eval_0176 | ar | easy | No | 0.000 | 0.200 | -0.200 | 1 | 1 | ❌ |
| eval_0177 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0178 | ar | easy | No | 0.000 | 0.250 | -0.250 | 1 | 1 | ❌ |
| eval_0179 | mixed | hard | Yes | 0.400 | 0.000 | +0.400 | 2 | 2 | ✅ |
| eval_0180 | mixed | medium | Yes | 0.250 | 0.250 | +0.000 | 0 | 1 | ✅ |
| eval_0181 | mixed | easy | No | 0.000 | 0.833 | -0.833 | 2 | 2 | ❌ |
| eval_0182 | mixed | hard | Yes | 0.286 | 0.286 | +0.000 | 0 | 1 | ✅ |
| eval_0183 | mixed | medium | Yes | 0.250 | 0.500 | -0.250 | 2 | 2 | ❌ |
| eval_0184 | mixed | medium | Yes | 0.286 | 0.143 | +0.143 | 1 | 1 | ✅ |
| eval_0185 | mixed | hard | Yes | 0.200 | 0.200 | +0.000 | 2 | 2 | ❌ |
| eval_0186 | mixed | medium | Yes | 0.200 | 0.200 | +0.000 | 0 | 0 | ✅ |
| eval_0187 | mixed | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0188 | mixed | hard | Yes | 0.286 | 0.143 | +0.143 | 1 | 1 | ✅ |
| eval_0189 | mixed | medium | Yes | 0.250 | 0.000 | +0.250 | 1 | 1 | ✅ |
| eval_0190 | mixed | medium | Yes | 0.250 | 0.000 | +0.250 | 1 | 1 | ✅ |
| eval_0191 | mixed | hard | Yes | 0.500 | 0.000 | +0.500 | 2 | 2 | ✅ |
| eval_0192 | mixed | easy | No | 0.000 | 0.286 | -0.286 | 1 | 1 | ❌ |
| eval_0193 | mixed | hard | Yes | 0.429 | 0.429 | +0.000 | 0 | 2 | ✅ |
| eval_0194 | en | medium | Yes | 0.167 | 0.167 | +0.000 | 0 | 1 | ✅ |
| eval_0195 | en | easy | Yes | 0.250 | 0.000 | +0.250 | 1 | 1 | ✅ |
| eval_0196 | en | medium | Yes | 0.200 | 0.200 | +0.000 | 0 | 1 | ✅ |
| eval_0197 | en | medium | Yes | 0.400 | 0.200 | +0.200 | 1 | 2 | ✅ |
| eval_0198 | en | medium | Yes | 0.200 | 0.000 | +0.200 | 1 | 1 | ✅ |
| eval_0199 | en | medium | Yes | 0.400 | 0.200 | +0.200 | 1 | 1 | ✅ |
| eval_0200 | en | easy | Yes | 0.333 | 0.000 | +0.333 | 1 | 1 | ✅ |
| eval_0201 | en | medium | Yes | 0.333 | 0.333 | +0.000 | 0 | 0 | ✅ |
| eval_0202 | en | medium | Yes | 0.400 | 0.000 | +0.400 | 2 | 2 | ✅ |
| eval_0203 | en | easy | Yes | 0.250 | 0.250 | +0.000 | 0 | 0 | ✅ |
| eval_0204 | en | easy | Yes | 0.333 | 0.333 | +0.000 | 0 | 0 | ✅ |
| eval_0205 | en | medium | Yes | 0.333 | 0.333 | +0.000 | 0 | 0 | ✅ |
| eval_0206 | en | easy | Yes | 0.333 | 0.000 | +0.333 | 1 | 1 | ✅ |
| eval_0207 | en | medium | Yes | 0.333 | 0.333 | +0.000 | 0 | 0 | ✅ |
| eval_0208 | en | hard | Yes | 0.250 | 0.250 | +0.000 | 0 | 0 | ✅ |
| eval_0209 | en | hard | Yes | 0.500 | 0.500 | +0.000 | 0 | 0 | ✅ |
| eval_0210 | en | medium | Yes | 0.250 | 0.250 | +0.000 | 0 | 0 | ✅ |
| eval_0211 | en | medium | Yes | 0.250 | 0.250 | +0.000 | 0 | 0 | ✅ |
| eval_0212 | en | medium | Yes | 0.200 | 0.000 | +0.200 | 1 | 1 | ✅ |
| eval_0213 | en | easy | Yes | 0.250 | 0.000 | +0.250 | 1 | 1 | ✅ |
| eval_0214 | en | easy | Yes | 0.200 | 0.200 | +0.000 | 0 | 0 | ✅ |
| eval_0215 | en | hard | Yes | 0.333 | 0.333 | +0.000 | 0 | 0 | ✅ |

---

## Failure Cases

**27 records with harmful changes (a token gold says should stay was changed):**

- **eval_0000** (had-error, en): 'breathing'→'breath'
- **eval_0001** (had-error, en): 'hypotension'→'hypertension'
- **eval_0016** (had-error, en): 'labs'→'results'
- **eval_0084** (clean, ar): 'تعبان'→'tuberculosis'
- **eval_0133** (clean, ar): 'السلام'→'systemic lupus erythematosus', 'دكتور'→'valproic acid'
- **eval_0139** (clean, ar): 'نتابع'→'tuberculosis'
- **eval_0143** (clean, ar): 'ضمن'→'diabetes'
- **eval_0146** (clean, ar): 'سيتم متابعة'→'asthma myopathy'
- **eval_0150** (clean, ar): 'سيتم'→'asthma'
- **eval_0152** (clean, ar): 'الطبيب'→'tuberculosis'
- **eval_0154** (clean, ar): 'منخفض'→'covid-19'
- **eval_0157** (clean, ar): 'نسبة الأكسجين'→'eplerenone edoxaban'
- **eval_0159** (clean, ar): 'العملية'→'acute myeloid leukemia'
- **eval_0161** (clean, ar): 'العملية'→'acute myeloid leukemia'
- **eval_0162** (clean, ar): 'التحاليل'→'sotalol', 'ضمن المعدل'→'diabetes methylprednisolone'
- **eval_0163** (clean, ar): 'الكامل'→'acute myeloid leukemia'
- **eval_0167** (clean, ar): 'منتظم'→'enzymes'
- **eval_0169** (clean, ar): 'مرتين'→'azathioprine'
- **eval_0171** (clean, ar): 'الالتزام'→'diltiazem'
- **eval_0174** (clean, ar): 'سليم'→'systemic lupus erythematosus'
- **eval_0175** (clean, ar): 'الكبد'→'inflammatory bowel disease'
- **eval_0176** (clean, ar): 'المفاصل'→'myofascial'
- **eval_0178** (clean, ar): 'الفحوصات'→'bevacizumab'
- **eval_0181** (clean, mixed): 'السلام'→'systemic lupus erythematosus', 'دكتور'→'valproic acid'
- **eval_0183** (had-error, mixed): 'أظهر'→'heart rate'
- **eval_0185** (had-error, mixed): 'ريزلتس'→'results'
- **eval_0192** (clean, mixed): 'أظهر'→'heart rate'

**87 records with missed corrections:**

- **eval_0030** (lang=en, diff=medium)
  - Gold: 'clopidogr 75 mg daily'→'clopidogrel 75 mg daily'
  - Changes applied: (none)
- **eval_0079** (lang=ar, diff=easy)
  - Gold: 'اضظراب'→'اضطراب'
  - Changes applied: (none)
- **eval_0045** (lang=ar, diff=medium)
  - Gold: 'هستوري مرض'→'history of disease'
  - Changes applied: 'هستوري'→'history'
- **eval_0073** (lang=ar, diff=easy)
  - Gold: 'طعم'→'تعب'
  - Changes applied: (none)
- **eval_0052** (lang=ar, diff=medium)
  - Gold: 'دزي نس'→'dizziness'
  - Changes applied: (none)
- **eval_0209** (lang=en, diff=hard)
  - Gold: 'myokardial'→'myocardial', 'infarcton'→'infarction'
  - Changes applied: (none)
- **eval_0066** (lang=ar, diff=medium)
  - Gold: 'الفيتل ساينز'→'vital signs'
  - Changes applied: (none)
- **eval_0013** (lang=en, diff=hard)
  - Gold: 'giant cell arteritus'→'giant cell arteritis', 'predni and sone'→'prednisone'
  - Changes applied: (none)
- **eval_0005** (lang=en, diff=medium)
  - Gold: 'pyelonefritis'→'pyelonephritis', 'cipro and floxacin'→'ciprofloxacin'
  - Changes applied: 'pyelonefritis'→'pyelonephritis'
- **eval_0204** (lang=en, diff=easy)
  - Gold: 'anemis'→'anemia'
  - Changes applied: (none)

---

## Configuration

- **Pipeline:** live deterministic path — Stage 1 MedicalCorrector (`_build_corrector`, accept=88) + Stage 2 HybridMatcher (≥80 auto-apply)
- **Eval set:** `eval/correction_eval.jsonl`
- **LLM stages:** disabled (deterministic measurement)
- **Do-no-harm:** span-level, measured on all records