# Correction Pipeline Evaluation — `phase2_v2`

**Date:** 2026-06-01 20:27:22  
**Elapsed:** 23.5s  
**Records:** 216 (0 errors)  

---

## Summary

| Metric | Value |
|--------|-------|
| **Records evaluated** | 216 |
| **Mean WER (raw → gold)** | 0.3304 |
| **Mean WER (corrected → gold)** | 0.2444 |
| **WER reduction (Δ)** | **+0.0860** |
| **Correction precision** | 0.1293 (38/294) |
| **Correction recall** | 0.3986 (57/143) |
| **F1 score** | 0.1952 |
| **Do-no-harm rate** | 0.8515 (86/101) |
| **Total flags (HITL)** | 121 |
| **Avg flags/record** | 0.56 |
| **Avg corrections applied/record** | 1.36 |

### WER by Language

| Language | N | WER (raw) | WER (corrected) | Δ |
|----------|---|-----------|-----------------|----|
| ar | 104 | 0.5207 | 0.3994 | +0.1214 |
| en | 97 | 0.1406 | 0.0941 | +0.0465 |
| mixed | 15 | 0.2390 | 0.1419 | +0.0971 |

### WER by Difficulty

| Difficulty | N | WER (raw) | WER (corrected) | Δ |
|------------|---|-----------|-----------------|----|
| easy | 122 | 0.1622 | 0.1312 | +0.0309 |
| hard | 33 | 0.5073 | 0.3263 | +0.1810 |
| medium | 61 | 0.5713 | 0.4264 | +0.1449 |

---

## Per-Record Details

| ID | Lang | Diff | Contains Error? | WER raw | WER corr | Δ | Changes | Flags | Do-no-harm |
|----|------|------|----------------|---------|----------|----|---------|-------|------------|
| eval_0000 | en | medium | Yes | 0.075 | 0.151 | -0.075 | 15 | 4 | ✅ |
| eval_0001 | en | medium | Yes | 0.093 | 0.116 | -0.023 | 1 | 2 | ✅ |
| eval_0002 | en | hard | Yes | 0.065 | 0.043 | +0.022 | 1 | 1 | ✅ |
| eval_0003 | en | medium | Yes | 0.067 | 0.044 | +0.022 | 35 | 1 | ✅ |
| eval_0004 | en | hard | Yes | 0.256 | 0.256 | +0.000 | 0 | 0 | ✅ |
| eval_0005 | en | medium | Yes | 0.098 | 0.171 | -0.073 | 12 | 3 | ✅ |
| eval_0006 | en | hard | Yes | 0.143 | 0.029 | +0.114 | 8 | 2 | ✅ |
| eval_0007 | en | medium | Yes | 0.098 | 0.049 | +0.049 | 10 | 1 | ✅ |
| eval_0008 | en | hard | Yes | 0.143 | 0.086 | +0.057 | 25 | 2 | ✅ |
| eval_0009 | en | medium | Yes | 0.118 | 0.000 | +0.118 | 25 | 2 | ✅ |
| eval_0010 | en | medium | Yes | 0.129 | 0.129 | +0.000 | 0 | 0 | ✅ |
| eval_0011 | en | medium | Yes | 0.125 | 0.000 | +0.125 | 19 | 3 | ✅ |
| eval_0012 | en | hard | Yes | 0.138 | 0.138 | +0.000 | 0 | 0 | ✅ |
| eval_0013 | en | hard | Yes | 0.125 | 0.031 | +0.094 | 7 | 1 | ✅ |
| eval_0014 | en | hard | Yes | 0.029 | 0.059 | -0.029 | 1 | 1 | ✅ |
| eval_0015 | en | hard | Yes | 0.094 | 0.125 | -0.031 | 1 | 1 | ✅ |
| eval_0016 | en | hard | Yes | 0.088 | 0.000 | +0.088 | 15 | 3 | ✅ |
| eval_0017 | en | medium | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0018 | en | medium | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0019 | en | hard | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0020 | en | hard | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0021 | en | hard | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0022 | ar | medium | Yes | 0.429 | 0.143 | +0.286 | 2 | 2 | ✅ |
| eval_0023 | en | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0024 | en | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0025 | en | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0026 | en | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0027 | ar | medium | Yes | 0.750 | 0.500 | +0.250 | 1 | 1 | ✅ |
| eval_0028 | ar | medium | Yes | 1.000 | 0.400 | +0.600 | 3 | 1 | ✅ |
| eval_0029 | ar | medium | Yes | 1.000 | 0.500 | +0.500 | 2 | 1 | ✅ |
| eval_0030 | en | medium | Yes | 0.250 | 0.750 | -0.500 | 2 | 1 | ✅ |
| eval_0031 | en | medium | Yes | 0.333 | 0.667 | -0.333 | 3 | 2 | ✅ |
| eval_0032 | en | medium | Yes | 0.667 | 0.333 | +0.333 | 1 | 1 | ✅ |
| eval_0033 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0034 | ar | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0035 | ar | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0036 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0037 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0038 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0039 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0040 | ar | medium | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0041 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0042 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0043 | ar | medium | Yes | 0.667 | 0.333 | +0.333 | 1 | 1 | ✅ |
| eval_0044 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0045 | ar | medium | Yes | 1.000 | 0.667 | +0.333 | 1 | 1 | ✅ |
| eval_0046 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0047 | ar | hard | Yes | 1.000 | 0.000 | +1.000 | 2 | 1 | ✅ |
| eval_0048 | ar | medium | Yes | 1.000 | 0.000 | +1.000 | 2 | 1 | ✅ |
| eval_0049 | ar | medium | Yes | 1.000 | 0.000 | +1.000 | 2 | 1 | ✅ |
| eval_0050 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 1 | 1 | ✅ |
| eval_0051 | ar | hard | Yes | 1.000 | 1.000 | +0.000 | 1 | 1 | ✅ |
| eval_0052 | ar | medium | Yes | 2.000 | 2.000 | +0.000 | 0 | 0 | ✅ |
| eval_0053 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0054 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0055 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0056 | ar | hard | Yes | 1.000 | 1.000 | +0.000 | 0 | 1 | ✅ |
| eval_0057 | ar | medium | Yes | 1.000 | 0.500 | +0.500 | 2 | 1 | ✅ |
| eval_0058 | ar | hard | Yes | 1.000 | 0.400 | +0.600 | 3 | 1 | ✅ |
| eval_0059 | ar | hard | Yes | 1.000 | 0.833 | +0.167 | 1 | 2 | ✅ |
| eval_0060 | ar | hard | Yes | 1.000 | 0.333 | +0.667 | 2 | 1 | ✅ |
| eval_0061 | ar | hard | Yes | 1.000 | 0.500 | +0.500 | 2 | 1 | ✅ |
| eval_0062 | ar | hard | Yes | 1.500 | 1.500 | +0.000 | 0 | 0 | ✅ |
| eval_0063 | ar | hard | Yes | 0.833 | 0.500 | +0.333 | 2 | 1 | ✅ |
| eval_0064 | ar | hard | Yes | 1.000 | 0.600 | +0.400 | 2 | 2 | ✅ |
| eval_0065 | ar | hard | Yes | 1.143 | 0.714 | +0.429 | 3 | 1 | ✅ |
| eval_0066 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 1 | 1 | ✅ |
| eval_0067 | ar | medium | Yes | 1.000 | 0.000 | +1.000 | 2 | 1 | ✅ |
| eval_0068 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0069 | ar | hard | Yes | 1.000 | 1.000 | +0.000 | 1 | 1 | ✅ |
| eval_0070 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0071 | ar | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0072 | ar | medium | Yes | 0.500 | 0.000 | +0.500 | 1 | 1 | ✅ |
| eval_0073 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0074 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0075 | ar | medium | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0076 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0077 | ar | easy | Yes | 0.333 | 0.333 | +0.000 | 0 | 0 | ✅ |
| eval_0078 | ar | medium | Yes | 0.500 | 0.500 | +0.000 | 0 | 0 | ✅ |
| eval_0079 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0080 | ar | easy | Yes | 0.500 | 1.000 | -0.500 | 1 | 1 | ✅ |
| eval_0081 | ar | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0082 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0083 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0084 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0085 | ar | medium | Yes | 0.500 | 0.500 | +0.000 | 0 | 0 | ✅ |
| eval_0086 | ar | medium | Yes | 0.500 | 0.500 | +0.000 | 0 | 0 | ✅ |
| eval_0087 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0088 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0089 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0090 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0091 | en | easy | No | 0.000 | 0.750 | -0.750 | 5 | 2 | ❌ |
| eval_0092 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0093 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0094 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0095 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0096 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0097 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0098 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0099 | en | easy | No | 0.000 | 0.400 | -0.400 | 3 | 1 | ❌ |
| eval_0100 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0101 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0102 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0103 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0104 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0105 | en | easy | No | 0.000 | 0.100 | -0.100 | 3 | 1 | ❌ |
| eval_0106 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0107 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0108 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0109 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0110 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0111 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0112 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0113 | en | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0114 | en | easy | No | 0.000 | 0.200 | -0.200 | 2 | 1 | ❌ |
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
| eval_0133 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0134 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 1 | ❌ |
| eval_0135 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0136 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0137 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0138 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 1 | ❌ |
| eval_0139 | ar | easy | No | 0.000 | 0.143 | -0.143 | 1 | 2 | ❌ |
| eval_0140 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0141 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0142 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0143 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0144 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0145 | ar | easy | No | 0.000 | 0.200 | -0.200 | 1 | 1 | ❌ |
| eval_0146 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0147 | ar | easy | No | 0.000 | 0.167 | -0.167 | 1 | 1 | ❌ |
| eval_0148 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0149 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0150 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0151 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0152 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0153 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0154 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 1 | ❌ |
| eval_0155 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0156 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0157 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0158 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0159 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 1 | ❌ |
| eval_0160 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0161 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0162 | ar | easy | No | 0.000 | 0.200 | -0.200 | 1 | 1 | ❌ |
| eval_0163 | ar | easy | No | 0.000 | 0.400 | -0.400 | 2 | 2 | ❌ |
| eval_0164 | ar | easy | No | 0.000 | 0.333 | -0.333 | 2 | 2 | ❌ |
| eval_0165 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0166 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0167 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0168 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0169 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0170 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0171 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0172 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0173 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0174 | ar | easy | No | 0.000 | 0.500 | -0.500 | 1 | 1 | ❌ |
| eval_0175 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0176 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0177 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0178 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0179 | mixed | hard | Yes | 0.400 | 0.000 | +0.400 | 2 | 2 | ✅ |
| eval_0180 | mixed | medium | Yes | 0.250 | 0.750 | -0.500 | 2 | 1 | ✅ |
| eval_0181 | mixed | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0182 | mixed | hard | Yes | 0.286 | 0.000 | +0.286 | 2 | 1 | ✅ |
| eval_0183 | mixed | medium | Yes | 0.250 | 0.000 | +0.250 | 1 | 1 | ✅ |
| eval_0184 | mixed | medium | Yes | 0.286 | 0.143 | +0.143 | 1 | 1 | ✅ |
| eval_0185 | mixed | hard | Yes | 0.200 | 0.000 | +0.200 | 1 | 1 | ✅ |
| eval_0186 | mixed | medium | Yes | 0.200 | 0.200 | +0.000 | 0 | 0 | ✅ |
| eval_0187 | mixed | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0188 | mixed | hard | Yes | 0.286 | 0.143 | +0.143 | 1 | 1 | ✅ |
| eval_0189 | mixed | medium | Yes | 0.250 | 0.250 | +0.000 | 0 | 1 | ✅ |
| eval_0190 | mixed | medium | Yes | 0.250 | 0.250 | +0.000 | 0 | 1 | ✅ |
| eval_0191 | mixed | hard | Yes | 0.500 | 0.250 | +0.250 | 1 | 2 | ✅ |
| eval_0192 | mixed | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0193 | mixed | hard | Yes | 0.429 | 0.143 | +0.286 | 3 | 2 | ✅ |
| eval_0194 | en | medium | Yes | 0.167 | 0.500 | -0.333 | 2 | 1 | ✅ |
| eval_0195 | en | easy | Yes | 0.250 | 0.500 | -0.250 | 3 | 2 | ✅ |
| eval_0196 | en | medium | Yes | 0.200 | 0.400 | -0.200 | 3 | 1 | ✅ |
| eval_0197 | en | medium | Yes | 0.400 | 0.000 | +0.400 | 2 | 2 | ✅ |
| eval_0198 | en | medium | Yes | 0.200 | 0.000 | +0.200 | 1 | 1 | ✅ |
| eval_0199 | en | medium | Yes | 0.400 | 0.200 | +0.200 | 1 | 1 | ✅ |
| eval_0200 | en | easy | Yes | 0.333 | 0.000 | +0.333 | 1 | 1 | ✅ |
| eval_0201 | en | medium | Yes | 0.333 | 0.000 | +0.333 | 1 | 1 | ✅ |
| eval_0202 | en | medium | Yes | 0.400 | 0.000 | +0.400 | 2 | 2 | ✅ |
| eval_0203 | en | easy | Yes | 0.250 | 0.250 | +0.000 | 0 | 0 | ✅ |
| eval_0204 | en | easy | Yes | 0.333 | 0.000 | +0.333 | 1 | 1 | ✅ |
| eval_0205 | en | medium | Yes | 0.333 | 0.333 | +0.000 | 0 | 0 | ✅ |
| eval_0206 | en | easy | Yes | 0.333 | 0.000 | +0.333 | 1 | 1 | ✅ |
| eval_0207 | en | medium | Yes | 0.333 | 0.333 | +0.000 | 0 | 0 | ✅ |
| eval_0208 | en | hard | Yes | 0.250 | 0.250 | +0.000 | 0 | 0 | ✅ |
| eval_0209 | en | hard | Yes | 0.500 | 0.500 | +0.000 | 0 | 0 | ✅ |
| eval_0210 | en | medium | Yes | 0.250 | 0.250 | +0.000 | 0 | 1 | ✅ |
| eval_0211 | en | medium | Yes | 0.250 | 0.250 | +0.000 | 0 | 0 | ✅ |
| eval_0212 | en | medium | Yes | 0.200 | 0.200 | +0.000 | 2 | 2 | ✅ |
| eval_0213 | en | easy | Yes | 0.250 | 0.000 | +0.250 | 1 | 1 | ✅ |
| eval_0214 | en | easy | Yes | 0.200 | 0.200 | +0.000 | 0 | 0 | ✅ |
| eval_0215 | en | hard | Yes | 0.333 | 0.333 | +0.000 | 0 | 0 | ✅ |

---

## Failure Cases

**15 false positives (changes to clean input):**

- **eval_0174**: 'سليم'→'سلام'
- **eval_0162**: 'المعدل'→'المعده'
- **eval_0134**: 'حالك'→'حالي'
- **eval_0099**: 'examination'→'reveals', 'reveals'→'no', 'no'→'abnormalities.'
- **eval_0138**: 'بتحسن'→'بحسن'
- **eval_0164**: 'يدخن'→'يدخل', 'يشرب'→'بشرب'
- **eval_0159**: 'تمت'→'تمتد'
- **eval_0105**: 'or'→'shortness', 'shortness'→'of', 'of'→'breath.'
- **eval_0145**: 'شرح'→'جرح'
- **eval_0147**: 'نظيف'→'نزيف'
- **eval_0091**: '120/80'→'heart', 'HR'→'rate', '72'→'Temp', 'Temp'→'rheumatoid', '37.2'→'arthritis'
- **eval_0114**: 'exam'→'is', 'is'→'unremarkable.'
- **eval_0154**: 'قليلا'→'قليل'
- **eval_0139**: 'الله'→'اكله'
- **eval_0163**: 'المرضي'→'المرض', 'الكامل'→'الشامل'

**81 records with missed corrections:**

- **eval_0036** (lang=ar, diff=easy)
  - Gold: 'هايبرتنشن'→'hypertension'
  - Changes applied: (none)
- **eval_0030** (lang=en, diff=medium)
  - Gold: 'clopidogr 75 mg daily'→'clopidogrel 75 mg daily'
  - Changes applied: '75'→'myasthenia', 'mg'→'gravis'
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
  - Changes applied: 'الفيتل'→'الفيصل'
- **eval_0013** (lang=en, diff=hard)
  - Gold: 'giant cell arteritus'→'giant cell arteritis', 'predni and sone'→'prednisone'
  - Changes applied: 'predni'→'prednisone', 'and'→'today', 'sone'→'and', 'today'→'arrange', 'and'→'temporal', 'arrange'→'artery', 'temporal'→'evaluation.'
- **eval_0005** (lang=en, diff=medium)
  - Gold: 'pyelonefritis'→'pyelonephritis', 'cipro and floxacin'→'ciprofloxacin'
  - Changes applied: 'pyelonefritis'→'pyelonephritis', 'back'→'blood', 'pain,'→'sugar', 'she'→'pain,', 'should'→'she', 'come'→'should', 'back'→'come', 'to'→'blood', 'the'→'sugar', 'emergency'→'to', 'department'→'the', 'immediately.'→'emergency'

---

## Configuration

- **Pipeline:** MedicalCorrector (deterministic, no LLM)
- **Eval set:** `eval/correction_eval.jsonl`
- **LLM:** Disabled (baseline)
- **USE_LLM=0**