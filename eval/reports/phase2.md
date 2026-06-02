# Correction Pipeline Evaluation — `phase2`

**Date:** 2026-06-01 20:25:00  
**Elapsed:** 3.0s  
**Records:** 216 (108 errors)  

---

## Summary

| Metric | Value |
|--------|-------|
| **Records evaluated** | 108 |
| **Mean WER (raw → gold)** | 0.5124 |
| **Mean WER (corrected → gold)** | 0.4230 |
| **WER reduction (Δ)** | **+0.0894** |
| **Correction precision** | 0.1098 (9/82) |
| **Correction recall** | 0.2222 (14/63) |
| **F1 score** | 0.1469 |
| **Do-no-harm rate** | 0.5532 (26/47) |
| **Total flags (HITL)** | 71 |
| **Avg flags/record** | 0.66 |
| **Avg corrections applied/record** | 0.76 |

### WER by Language

| Language | N | WER (raw) | WER (corrected) | Δ |
|----------|---|-----------|-----------------|----|
| ar | 104 | 0.5207 | 0.4331 | +0.0876 |
| mixed | 4 | 0.2964 | 0.1607 | +0.1357 |

### WER by Difficulty

| Difficulty | N | WER (raw) | WER (corrected) | Δ |
|------------|---|-----------|-----------------|----|
| easy | 62 | 0.2231 | 0.2761 | -0.0530 |
| hard | 14 | 0.9401 | 0.6088 | +0.3313 |
| medium | 32 | 0.8858 | 0.6263 | +0.2594 |

---

## Per-Record Details

| ID | Lang | Diff | Contains Error? | WER raw | WER corr | Δ | Changes | Flags | Do-no-harm |
|----|------|------|----------------|---------|----------|----|---------|-------|------------|
| eval_0022 | ar | medium | Yes | 0.429 | 0.143 | +0.286 | 2 | 2 | ✅ |
| eval_0027 | ar | medium | Yes | 0.750 | 0.500 | +0.250 | 1 | 1 | ✅ |
| eval_0028 | ar | medium | Yes | 1.000 | 0.400 | +0.600 | 3 | 1 | ✅ |
| eval_0029 | ar | medium | Yes | 1.000 | 0.500 | +0.500 | 2 | 1 | ✅ |
| eval_0033 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 1 | 1 | ✅ |
| eval_0034 | ar | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0035 | ar | easy | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0036 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0037 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0038 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0039 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 0 | 0 | ✅ |
| eval_0040 | ar | medium | Yes | 1.000 | 0.000 | +1.000 | 1 | 1 | ✅ |
| eval_0041 | ar | easy | Yes | 1.000 | 1.000 | +0.000 | 1 | 1 | ✅ |
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
| eval_0056 | ar | hard | Yes | 1.000 | 1.000 | +0.000 | 1 | 1 | ✅ |
| eval_0057 | ar | medium | Yes | 1.000 | 0.500 | +0.500 | 2 | 1 | ✅ |
| eval_0058 | ar | hard | Yes | 1.000 | 0.400 | +0.600 | 3 | 1 | ✅ |
| eval_0059 | ar | hard | Yes | 1.000 | 0.833 | +0.167 | 1 | 2 | ✅ |
| eval_0060 | ar | hard | Yes | 1.000 | 0.333 | +0.667 | 2 | 1 | ✅ |
| eval_0061 | ar | hard | Yes | 1.000 | 0.500 | +0.500 | 2 | 1 | ✅ |
| eval_0062 | ar | hard | Yes | 1.500 | 1.500 | +0.000 | 0 | 0 | ✅ |
| eval_0063 | ar | hard | Yes | 0.833 | 0.500 | +0.333 | 2 | 1 | ✅ |
| eval_0064 | ar | hard | Yes | 1.000 | 0.600 | +0.400 | 2 | 2 | ✅ |
| eval_0065 | ar | hard | Yes | 1.143 | 0.714 | +0.429 | 3 | 1 | ✅ |
| eval_0066 | ar | medium | Yes | 1.000 | 1.000 | +0.000 | 2 | 2 | ✅ |
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
| eval_0133 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0134 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 1 | ❌ |
| eval_0135 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0136 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0137 | ar | easy | No | 0.000 | 0.250 | -0.250 | 1 | 1 | ❌ |
| eval_0138 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 1 | ❌ |
| eval_0139 | ar | easy | No | 0.000 | 0.286 | -0.286 | 2 | 3 | ❌ |
| eval_0140 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0141 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0142 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 1 | ❌ |
| eval_0143 | ar | easy | No | 0.000 | 0.400 | -0.400 | 2 | 2 | ❌ |
| eval_0144 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0145 | ar | easy | No | 0.000 | 0.200 | -0.200 | 1 | 1 | ❌ |
| eval_0146 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0147 | ar | easy | No | 0.000 | 0.167 | -0.167 | 1 | 1 | ❌ |
| eval_0148 | ar | easy | No | 0.000 | 0.250 | -0.250 | 1 | 1 | ❌ |
| eval_0149 | ar | easy | No | 0.000 | 0.250 | -0.250 | 1 | 1 | ❌ |
| eval_0150 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0151 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0152 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0153 | ar | easy | No | 0.000 | 0.250 | -0.250 | 1 | 1 | ❌ |
| eval_0154 | ar | easy | No | 0.000 | 0.333 | -0.333 | 1 | 1 | ❌ |
| eval_0155 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0156 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0157 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0158 | ar | easy | No | 0.000 | 0.200 | -0.200 | 1 | 1 | ❌ |
| eval_0159 | ar | easy | No | 0.000 | 0.667 | -0.667 | 2 | 2 | ❌ |
| eval_0160 | ar | easy | No | 0.000 | 0.250 | -0.250 | 1 | 1 | ❌ |
| eval_0161 | ar | easy | No | 0.000 | 0.200 | -0.200 | 1 | 1 | ❌ |
| eval_0162 | ar | easy | No | 0.000 | 0.400 | -0.400 | 2 | 2 | ❌ |
| eval_0163 | ar | easy | No | 0.000 | 0.600 | -0.600 | 3 | 3 | ❌ |
| eval_0164 | ar | easy | No | 0.000 | 0.333 | -0.333 | 2 | 2 | ❌ |
| eval_0165 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0166 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0167 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0168 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0169 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0170 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0171 | ar | easy | No | 0.000 | 0.250 | -0.250 | 1 | 1 | ❌ |
| eval_0172 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0173 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0174 | ar | easy | No | 0.000 | 0.500 | -0.500 | 1 | 1 | ❌ |
| eval_0175 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0176 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0177 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0178 | ar | easy | No | 0.000 | 0.000 | +0.000 | 0 | 0 | ✅ |
| eval_0179 | mixed | hard | Yes | 0.400 | 0.000 | +0.400 | 2 | 2 | ✅ |
| eval_0188 | mixed | hard | Yes | 0.286 | 0.143 | +0.143 | 1 | 1 | ✅ |
| eval_0189 | mixed | medium | Yes | 0.250 | 0.250 | +0.000 | 0 | 1 | ✅ |
| eval_0190 | mixed | medium | Yes | 0.250 | 0.250 | +0.000 | 0 | 1 | ✅ |

---

## Failure Cases

**21 false positives (changes to clean input):**

- **eval_0174**: 'سليم'→'سلام'
- **eval_0171**: 'الالتزام'→'الالطعام'
- **eval_0162**: 'ضمن'→'من', 'المعدل'→'المعده'
- **eval_0134**: 'حالك'→'حالي'
- **eval_0138**: 'بتحسن'→'بحسن'
- **eval_0142**: 'السريري'→'الضروري'
- **eval_0164**: 'يدخن'→'يدخل', 'يشرب'→'بشرب'
- **eval_0153**: 'لله'→'للهم'
- **eval_0159**: 'تمت'→'تمتد', 'بنجاح'→'براح'
- **eval_0148**: 'التعليمات'→'العلامات'
- **eval_0149**: 'يتحمل'→'بتشمل'
- **eval_0161**: 'مضاعفات'→'مضادات'
- **eval_0160**: 'تتطلب'→'بيطلب'
- **eval_0145**: 'شرح'→'جرح'
- **eval_0158**: 'يمارس'→'يسار'
- **eval_0147**: 'نظيف'→'نزيف'
- **eval_0137**: 'لله'→'للهم'
- **eval_0143**: 'ضمن'→'من', 'الحدود'→'الحضور'
- **eval_0154**: 'قليلا'→'قليل'
- **eval_0139**: 'نتابع'→'متابعة', 'الله'→'اكله'
- **eval_0163**: 'التاريخ'→'الياريت', 'المرضي'→'المرض', 'الكامل'→'الشامل'

**49 records with missed corrections:**

- **eval_0036** (lang=ar, diff=easy)
  - Gold: 'هايبرتنشن'→'hypertension'
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
- **eval_0066** (lang=ar, diff=medium)
  - Gold: 'الفيتل ساينز'→'vital signs'
  - Changes applied: 'الفيتل'→'الفيصل', 'ساينز'→'ساقين'
- **eval_0042** (lang=ar, diff=easy)
  - Gold: 'شوجر'→'sugar'
  - Changes applied: (none)
- **eval_0044** (lang=ar, diff=medium)
  - Gold: 'هايبرتنشن شديد'→'hypertension severe'
  - Changes applied: (none)
- **eval_0064** (lang=ar, diff=hard)
  - Gold: 'دكتور عنده هستوري اوف دايابيتس'→'doctor has history of diabetes'
  - Changes applied: 'هستوري'→'history', 'دايابيتس'→'diabetes'
- **eval_0038** (lang=ar, diff=easy)
  - Gold: 'بريث'→'breath'
  - Changes applied: (none)

---

## Configuration

- **Pipeline:** MedicalCorrector (deterministic, no LLM)
- **Eval set:** `eval/correction_eval.jsonl`
- **LLM:** Disabled (baseline)
- **USE_LLM=0**