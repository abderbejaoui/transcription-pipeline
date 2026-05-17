"""Why is Kuwait WER stuck at 60%? Print all clean KW clips to diagnose."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_v2 import load_model, score_records

records = load_model("qwen3-asr-1.7b")
scored = score_records(records)
kw = [r for r in scored if r["source"] == "WorldSpeech-KW" and not r["broken_ref"]]
kw.sort(key=lambda r: r["wer"])

print(f"\nWorldSpeech-KW clean: {len(kw)} clips")
for i, r in enumerate(kw, 1):
    print(f"\n[{i:2d}] WER={r['wer']*100:5.1f}%  CER={r['cer']*100:5.1f}%  ({r['id']})")
    print(f"     ref : {r['ref']}")
    print(f"     pred: {r['pred']}")
