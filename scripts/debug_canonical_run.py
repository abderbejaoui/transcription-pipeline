from app.pipeline.scorer import tokenize_transcript
from app.pipeline.flagger import flag_suspicious_spans
from app.pipeline.retriever import retrieve_candidates
from app.pipeline.decider import decide_spans
from app.pipeline.runner import _apply_replacements
from app.pipeline.models import ScoredWord

transcript = (
    "The patient presents with fever and should take dolly prahn twice daily "
    "alongside salbu tamol for the wheeze. Blood pressure was measured using a sfigmomanometre. "
    "The attending physician prescribed amoxicilin for the secondary infection."
)

toks = tokenize_transcript(transcript)
print('tokens:', toks)

# canonical scores
suspicion_by_index = {1:0.05,2:0.08,4:0.06,6:0.03,7:0.04,8:0.87,9:0.92,10:0.04,11:0.04,13:0.84,14:0.81,17:0.09,19:0.05,21:0.04,24:0.96,26:0.04,27:0.06,28:0.05,29:0.71,32:0.04,33:0.07}
in_lex = {4,17,33}
scored = []
for i,(t,s,e) in enumerate(toks):
    scored.append(ScoredWord(index=i,text=t,suspicion=float(suspicion_by_index.get(i,0.0)),in_lexicon=(i in in_lex),start=s,end=e))
print('\nScored words:')
for w in scored:
    print(w)

spans = flag_suspicious_spans(scored)
print('\nSpans:')
for sp in spans:
    print(sp)

span_candidates = [retrieve_candidates(sp) for sp in spans]
print('\nCandidates:')
for sc in span_candidates:
    print(sc.span, '->', [c.term+':'+str(c.phonetic_score) for c in sc.candidates[:5]])

decisions = decide_spans(transcript, span_candidates)
print('\nDecisions:')
for d in decisions:
    print(d)

corrected = _apply_replacements(transcript, scored, decisions)
print('\nCorrected:\n', corrected)
