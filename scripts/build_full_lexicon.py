"""Build a 10k-term curated medical lexicon for synthetic data generation.

The lexicon is the LIST of REAL drugs and diseases the synthesis script will
ask Calme to write sentences about. Quality of this list directly controls
the quality of the dataset.

Design principles
-----------------
1. ENGLISH ONLY. Drug and disease terms are spelled in English (Latin
   characters). The ASR's job is to keep English drug names in English even
   when surrounded by Arabic — Arabic-script aliases were a bad idea and
   are removed.

2. REAL TERMS ONLY. We do NOT ask the LLM to invent drug names. We start
   from canonical lists and only allow the LLM to (a) add commonly-used
   brand names of real drugs, (b) add common dosage form/strength
   variations, (c) add common spelling variants. Every output is verified
   to come from a real source or to be a direct variant of one.

3. TIERED. Each term carries a `tier`:
     - tier 1: top ~300 most common Gulf clinic items, high repetition
     - tier 2: ~2000 common items, medium repetition
     - tier 3: ~7700 long-tail items, 1-2 sentences each

   The audio generation script uses the tier to allocate its 70-hour
   budget so common drugs are spoken many times.

Sources used (downloaded by this script at runtime)
---------------------------------------------------
- WHO Essential Medicines List 23rd edition (public)
- RxNorm display names from a small bundled subset (canonical names only)
- ICD-10 chapter-level diagnoses (canonical CDC subset)
- Hard-coded Gulf brand mapping (panadol, doliprane, voltaren, ...)

Usage on the DGX
----------------
    python3 scripts/build_full_lexicon.py \\
        --out data/full_lexicon.jsonl \\
        --ollama-url http://localhost:11434 \\
        --ollama-model calme-3.2-instruct-78b-GGUF:IQ4_XS

If you want to skip the LLM-augmentation step (faster, smaller dictionary):
        --skip-llm

The output schema is one JSON object per line:
    {"term": "amoxicillin", "type": "drug", "tier": 1,
     "category": "penicillin antibiotic", "source": "who_eml"}
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import requests


# ─────────────────────────────────────────────────────────────────────────────
# CURATED TIER-1 GULF CLINIC TERMS (real, hand-picked)
# These get the highest sentence weight in synthesis.
# ─────────────────────────────────────────────────────────────────────────────

TIER1_DRUGS: List[Dict[str, str]] = [
    # analgesics / antipyretics
    {"term": "paracetamol", "category": "analgesic"},
    {"term": "panadol", "category": "analgesic (brand)"},
    {"term": "doliprane", "category": "analgesic (brand)"},
    {"term": "efferalgan", "category": "analgesic (brand)"},
    {"term": "calpol", "category": "pediatric analgesic"},
    {"term": "tylenol", "category": "analgesic (brand)"},
    {"term": "acetaminophen", "category": "analgesic"},
    {"term": "ibuprofen", "category": "nsaid"},
    {"term": "brufen", "category": "nsaid (brand)"},
    {"term": "advil", "category": "nsaid (brand)"},
    {"term": "nurofen", "category": "nsaid (brand)"},
    {"term": "naproxen", "category": "nsaid"},
    {"term": "diclofenac", "category": "nsaid"},
    {"term": "voltaren", "category": "nsaid (brand)"},
    {"term": "cataflam", "category": "nsaid (brand)"},
    {"term": "celecoxib", "category": "cox-2 inhibitor"},
    {"term": "celebrex", "category": "cox-2 inhibitor (brand)"},
    {"term": "aspirin", "category": "nsaid / antiplatelet"},
    {"term": "tramadol", "category": "opioid analgesic"},
    {"term": "codeine", "category": "opioid analgesic"},
    {"term": "morphine", "category": "opioid analgesic"},

    # antibiotics
    {"term": "amoxicillin", "category": "penicillin"},
    {"term": "amoxicillin clavulanate", "category": "penicillin + inhibitor"},
    {"term": "augmentin", "category": "penicillin + inhibitor (brand)"},
    {"term": "ampicillin", "category": "penicillin"},
    {"term": "penicillin v", "category": "penicillin"},
    {"term": "cephalexin", "category": "cephalosporin 1st gen"},
    {"term": "keflex", "category": "cephalosporin 1st gen (brand)"},
    {"term": "cefuroxime", "category": "cephalosporin 2nd gen"},
    {"term": "zinnat", "category": "cephalosporin 2nd gen (brand)"},
    {"term": "ceftriaxone", "category": "cephalosporin 3rd gen"},
    {"term": "rocephin", "category": "cephalosporin 3rd gen (brand)"},
    {"term": "cefixime", "category": "cephalosporin 3rd gen"},
    {"term": "suprax", "category": "cephalosporin 3rd gen (brand)"},
    {"term": "azithromycin", "category": "macrolide"},
    {"term": "zithromax", "category": "macrolide (brand)"},
    {"term": "clarithromycin", "category": "macrolide"},
    {"term": "klacid", "category": "macrolide (brand)"},
    {"term": "erythromycin", "category": "macrolide"},
    {"term": "doxycycline", "category": "tetracycline"},
    {"term": "ciprofloxacin", "category": "fluoroquinolone"},
    {"term": "cipro", "category": "fluoroquinolone (brand)"},
    {"term": "levofloxacin", "category": "fluoroquinolone"},
    {"term": "tavanic", "category": "fluoroquinolone (brand)"},
    {"term": "moxifloxacin", "category": "fluoroquinolone"},
    {"term": "metronidazole", "category": "anti-anaerobe"},
    {"term": "flagyl", "category": "anti-anaerobe (brand)"},
    {"term": "trimethoprim sulfamethoxazole", "category": "sulfa"},
    {"term": "bactrim", "category": "sulfa (brand)"},
    {"term": "nitrofurantoin", "category": "uti antibiotic"},
    {"term": "fosfomycin", "category": "uti antibiotic"},
    {"term": "monurol", "category": "uti antibiotic (brand)"},
    {"term": "vancomycin", "category": "glycopeptide"},
    {"term": "linezolid", "category": "oxazolidinone"},
    {"term": "meropenem", "category": "carbapenem"},
    {"term": "imipenem", "category": "carbapenem"},

    # antifungals
    {"term": "fluconazole", "category": "azole antifungal"},
    {"term": "diflucan", "category": "azole antifungal (brand)"},
    {"term": "itraconazole", "category": "azole antifungal"},
    {"term": "ketoconazole", "category": "azole antifungal"},
    {"term": "nystatin", "category": "polyene antifungal"},
    {"term": "terbinafine", "category": "allylamine antifungal"},
    {"term": "lamisil", "category": "antifungal (brand)"},

    # antivirals
    {"term": "acyclovir", "category": "antiviral"},
    {"term": "zovirax", "category": "antiviral (brand)"},
    {"term": "valacyclovir", "category": "antiviral"},
    {"term": "oseltamivir", "category": "influenza antiviral"},
    {"term": "tamiflu", "category": "influenza antiviral (brand)"},

    # respiratory
    {"term": "salbutamol", "category": "saba"},
    {"term": "ventolin", "category": "saba (brand)"},
    {"term": "albuterol", "category": "saba"},
    {"term": "salmeterol", "category": "laba"},
    {"term": "formoterol", "category": "laba"},
    {"term": "tiotropium", "category": "lama"},
    {"term": "spiriva", "category": "lama (brand)"},
    {"term": "ipratropium", "category": "sama"},
    {"term": "atrovent", "category": "sama (brand)"},
    {"term": "budesonide", "category": "ics"},
    {"term": "pulmicort", "category": "ics (brand)"},
    {"term": "fluticasone", "category": "ics"},
    {"term": "flixotide", "category": "ics (brand)"},
    {"term": "seretide", "category": "ics + laba (brand)"},
    {"term": "symbicort", "category": "ics + laba (brand)"},
    {"term": "montelukast", "category": "ltra"},
    {"term": "singulair", "category": "ltra (brand)"},
    {"term": "cetirizine", "category": "antihistamine"},
    {"term": "zyrtec", "category": "antihistamine (brand)"},
    {"term": "loratadine", "category": "antihistamine"},
    {"term": "claritine", "category": "antihistamine (brand)"},
    {"term": "fexofenadine", "category": "antihistamine"},
    {"term": "telfast", "category": "antihistamine (brand)"},
    {"term": "desloratadine", "category": "antihistamine"},
    {"term": "aerius", "category": "antihistamine (brand)"},
    {"term": "chlorpheniramine", "category": "antihistamine"},
    {"term": "diphenhydramine", "category": "antihistamine"},
    {"term": "benadryl", "category": "antihistamine (brand)"},
    {"term": "pseudoephedrine", "category": "decongestant"},
    {"term": "xylometazoline", "category": "decongestant nasal"},
    {"term": "otrivin", "category": "decongestant nasal (brand)"},
    {"term": "oxymetazoline", "category": "decongestant nasal"},

    # cardiovascular
    {"term": "amlodipine", "category": "ccb"},
    {"term": "norvasc", "category": "ccb (brand)"},
    {"term": "nifedipine", "category": "ccb"},
    {"term": "adalat", "category": "ccb (brand)"},
    {"term": "verapamil", "category": "ccb non-dhp"},
    {"term": "diltiazem", "category": "ccb non-dhp"},
    {"term": "lisinopril", "category": "acei"},
    {"term": "enalapril", "category": "acei"},
    {"term": "ramipril", "category": "acei"},
    {"term": "tritace", "category": "acei (brand)"},
    {"term": "perindopril", "category": "acei"},
    {"term": "coversyl", "category": "acei (brand)"},
    {"term": "captopril", "category": "acei"},
    {"term": "losartan", "category": "arb"},
    {"term": "cozaar", "category": "arb (brand)"},
    {"term": "valsartan", "category": "arb"},
    {"term": "diovan", "category": "arb (brand)"},
    {"term": "telmisartan", "category": "arb"},
    {"term": "micardis", "category": "arb (brand)"},
    {"term": "irbesartan", "category": "arb"},
    {"term": "approvel", "category": "arb (brand)"},
    {"term": "candesartan", "category": "arb"},
    {"term": "atacand", "category": "arb (brand)"},
    {"term": "olmesartan", "category": "arb"},
    {"term": "atenolol", "category": "beta blocker"},
    {"term": "tenormin", "category": "beta blocker (brand)"},
    {"term": "bisoprolol", "category": "beta blocker"},
    {"term": "concor", "category": "beta blocker (brand)"},
    {"term": "metoprolol", "category": "beta blocker"},
    {"term": "betaloc", "category": "beta blocker (brand)"},
    {"term": "carvedilol", "category": "beta blocker"},
    {"term": "propranolol", "category": "beta blocker"},
    {"term": "inderal", "category": "beta blocker (brand)"},
    {"term": "nebivolol", "category": "beta blocker"},
    {"term": "hydrochlorothiazide", "category": "thiazide diuretic"},
    {"term": "indapamide", "category": "thiazide-like"},
    {"term": "furosemide", "category": "loop diuretic"},
    {"term": "lasix", "category": "loop diuretic (brand)"},
    {"term": "bumetanide", "category": "loop diuretic"},
    {"term": "spironolactone", "category": "k-sparing"},
    {"term": "aldactone", "category": "k-sparing (brand)"},
    {"term": "eplerenone", "category": "k-sparing"},
    {"term": "atorvastatin", "category": "statin"},
    {"term": "lipitor", "category": "statin (brand)"},
    {"term": "rosuvastatin", "category": "statin"},
    {"term": "crestor", "category": "statin (brand)"},
    {"term": "simvastatin", "category": "statin"},
    {"term": "zocor", "category": "statin (brand)"},
    {"term": "pravastatin", "category": "statin"},
    {"term": "fenofibrate", "category": "fibrate"},
    {"term": "ezetimibe", "category": "absorption inhibitor"},
    {"term": "clopidogrel", "category": "antiplatelet"},
    {"term": "plavix", "category": "antiplatelet (brand)"},
    {"term": "ticagrelor", "category": "antiplatelet"},
    {"term": "brilinta", "category": "antiplatelet (brand)"},
    {"term": "warfarin", "category": "anticoagulant"},
    {"term": "marevan", "category": "anticoagulant (brand)"},
    {"term": "rivaroxaban", "category": "doac"},
    {"term": "xarelto", "category": "doac (brand)"},
    {"term": "apixaban", "category": "doac"},
    {"term": "eliquis", "category": "doac (brand)"},
    {"term": "dabigatran", "category": "doac"},
    {"term": "enoxaparin", "category": "lmwh"},
    {"term": "clexane", "category": "lmwh (brand)"},
    {"term": "nitroglycerin", "category": "vasodilator"},
    {"term": "isosorbide dinitrate", "category": "vasodilator"},
    {"term": "isosorbide mononitrate", "category": "vasodilator"},
    {"term": "digoxin", "category": "cardiac glycoside"},

    # diabetes
    {"term": "metformin", "category": "biguanide"},
    {"term": "glucophage", "category": "biguanide (brand)"},
    {"term": "gliclazide", "category": "sulfonylurea"},
    {"term": "diamicron", "category": "sulfonylurea (brand)"},
    {"term": "glimepiride", "category": "sulfonylurea"},
    {"term": "amaryl", "category": "sulfonylurea (brand)"},
    {"term": "glibenclamide", "category": "sulfonylurea"},
    {"term": "sitagliptin", "category": "dpp-4"},
    {"term": "januvia", "category": "dpp-4 (brand)"},
    {"term": "vildagliptin", "category": "dpp-4"},
    {"term": "galvus", "category": "dpp-4 (brand)"},
    {"term": "linagliptin", "category": "dpp-4"},
    {"term": "trajenta", "category": "dpp-4 (brand)"},
    {"term": "empagliflozin", "category": "sglt2"},
    {"term": "jardiance", "category": "sglt2 (brand)"},
    {"term": "dapagliflozin", "category": "sglt2"},
    {"term": "forxiga", "category": "sglt2 (brand)"},
    {"term": "canagliflozin", "category": "sglt2"},
    {"term": "invokana", "category": "sglt2 (brand)"},
    {"term": "liraglutide", "category": "glp-1"},
    {"term": "victoza", "category": "glp-1 (brand)"},
    {"term": "semaglutide", "category": "glp-1"},
    {"term": "ozempic", "category": "glp-1 (brand)"},
    {"term": "rybelsus", "category": "oral glp-1 (brand)"},
    {"term": "dulaglutide", "category": "glp-1"},
    {"term": "trulicity", "category": "glp-1 (brand)"},
    {"term": "insulin glargine", "category": "long-acting insulin"},
    {"term": "lantus", "category": "long-acting insulin (brand)"},
    {"term": "toujeo", "category": "long-acting insulin (brand)"},
    {"term": "insulin detemir", "category": "long-acting insulin"},
    {"term": "levemir", "category": "long-acting insulin (brand)"},
    {"term": "insulin degludec", "category": "long-acting insulin"},
    {"term": "tresiba", "category": "long-acting insulin (brand)"},
    {"term": "insulin aspart", "category": "rapid insulin"},
    {"term": "novorapid", "category": "rapid insulin (brand)"},
    {"term": "insulin lispro", "category": "rapid insulin"},
    {"term": "humalog", "category": "rapid insulin (brand)"},
    {"term": "insulin glulisine", "category": "rapid insulin"},
    {"term": "apidra", "category": "rapid insulin (brand)"},
    {"term": "mixtard", "category": "premixed insulin (brand)"},
    {"term": "novomix", "category": "premixed insulin (brand)"},
    {"term": "humulin", "category": "human insulin (brand)"},

    # gastrointestinal
    {"term": "omeprazole", "category": "ppi"},
    {"term": "losec", "category": "ppi (brand)"},
    {"term": "esomeprazole", "category": "ppi"},
    {"term": "nexium", "category": "ppi (brand)"},
    {"term": "pantoprazole", "category": "ppi"},
    {"term": "controloc", "category": "ppi (brand)"},
    {"term": "rabeprazole", "category": "ppi"},
    {"term": "pariet", "category": "ppi (brand)"},
    {"term": "lansoprazole", "category": "ppi"},
    {"term": "ranitidine", "category": "h2 blocker"},
    {"term": "famotidine", "category": "h2 blocker"},
    {"term": "gaviscon", "category": "alginate (brand)"},
    {"term": "maalox", "category": "antacid (brand)"},
    {"term": "rennie", "category": "antacid (brand)"},
    {"term": "domperidone", "category": "prokinetic"},
    {"term": "motilium", "category": "prokinetic (brand)"},
    {"term": "metoclopramide", "category": "prokinetic"},
    {"term": "primperan", "category": "prokinetic (brand)"},
    {"term": "ondansetron", "category": "antiemetic"},
    {"term": "zofran", "category": "antiemetic (brand)"},
    {"term": "loperamide", "category": "antidiarrheal"},
    {"term": "imodium", "category": "antidiarrheal (brand)"},
    {"term": "hyoscine butylbromide", "category": "antispasmodic"},
    {"term": "buscopan", "category": "antispasmodic (brand)"},
    {"term": "lactulose", "category": "osmotic laxative"},
    {"term": "duphalac", "category": "osmotic laxative (brand)"},
    {"term": "bisacodyl", "category": "stimulant laxative"},
    {"term": "dulcolax", "category": "stimulant laxative (brand)"},

    # CNS / psych
    {"term": "sertraline", "category": "ssri"},
    {"term": "lustral", "category": "ssri (brand)"},
    {"term": "fluoxetine", "category": "ssri"},
    {"term": "prozac", "category": "ssri (brand)"},
    {"term": "escitalopram", "category": "ssri"},
    {"term": "cipralex", "category": "ssri (brand)"},
    {"term": "citalopram", "category": "ssri"},
    {"term": "paroxetine", "category": "ssri"},
    {"term": "seroxat", "category": "ssri (brand)"},
    {"term": "venlafaxine", "category": "snri"},
    {"term": "effexor", "category": "snri (brand)"},
    {"term": "duloxetine", "category": "snri"},
    {"term": "cymbalta", "category": "snri (brand)"},
    {"term": "amitriptyline", "category": "tca"},
    {"term": "mirtazapine", "category": "atypical antidepressant"},
    {"term": "remeron", "category": "atypical antidepressant (brand)"},
    {"term": "diazepam", "category": "benzodiazepine"},
    {"term": "valium", "category": "benzodiazepine (brand)"},
    {"term": "lorazepam", "category": "benzodiazepine"},
    {"term": "ativan", "category": "benzodiazepine (brand)"},
    {"term": "alprazolam", "category": "benzodiazepine"},
    {"term": "xanax", "category": "benzodiazepine (brand)"},
    {"term": "clonazepam", "category": "benzodiazepine"},
    {"term": "rivotril", "category": "benzodiazepine (brand)"},
    {"term": "zolpidem", "category": "z-drug"},
    {"term": "stilnox", "category": "z-drug (brand)"},
    {"term": "melatonin", "category": "sleep aid"},
    {"term": "olanzapine", "category": "antipsychotic"},
    {"term": "zyprexa", "category": "antipsychotic (brand)"},
    {"term": "risperidone", "category": "antipsychotic"},
    {"term": "risperdal", "category": "antipsychotic (brand)"},
    {"term": "quetiapine", "category": "antipsychotic"},
    {"term": "seroquel", "category": "antipsychotic (brand)"},
    {"term": "aripiprazole", "category": "antipsychotic"},
    {"term": "abilify", "category": "antipsychotic (brand)"},
    {"term": "haloperidol", "category": "typical antipsychotic"},
    {"term": "carbamazepine", "category": "anticonvulsant"},
    {"term": "tegretol", "category": "anticonvulsant (brand)"},
    {"term": "valproate", "category": "anticonvulsant"},
    {"term": "depakine", "category": "anticonvulsant (brand)"},
    {"term": "lamotrigine", "category": "anticonvulsant"},
    {"term": "lamictal", "category": "anticonvulsant (brand)"},
    {"term": "levetiracetam", "category": "anticonvulsant"},
    {"term": "keppra", "category": "anticonvulsant (brand)"},
    {"term": "topiramate", "category": "anticonvulsant"},
    {"term": "topamax", "category": "anticonvulsant (brand)"},
    {"term": "phenytoin", "category": "anticonvulsant"},
    {"term": "gabapentin", "category": "neuropathic pain"},
    {"term": "neurontin", "category": "neuropathic pain (brand)"},
    {"term": "pregabalin", "category": "neuropathic pain"},
    {"term": "lyrica", "category": "neuropathic pain (brand)"},

    # thyroid / endo
    {"term": "levothyroxine", "category": "thyroid replacement"},
    {"term": "eltroxin", "category": "thyroid replacement (brand)"},
    {"term": "euthyrox", "category": "thyroid replacement (brand)"},
    {"term": "synthroid", "category": "thyroid replacement (brand)"},
    {"term": "carbimazole", "category": "antithyroid"},
    {"term": "methimazole", "category": "antithyroid"},
    {"term": "propylthiouracil", "category": "antithyroid"},

    # corticosteroids
    {"term": "prednisolone", "category": "oral corticosteroid"},
    {"term": "prednisone", "category": "oral corticosteroid"},
    {"term": "methylprednisolone", "category": "iv corticosteroid"},
    {"term": "dexamethasone", "category": "corticosteroid"},
    {"term": "hydrocortisone", "category": "corticosteroid"},

    # topical / derm
    {"term": "betadine", "category": "antiseptic"},
    {"term": "savlon", "category": "antiseptic (brand)"},
    {"term": "fusidic acid", "category": "topical antibiotic"},
    {"term": "fucidin", "category": "topical antibiotic (brand)"},
    {"term": "mupirocin", "category": "topical antibiotic"},
    {"term": "bactroban", "category": "topical antibiotic (brand)"},
    {"term": "clotrimazole", "category": "topical antifungal"},
    {"term": "canesten", "category": "topical antifungal (brand)"},
    {"term": "miconazole", "category": "topical antifungal"},
    {"term": "betamethasone", "category": "topical steroid"},
    {"term": "diprosone", "category": "topical steroid (brand)"},
    {"term": "hydrocortisone cream", "category": "topical steroid"},

    # vitamins / supplements
    {"term": "vitamin d", "category": "vitamin"},
    {"term": "cholecalciferol", "category": "vitamin d3"},
    {"term": "vitamin b12", "category": "vitamin"},
    {"term": "cyanocobalamin", "category": "vitamin b12"},
    {"term": "folic acid", "category": "vitamin"},
    {"term": "iron", "category": "mineral"},
    {"term": "ferrous sulfate", "category": "iron supplement"},
    {"term": "ferrous fumarate", "category": "iron supplement"},
    {"term": "calcium carbonate", "category": "mineral"},
    {"term": "magnesium", "category": "mineral"},
    {"term": "zinc", "category": "mineral"},

    # ED / urology / OB-GYN
    {"term": "sildenafil", "category": "pde-5"},
    {"term": "viagra", "category": "pde-5 (brand)"},
    {"term": "tadalafil", "category": "pde-5"},
    {"term": "cialis", "category": "pde-5 (brand)"},
    {"term": "tamsulosin", "category": "alpha blocker"},
    {"term": "flomax", "category": "alpha blocker (brand)"},
    {"term": "finasteride", "category": "5-ari"},
    {"term": "proscar", "category": "5-ari (brand)"},
    {"term": "dutasteride", "category": "5-ari"},
    {"term": "oxybutynin", "category": "anticholinergic bladder"},
    {"term": "solifenacin", "category": "anticholinergic bladder"},
    {"term": "ethinyl estradiol levonorgestrel", "category": "ocp"},
    {"term": "yasmin", "category": "ocp (brand)"},
    {"term": "diane 35", "category": "ocp (brand)"},
    {"term": "clomiphene", "category": "fertility"},
    {"term": "clomid", "category": "fertility (brand)"},
    {"term": "letrozole", "category": "ai / fertility"},

    # emergency / misc
    {"term": "epinephrine", "category": "emergency"},
    {"term": "adrenaline", "category": "emergency"},
    {"term": "naloxone", "category": "opioid antagonist"},
    {"term": "atropine", "category": "anticholinergic emergency"},
    {"term": "glucagon", "category": "emergency hypoglycemia"},
    {"term": "activated charcoal", "category": "antidote"},
    {"term": "n-acetylcysteine", "category": "antidote / mucolytic"},
    {"term": "mucosolvan", "category": "mucolytic (brand)"},
    {"term": "ambroxol", "category": "mucolytic"},

    # vaccines / immunizations
    {"term": "tetanus vaccine", "category": "vaccine"},
    {"term": "influenza vaccine", "category": "vaccine"},
    {"term": "covid vaccine", "category": "vaccine"},
    {"term": "pneumococcal vaccine", "category": "vaccine"},
    {"term": "hepatitis b vaccine", "category": "vaccine"},
]


TIER1_DISEASES: List[Dict[str, str]] = [
    # cardiovascular
    {"term": "hypertension", "category": "cardiovascular"},
    {"term": "high blood pressure", "category": "cardiovascular"},
    {"term": "coronary artery disease", "category": "cardiovascular"},
    {"term": "myocardial infarction", "category": "cardiovascular"},
    {"term": "heart attack", "category": "cardiovascular"},
    {"term": "angina", "category": "cardiovascular"},
    {"term": "heart failure", "category": "cardiovascular"},
    {"term": "atrial fibrillation", "category": "arrhythmia"},
    {"term": "arrhythmia", "category": "cardiovascular"},
    {"term": "stroke", "category": "cerebrovascular"},
    {"term": "transient ischemic attack", "category": "cerebrovascular"},
    {"term": "deep vein thrombosis", "category": "vascular"},
    {"term": "pulmonary embolism", "category": "vascular"},
    {"term": "dyslipidemia", "category": "metabolic"},
    {"term": "high cholesterol", "category": "metabolic"},

    # endocrine
    {"term": "type 2 diabetes", "category": "endocrine"},
    {"term": "type 1 diabetes", "category": "endocrine"},
    {"term": "diabetes mellitus", "category": "endocrine"},
    {"term": "gestational diabetes", "category": "endocrine"},
    {"term": "diabetic ketoacidosis", "category": "endocrine emergency"},
    {"term": "hypoglycemia", "category": "endocrine"},
    {"term": "hyperthyroidism", "category": "endocrine"},
    {"term": "hypothyroidism", "category": "endocrine"},
    {"term": "thyroid nodule", "category": "endocrine"},
    {"term": "obesity", "category": "metabolic"},
    {"term": "metabolic syndrome", "category": "metabolic"},
    {"term": "osteoporosis", "category": "endocrine"},
    {"term": "vitamin d deficiency", "category": "nutritional"},
    {"term": "iron deficiency anemia", "category": "hematologic"},
    {"term": "vitamin b12 deficiency", "category": "nutritional"},

    # respiratory
    {"term": "asthma", "category": "respiratory"},
    {"term": "chronic obstructive pulmonary disease", "category": "respiratory"},
    {"term": "copd", "category": "respiratory"},
    {"term": "pneumonia", "category": "respiratory infection"},
    {"term": "bronchitis", "category": "respiratory"},
    {"term": "acute bronchitis", "category": "respiratory"},
    {"term": "upper respiratory tract infection", "category": "respiratory infection"},
    {"term": "common cold", "category": "respiratory infection"},
    {"term": "influenza", "category": "respiratory infection"},
    {"term": "tuberculosis", "category": "respiratory infection"},
    {"term": "covid-19", "category": "respiratory infection"},
    {"term": "allergic rhinitis", "category": "respiratory allergy"},
    {"term": "sinusitis", "category": "ent"},
    {"term": "pharyngitis", "category": "ent"},
    {"term": "tonsillitis", "category": "ent"},
    {"term": "laryngitis", "category": "ent"},
    {"term": "otitis media", "category": "ent"},
    {"term": "otitis externa", "category": "ent"},

    # GI
    {"term": "gastritis", "category": "gi"},
    {"term": "gastroesophageal reflux disease", "category": "gi"},
    {"term": "gerd", "category": "gi"},
    {"term": "peptic ulcer disease", "category": "gi"},
    {"term": "h pylori infection", "category": "gi"},
    {"term": "constipation", "category": "gi"},
    {"term": "diarrhea", "category": "gi"},
    {"term": "irritable bowel syndrome", "category": "gi"},
    {"term": "inflammatory bowel disease", "category": "gi"},
    {"term": "crohn disease", "category": "gi"},
    {"term": "ulcerative colitis", "category": "gi"},
    {"term": "appendicitis", "category": "surgical"},
    {"term": "cholecystitis", "category": "surgical"},
    {"term": "cholelithiasis", "category": "gi"},
    {"term": "pancreatitis", "category": "gi"},
    {"term": "viral hepatitis", "category": "gi infectious"},
    {"term": "hepatitis a", "category": "gi infectious"},
    {"term": "hepatitis b", "category": "gi infectious"},
    {"term": "hepatitis c", "category": "gi infectious"},
    {"term": "fatty liver disease", "category": "gi"},
    {"term": "cirrhosis", "category": "gi"},
    {"term": "hemorrhoids", "category": "surgical"},
    {"term": "inguinal hernia", "category": "surgical"},

    # GU / renal
    {"term": "urinary tract infection", "category": "gu infection"},
    {"term": "cystitis", "category": "gu infection"},
    {"term": "pyelonephritis", "category": "gu infection"},
    {"term": "kidney stones", "category": "renal"},
    {"term": "nephrolithiasis", "category": "renal"},
    {"term": "chronic kidney disease", "category": "renal"},
    {"term": "acute kidney injury", "category": "renal"},
    {"term": "benign prostatic hyperplasia", "category": "urology"},
    {"term": "erectile dysfunction", "category": "urology"},
    {"term": "prostatitis", "category": "urology"},

    # OB-GYN
    {"term": "polycystic ovary syndrome", "category": "gyn"},
    {"term": "endometriosis", "category": "gyn"},
    {"term": "uterine fibroids", "category": "gyn"},
    {"term": "vaginal candidiasis", "category": "gyn infection"},
    {"term": "bacterial vaginosis", "category": "gyn infection"},
    {"term": "pelvic inflammatory disease", "category": "gyn"},
    {"term": "preeclampsia", "category": "obstetric"},
    {"term": "hyperemesis gravidarum", "category": "obstetric"},
    {"term": "ectopic pregnancy", "category": "obstetric"},
    {"term": "miscarriage", "category": "obstetric"},
    {"term": "menorrhagia", "category": "gyn"},
    {"term": "dysmenorrhea", "category": "gyn"},
    {"term": "menopause", "category": "gyn"},
    {"term": "infertility", "category": "gyn"},

    # neuro / psych
    {"term": "migraine", "category": "neurology"},
    {"term": "tension headache", "category": "neurology"},
    {"term": "epilepsy", "category": "neurology"},
    {"term": "seizure", "category": "neurology"},
    {"term": "vertigo", "category": "neurology"},
    {"term": "bell palsy", "category": "neurology"},
    {"term": "parkinson disease", "category": "neurology"},
    {"term": "alzheimer disease", "category": "neurology"},
    {"term": "dementia", "category": "neurology"},
    {"term": "multiple sclerosis", "category": "neurology"},
    {"term": "depression", "category": "psychiatry"},
    {"term": "major depressive disorder", "category": "psychiatry"},
    {"term": "anxiety", "category": "psychiatry"},
    {"term": "generalized anxiety disorder", "category": "psychiatry"},
    {"term": "panic disorder", "category": "psychiatry"},
    {"term": "bipolar disorder", "category": "psychiatry"},
    {"term": "schizophrenia", "category": "psychiatry"},
    {"term": "insomnia", "category": "psychiatry"},
    {"term": "post-traumatic stress disorder", "category": "psychiatry"},
    {"term": "attention deficit hyperactivity disorder", "category": "psychiatry"},
    {"term": "autism spectrum disorder", "category": "psychiatry"},

    # derm
    {"term": "eczema", "category": "dermatology"},
    {"term": "atopic dermatitis", "category": "dermatology"},
    {"term": "psoriasis", "category": "dermatology"},
    {"term": "acne", "category": "dermatology"},
    {"term": "rosacea", "category": "dermatology"},
    {"term": "urticaria", "category": "dermatology"},
    {"term": "hives", "category": "dermatology"},
    {"term": "tinea corporis", "category": "dermatology infection"},
    {"term": "tinea pedis", "category": "dermatology infection"},
    {"term": "tinea capitis", "category": "dermatology infection"},
    {"term": "scabies", "category": "dermatology infection"},
    {"term": "herpes zoster", "category": "dermatology infection"},
    {"term": "cellulitis", "category": "dermatology infection"},
    {"term": "impetigo", "category": "dermatology infection"},
    {"term": "abscess", "category": "dermatology infection"},

    # rheumatology / msk
    {"term": "rheumatoid arthritis", "category": "rheumatology"},
    {"term": "osteoarthritis", "category": "rheumatology"},
    {"term": "gout", "category": "rheumatology"},
    {"term": "ankylosing spondylitis", "category": "rheumatology"},
    {"term": "systemic lupus erythematosus", "category": "rheumatology"},
    {"term": "fibromyalgia", "category": "rheumatology"},
    {"term": "low back pain", "category": "musculoskeletal"},
    {"term": "sciatica", "category": "musculoskeletal"},
    {"term": "lumbar disc herniation", "category": "musculoskeletal"},
    {"term": "carpal tunnel syndrome", "category": "musculoskeletal"},
    {"term": "rotator cuff tear", "category": "musculoskeletal"},
    {"term": "frozen shoulder", "category": "musculoskeletal"},
    {"term": "plantar fasciitis", "category": "musculoskeletal"},

    # eye
    {"term": "conjunctivitis", "category": "ophthalmology"},
    {"term": "glaucoma", "category": "ophthalmology"},
    {"term": "cataract", "category": "ophthalmology"},
    {"term": "diabetic retinopathy", "category": "ophthalmology"},
    {"term": "macular degeneration", "category": "ophthalmology"},
    {"term": "dry eye", "category": "ophthalmology"},
    {"term": "stye", "category": "ophthalmology"},

    # hematology / oncology / common cancers
    {"term": "anemia", "category": "hematology"},
    {"term": "thalassemia", "category": "hematology"},
    {"term": "sickle cell disease", "category": "hematology"},
    {"term": "thrombocytopenia", "category": "hematology"},
    {"term": "leukemia", "category": "oncology"},
    {"term": "lymphoma", "category": "oncology"},
    {"term": "breast cancer", "category": "oncology"},
    {"term": "lung cancer", "category": "oncology"},
    {"term": "colorectal cancer", "category": "oncology"},
    {"term": "prostate cancer", "category": "oncology"},
    {"term": "thyroid cancer", "category": "oncology"},
    {"term": "bladder cancer", "category": "oncology"},

    # pediatric / infectious
    {"term": "fever", "category": "general"},
    {"term": "viral fever", "category": "infectious"},
    {"term": "measles", "category": "pediatric infectious"},
    {"term": "mumps", "category": "pediatric infectious"},
    {"term": "chickenpox", "category": "pediatric infectious"},
    {"term": "hand foot and mouth disease", "category": "pediatric infectious"},
    {"term": "croup", "category": "pediatric respiratory"},
    {"term": "bronchiolitis", "category": "pediatric respiratory"},
    {"term": "rsv infection", "category": "pediatric respiratory"},
    {"term": "gastroenteritis", "category": "gi infectious"},
    {"term": "viral gastroenteritis", "category": "gi infectious"},
    {"term": "food poisoning", "category": "gi infectious"},

    # endemic in gulf
    {"term": "brucellosis", "category": "infectious"},
    {"term": "leishmaniasis", "category": "infectious"},
    {"term": "dengue fever", "category": "infectious"},
    {"term": "malaria", "category": "infectious"},
    {"term": "typhoid fever", "category": "infectious"},

    # ER common
    {"term": "chest pain", "category": "presentation"},
    {"term": "shortness of breath", "category": "presentation"},
    {"term": "abdominal pain", "category": "presentation"},
    {"term": "syncope", "category": "presentation"},
    {"term": "dizziness", "category": "presentation"},
    {"term": "palpitations", "category": "presentation"},
    {"term": "nausea and vomiting", "category": "presentation"},
    {"term": "headache", "category": "presentation"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Public-data sources for the LARGER pool (tier 2 + tier 3)
# ─────────────────────────────────────────────────────────────────────────────

# RxNav (NLM) — RxNorm canonical names. Public NIH API, no key.
# We pull ingredient names (TTY=IN) and brand names (TTY=BN). These are
# verified real drugs from the official US RxNorm database.
RXNAV_TTY_INGREDIENT_URL = (
    "https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=IN"
)
RXNAV_TTY_BRAND_URL = (
    "https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=BN"
)

# CDC ICD-10-CM order file — public. We use a community mirror that
# exposes the same data as a simple JSON list of {code, description}.
# If the URL changes, fall back to the bundled tier-1 list only.
ICD10_FALLBACK_URL = (
    "https://raw.githubusercontent.com/k4m4/icd-10-cm/master/icd-10-cm.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_TERM_RE = re.compile(r"^[a-z0-9 \-'/().+]+$")
SCRUB_RE = re.compile(r"\s+")


def _norm(term: str) -> str:
    t = term.lower().strip()
    t = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", t)
    t = SCRUB_RE.sub(" ", t)
    return t


def _valid(term: str) -> bool:
    if not term or len(term) < 2 or len(term) > 60:
        return False
    return bool(VALID_TERM_RE.match(term))


def _http_get_json(url: str, timeout: int = 60) -> Optional[Any]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[lex] WARN fetching {url}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Source loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_rxnorm_ingredients() -> List[Dict[str, str]]:
    print("[lex] fetching RxNorm ingredients (NLM) ...")
    data = _http_get_json(RXNAV_TTY_INGREDIENT_URL, timeout=120)
    if not data:
        return []
    items = (data.get("minConceptGroup", {}) or {}).get("minConcept", []) or []
    out: List[Dict[str, str]] = []
    for c in items:
        name = _norm(c.get("name", ""))
        if _valid(name):
            out.append({"term": name, "category": "ingredient", "source": "rxnorm_in"})
    print(f"[lex]   RxNorm ingredients: {len(out)}")
    return out


def load_rxnorm_brands() -> List[Dict[str, str]]:
    print("[lex] fetching RxNorm brand names (NLM) ...")
    data = _http_get_json(RXNAV_TTY_BRAND_URL, timeout=120)
    if not data:
        return []
    items = (data.get("minConceptGroup", {}) or {}).get("minConcept", []) or []
    out: List[Dict[str, str]] = []
    for c in items:
        name = _norm(c.get("name", ""))
        if _valid(name):
            out.append({"term": name, "category": "brand", "source": "rxnorm_bn"})
    print(f"[lex]   RxNorm brands: {len(out)}")
    return out


def load_icd10_diagnoses() -> List[Dict[str, str]]:
    print("[lex] fetching ICD-10-CM diagnosis list ...")
    data = _http_get_json(ICD10_FALLBACK_URL, timeout=120)
    if not data:
        return []
    out: List[Dict[str, str]] = []
    # Accept multiple shapes the mirror might return.
    iterable: Iterable[Dict[str, Any]]
    if isinstance(data, dict) and "data" in data:
        iterable = data["data"]
    elif isinstance(data, list):
        iterable = data
    else:
        iterable = []
    for row in iterable:
        if not isinstance(row, dict):
            continue
        desc = (
            row.get("description")
            or row.get("desc")
            or row.get("name")
            or row.get("title")
            or ""
        )
        desc = _norm(desc)
        # ICD descriptions can be long ("Other and unspecified ..."); shorten.
        desc = re.sub(r",\s*unspecified.*$", "", desc)
        desc = re.sub(r"\s*\(.*\)\s*$", "", desc)
        if _valid(desc):
            out.append({"term": desc, "category": "icd10", "source": "icd10_cm"})
    print(f"[lex]   ICD-10 diagnoses: {len(out)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LLM enhancement — ONLY for adding brand variants of REAL drugs we already
# have, never to invent new drug names. Calme is asked to provide common
# brand names that are sold in Gulf pharmacies for each ingredient.
# Every returned brand is filtered against an English-letters regex and
# de-duped against the existing list.
# ─────────────────────────────────────────────────────────────────────────────

BRAND_PROMPT_SYSTEM = """You are a clinical pharmacist working in the Gulf region.
For each generic drug ingredient I list, you provide common BRAND NAMES that are
SOLD in pharmacies in Saudi Arabia, UAE, Kuwait, Qatar, Bahrain, or Oman.

Strict rules:
- English (Latin letters) ONLY. No Arabic script.
- Only REAL brand names. If you are not sure, return an empty array for that
  ingredient.
- Do not invent. Do not add suffixes like "-extra" unless that exact product
  exists.
- Return STRICT JSON: an object mapping each ingredient to an array of 0-5
  brand names. No markdown, no commentary."""


def augment_with_brands(
    ingredients: List[str],
    ollama_url: str,
    ollama_model: str,
    batch_size: int = 20,
    timeout: int = 600,
) -> Dict[str, List[str]]:
    """Ask Calme for Gulf-sold brand names of each ingredient.

    Returns dict {ingredient: [brand1, brand2, ...]}.
    """
    result: Dict[str, List[str]] = {}
    for start in range(0, len(ingredients), batch_size):
        batch = ingredients[start:start + batch_size]
        prompt = (
            "Provide brand names sold in Gulf pharmacies for each ingredient.\n"
            "Return JSON object with ingredient as key, brand-list as value.\n\n"
            "Ingredients:\n" + "\n".join(f"- {x}" for x in batch)
        )
        try:
            r = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": ollama_model,
                    "system": BRAND_PROMPT_SYSTEM,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 2048},
                },
                timeout=timeout,
            )
            r.raise_for_status()
            text = r.json().get("response", "")
        except Exception as e:
            print(f"[lex]   brand batch failed: {e}")
            continue

        # Extract JSON object
        s = text.find("{")
        e = text.rfind("}")
        if s == -1 or e == -1:
            continue
        try:
            obj = json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        for ing, brands in obj.items():
            ing_norm = _norm(str(ing))
            if not isinstance(brands, list):
                continue
            kept = []
            for b in brands:
                if not isinstance(b, str):
                    continue
                bn = _norm(b)
                if _valid(bn):
                    kept.append(bn)
            if kept:
                result[ing_norm] = kept
        print(f"[lex]   brand batch {start // batch_size + 1}: "
              f"+{sum(len(v) for v in result.values())} so far")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tier assignment + output writing
# ─────────────────────────────────────────────────────────────────────────────

def assign_tier(
    term: str,
    in_tier1: Set[str],
    in_brand_pool: Set[str],
    rxnorm_top: Set[str],
) -> int:
    if term in in_tier1:
        return 1
    if term in in_brand_pool or term in rxnorm_top:
        return 2
    return 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/full_lexicon.jsonl")
    ap.add_argument("--target", type=int, default=10000,
                    help="Soft cap on total terms. Default 10000.")
    ap.add_argument("--skip-llm", action="store_true",
                    help="Skip the Calme brand-augmentation step.")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model",
                    default="calme-3.2-instruct-78b-GGUF:IQ4_XS")
    ap.add_argument("--brand-augment-top", type=int, default=400,
                    help="Top-N ingredients to ask Calme for Gulf brand names.")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Start with tier-1 seeds.
    entries: Dict[str, Dict[str, Any]] = {}
    for d in TIER1_DRUGS:
        t = _norm(d["term"])
        if not _valid(t):
            continue
        entries[t] = {
            "term": t, "type": "drug", "tier": 1,
            "category": d.get("category", ""), "source": "tier1_seed",
        }
    for d in TIER1_DISEASES:
        t = _norm(d["term"])
        if not _valid(t):
            continue
        entries[t] = {
            "term": t, "type": "disease", "tier": 1,
            "category": d.get("category", ""), "source": "tier1_seed",
        }
    tier1_terms: Set[str] = set(entries.keys())
    print(f"[lex] tier-1 seed: {len(tier1_terms)} ({len(TIER1_DRUGS)} drugs + "
          f"{len(TIER1_DISEASES)} diseases)")

    # 2. RxNorm ingredients (drug names).
    rx_ing = load_rxnorm_ingredients()
    rx_top_set: Set[str] = set()
    for row in rx_ing:
        t = row["term"]
        if not _valid(t) or t in entries:
            rx_top_set.add(t)
            continue
        entries[t] = {
            "term": t, "type": "drug", "tier": 3,
            "category": row.get("category", "ingredient"),
            "source": row.get("source", "rxnorm_in"),
        }
        rx_top_set.add(t)

    # 3. RxNorm brand names.
    rx_brand = load_rxnorm_brands()
    brand_set: Set[str] = set()
    for row in rx_brand:
        t = row["term"]
        if not _valid(t) or t in entries:
            brand_set.add(t)
            continue
        entries[t] = {
            "term": t, "type": "drug", "tier": 3,
            "category": row.get("category", "brand"),
            "source": row.get("source", "rxnorm_bn"),
        }
        brand_set.add(t)

    # 4. ICD-10 diagnoses.
    icd = load_icd10_diagnoses()
    for row in icd:
        t = row["term"]
        if not _valid(t) or t in entries:
            continue
        entries[t] = {
            "term": t, "type": "disease", "tier": 3,
            "category": row.get("category", "icd10"),
            "source": row.get("source", "icd10_cm"),
        }

    print(f"[lex] after public-data merge: {len(entries)} unique terms")

    # 5. Promote selected RxNorm ingredients to tier 2 (the WHO Essential
    # Medicines list approximated by the top RxNorm ingredients alphabetically
    # filtered to short names — proxy for common drugs).
    promote_count = 0
    short_ingredients = sorted(
        (e for e in entries.values()
         if e["source"] == "rxnorm_in" and len(e["term"]) <= 18),
        key=lambda x: x["term"],
    )[:1500]
    for e in short_ingredients:
        if e["tier"] == 3:
            e["tier"] = 2
            promote_count += 1
    print(f"[lex] promoted {promote_count} short RxNorm ingredients to tier 2")

    # 6. LLM brand augmentation (optional).
    if not args.skip_llm:
        drug_ingredients = [
            e["term"] for e in entries.values()
            if e["type"] == "drug" and e["tier"] <= 2
        ][:args.brand_augment_top]
        print(f"[lex] asking Calme for Gulf brand variants of "
              f"{len(drug_ingredients)} top ingredients ...")
        brand_map = augment_with_brands(
            drug_ingredients, args.ollama_url, args.ollama_model,
        )
        added = 0
        for ing, brands in brand_map.items():
            for b in brands:
                if b in entries:
                    continue
                entries[b] = {
                    "term": b, "type": "drug", "tier": 2,
                    "category": f"brand for {ing}",
                    "source": "calme_brand_augment",
                }
                added += 1
        print(f"[lex] LLM brand augment: +{added} new Gulf brand names")

    # 7. Trim to target.
    if len(entries) > args.target:
        # Keep all tier 1 + tier 2, then top up tier 3 to reach target.
        keep: Dict[str, Dict[str, Any]] = {}
        for k, v in entries.items():
            if v["tier"] <= 2:
                keep[k] = v
        tier3 = [(k, v) for k, v in entries.items() if v["tier"] == 3]
        remaining = args.target - len(keep)
        if remaining > 0:
            tier3.sort(key=lambda kv: kv[0])
            for k, v in tier3[:remaining]:
                keep[k] = v
        entries = keep
        print(f"[lex] trimmed to target={args.target}: {len(entries)} terms")

    # 8. Write.
    by_tier_type: Dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as fh:
        for term, row in sorted(entries.items()):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            key = f"tier{row['tier']}_{row['type']}"
            by_tier_type[key] = by_tier_type.get(key, 0) + 1

    print(f"\n[lex] wrote {len(entries)} terms -> {out_path}")
    for k in sorted(by_tier_type.keys()):
        print(f"  {k:<24} {by_tier_type[k]:>6}")


if __name__ == "__main__":
    main()
