"""Tests for the Arabic correction layer — false-positive preservation, Arabic
spelling correction, Arabic→English transliteration, and English smoke tests.

Four groups:
  1. Arabic false-positive preservation — clinical words that must never change
  2. Arabic spelling correction — ASR errors fixed by arabic_spelling.py
  3. Arabic→English transliteration — Arabic script to English medical terms
  4. English smoke tests — sanity check that English path still works
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_llm() -> None:
    """Disable LLM DETECT for all tests."""
    saved = getattr(app.main, "USE_LLM", False)
    app.main.USE_LLM = False
    yield
    app.main.USE_LLM = saved


@pytest.fixture(autouse=True)
def _reset_corrector() -> None:
    """Clear the module-level cached corrector so each test starts fresh."""
    app.main._TEXT_CORRECTOR = None
    yield


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient pointing at the real app."""
    with TestClient(app.main.app) as c:
        yield c


# ===========================================================================
# Group 1 — Arabic false-positive preservation (highest priority)
# ===========================================================================
# These words were producing false corrections before the filler set
# expansion and _bad_span_boundary fix. Every one must pass through
# unchanged.


class TestArabicFalsePositivePreservation:
    """Clinical Arabic words that must never be changed (individual tests
    so CI output shows exactly which word regresses)."""

    def test_candidate_gated_replacement_bug(self) -> None:
        """Regression test for the candidate-gated replacement safety bug.

        Uses direct MedicalCorrector (not API) to avoid server-caching issues.
        This exact input previously produced 5 forced false-positive
        corrections where normal Arabic words were replaced by random
        drug names via coincidental consonant-skeleton overlap.

        Required behavior:
          - bisoprlol → bisoprolol          (genuine misspelling, keep)
          - hypokalimia → hypokalemia        (genuine misspelling, keep)
          - esomeprazol → esomeprazole       (genuine misspelling, keep)
          - فيبريليشن → left UNCHANGED       (Arabic word, not replaced)
          - ومسجل → left UNCHANGED           (Arabic word, not replaced)
          - بشكل → left UNCHANGED            (Arabic word, not replaced)
          - الفحوصات → left UNCHANGED        (Arabic word, not replaced)
          - وصفت → left UNCHANGED            (Arabic word, not replaced)
          - No drug names leaked into Arabic words
        """
        from app.services.correction import MedicalCorrector

        text = (
            "المريض عنده اتريل فيبريليشن ومسجل عنده من زمان، "
            "وياخذ bisoprlol و digoxine بشكل يومي. "
            "اليوم جا يشتكي من تشيست بين وصعال جاف من ثلاث ايام. "
            "الفحوصات بينت عنده hypokalimia خفيف. "
            "وصفت له esomeprazol للمعده وطلبت اعادة تخطيط القلب بعد اسبوع."
        )

        corrector = MedicalCorrector()
        result = corrector.correct_transcript(text, use_llm=False)
        ct = result["corrected_text"]
        ct_lower = ct.lower()

        # Genuine corrections must still work
        assert "bisoprolol" in ct, f"bisoprolol should be corrected, got: {ct}"
        assert "hypokalemia" in ct, f"hypokalemia should be corrected, got: {ct}"
        assert "esomeprazole" in ct, f"esomeprazole should be corrected, got: {ct}"

        # Arabic words must NOT be replaced by drug names
        assert "فيبريليشن" in ct, f"فيبريليشن must not be replaced, got: {ct}"
        assert "ومسجل" in ct, f"ومسجل must not be replaced, got: {ct}"
        assert "بشكل" in ct, f"بشكل must not be replaced, got: {ct}"
        assert "الفحوصات" in ct, f"الفحوصات must not be replaced, got: {ct}"
        assert "وصفت" in ct, f"وصفت must not be replaced, got: {ct}"

        # Verify no drug names leaked into Arabic words
        assert "paclitaxel" not in ct_lower, f"paclitaxel leaked into text"
        assert "amoxicillin" not in ct_lower, f"amoxicillin leaked into text"
        assert "systemic lupus" not in ct_lower, f"systemic lupus leaked into text"
        assert "bevacizumab" not in ct_lower, f"bevacizumab leaked into text"
        assert "saturation" not in ct_lower, f"saturation leaked into text"
    """Clinical Arabic words that must never be changed (individual tests
    so CI output shows exactly which word regresses)."""

    def test_word_حضر(self, client: TestClient) -> None:
        """حضر was producing false correction to 'heart rate' before filler set
        expansion — حضر shares skeleton consonants with 'heart rate'."""
        resp = client.post("/api/correct", json={"text": "حضر المريض اليوم"})
        data = resp.json()
        assert "حضر" in data["corrected_text"]

    def test_word_بسبب(self, client: TestClient) -> None:
        """بسبب was producing false correction to 'shortness of breath' before
        filler set expansion — بسبب shares skeleton with 'sob'."""
        resp = client.post("/api/correct", json={"text": "دخل المستشفى بسبب الألم"})
        data = resp.json()
        assert "بسبب" in data["corrected_text"]

    def test_word_ادخال(self, client: TestClient) -> None:
        """ادخال (admission) was producing false correction to 'diabetic
        ketoacidosis' — skeleton 'adkhal' loosely matches 'dka'."""
        resp = client.post("/api/correct", json={"text": "تم ادخال المريض"})
        data = resp.json()
        assert "ادخال" in data["corrected_text"]

    def test_word_نبضة(self, client: TestClient) -> None:
        """نبضة (pulse/beat) was producing false correction to 'inflammatory
        bowel disease' — skeleton 'nbd' matches 'ibd'."""
        resp = client.post("/api/correct", json={"text": "معدل النبضة طبيعي"})
        data = resp.json()
        assert "نبضة" in data["corrected_text"]

    def test_word_أعراض(self, client: TestClient) -> None:
        """أعراض (symptoms) was being 'corrected' to اعراض (wrong direction
        — hamza removal is not a valid fix in MSA)."""
        resp = client.post("/api/correct", json={"text": "يعاني من أعراض شديدة"})
        data = resp.json()
        assert "أعراض" in data["corrected_text"]

    def test_word_إعطاء(self, client: TestClient) -> None:
        """إعطاء (giving) was being 'corrected' to اعطاء (wrong direction
        — hamza under alif is correct in this verb form)."""
        resp = client.post("/api/correct", json={"text": "تم إعطاء العلاج للمريض"})
        data = resp.json()
        assert "إعطاء" in data["corrected_text"]

    def test_word_ألم(self, client: TestClient) -> None:
        """ألم (pain) should never be corrected — it's a standard Arabic word."""
        resp = client.post("/api/correct", json={"text": "يشكو من ألم في الصدر"})
        data = resp.json()
        assert "ألم" in data["corrected_text"]

    def test_word_تنفس(self, client: TestClient) -> None:
        """تنفس (breathing) was producing false correction to 'tenecteplase'
        before filler set expansion."""
        resp = client.post("/api/correct", json={"text": "صعوبة في تنفس"})
        data = resp.json()
        assert "تنفس" in data["corrected_text"]

    def test_word_ضغط(self, client: TestClient) -> None:
        """ضغط (pressure) must never be changed — it's a vital sign term."""
        resp = client.post("/api/correct", json={"text": "ارتفاع ضغط الدم"})
        data = resp.json()
        assert "ضغط" in data["corrected_text"]

    def test_word_سكر(self, client: TestClient) -> None:
        """سكر (sugar/diabetes) must never be changed."""
        resp = client.post("/api/correct", json={"text": "مستوى سكر الدم مرتفع"})
        data = resp.json()
        assert "سكر" in data["corrected_text"]

    def test_word_قلب(self, client: TestClient) -> None:
        """قلب (heart) must never be changed — it's a basic anatomy term."""
        resp = client.post("/api/correct", json={"text": "ألم في القلب"})
        data = resp.json()
        assert "قلب" in data["corrected_text"]

    def test_word_رئة(self, client: TestClient) -> None:
        """رئة (lung) must never be changed."""
        resp = client.post("/api/correct", json={"text": "التهاب في الرئة"})
        data = resp.json()
        assert "رئة" in data["corrected_text"]

    def test_word_شديد(self, client: TestClient) -> None:
        """شديد (severe) must never be changed — common clinical adjective."""
        resp = client.post("/api/correct", json={"text": "ألم شديد في البطن"})
        data = resp.json()
        assert "شديد" in data["corrected_text"]

    def test_word_مزمن(self, client: TestClient) -> None:
        """مزمن (chronic) must never be changed."""
        resp = client.post("/api/correct", json={"text": "سعال مزمن منذ سنة"})
        data = resp.json()
        assert "مزمن" in data["corrected_text"]

    def test_word_التهاب(self, client: TestClient) -> None:
        """التهاب (inflammation) must never be changed."""
        resp = client.post("/api/correct", json={"text": "التهاب في الحلق"})
        data = resp.json()
        assert "التهاب" in data["corrected_text"]

    def test_word_يشتكي(self, client: TestClient) -> None:
        """يشتكي (complains of) was producing false correction to 'status
        epilepticus' or similar before filler set expansion."""
        resp = client.post("/api/correct", json={"text": "المريض يشتكي من صداع"})
        data = resp.json()
        assert "يشتكي" in data["corrected_text"]

    def test_word_يوجد(self, client: TestClient) -> None:
        """يوجد (there is/exists) must never be changed — common in reports."""
        resp = client.post("/api/correct", json={"text": "يوجد ألم في المفاصل"})
        data = resp.json()
        assert "يوجد" in data["corrected_text"]

    def test_word_تبين(self, client: TestClient) -> None:
        """تبين (it was found) was producing false corrections — skeleton
        'tbyn' loosely matches English terms via vowel-stripped comparison."""
        resp = client.post("/api/correct", json={"text": "تبين وجود كسر"})
        data = resp.json()
        assert "تبين" in data["corrected_text"]

    def test_word_أسبوع(self, client: TestClient) -> None:
        """أسبوع (week) must never be changed."""
        resp = client.post("/api/correct", json={"text": "سيبقى أسبوع في المستشفى"})
        data = resp.json()
        assert "أسبوع" in data["corrected_text"]

    def test_word_شهر(self, client: TestClient) -> None:
        """شهر (month) must never be changed."""
        resp = client.post("/api/correct", json={"text": "يأخذ العلاج منذ شهر"})
        data = resp.json()
        assert "شهر" in data["corrected_text"]


# ===========================================================================
# Group 2 — Arabic spelling correction
# ===========================================================================
# Known ASR Arabic misspellings that the pipeline must fix via
# arabic_spelling.py (phonetic merger map + explicit misspelling map).


class TestArabicSpellingCorrection:
    """Arabic→Arabic spelling corrections for Gulf ASR phonetic errors."""

    def test_sadau_to_sadau(self, client: TestClient) -> None:
        """سداع → صداع (س→ص substitution, common Gulf ASR error for 'headache')."""
        resp = client.post("/api/correct", json={"text": "يعاني من سداع حاد"})
        data = resp.json()
        assert "صداع" in data["corrected_text"]

    def test_aldght_to_aldght(self, client: TestClient) -> None:
        """الدغط → الضغط (د→ض substitution with ال prefix preserved)."""
        resp = client.post("/api/correct", json={"text": "ارتفاع الدغط"})
        data = resp.json()
        assert "الضغط" in data["corrected_text"]

    def test_alteb_to_alehtab(self, client: TestClient) -> None:
        """التهب → التهاب (هـ→ب substitution via explicit misspelling map)."""
        resp = client.post("/api/correct", json={"text": "التهب حاد في الرئة"})
        data = resp.json()
        assert "التهاب" in data["corrected_text"]

    def test_mrd_to_mryd(self, client: TestClient) -> None:
        """مرد → مريض (missing ي + د→ض via explicit misspelling map
        'مرد':'مريض')."""
        resp = client.post("/api/correct", json={"text": "المرد يشتكي من ألم"})
        data = resp.json()
        assert "المريض" in data["corrected_text"]

    def test_hsry_to_hstwy(self, client: TestClient) -> None:
        """هسري → هستوري (missing ت via explicit misspelling map
        'هسري':'هستوري')."""
        resp = client.post("/api/correct", json={"text": "عنده هسري مرض السكر"})
        data = resp.json()
        assert "هستوري" in data["corrected_text"]


# ===========================================================================
# Group 3 — Arabic→English transliteration
# ===========================================================================
# Arabic-script transliterations of English medical terms that must be
# converted to their canonical English form via consonant skeleton matching.


class TestArabicEnglishTransliteration:
    """Arabic-script medical transliterations → English canonical terms."""

    def test_history_arabic(self, client: TestClient) -> None:
        """هستوري (hstwry) → history — skeleton 'hstr' matches 'history'
        'hstr'."""
        resp = client.post("/api/correct", json={"text": "عنده هستوري مرض السكر"})
        data = resp.json()
        assert "history" in data["corrected_text"].lower()

    def test_diabetes_arabic(self, client: TestClient) -> None:
        """دايابيتس (dyabytes) → diabetes — skeleton 'dyabts' matches
        'diabetes' 'dbt'."""
        resp = client.post("/api/correct", json={"text": "يعاني من دايابيتس"})
        data = resp.json()
        assert "diabetes" in data["corrected_text"].lower()

    def test_blood_sugar_arabic(self, client: TestClient) -> None:
        """بلاد شوجر (blad shwjr) → blood sugar — multi-word phrase match
        via phonetic.py _MULTI_WORD_PHRASES."""
        resp = client.post("/api/correct", json={"text": "عنده بلاد شوجر مرتفع"})
        data = resp.json()
        assert "blood sugar" in data["corrected_text"].lower()

    def test_shortness_of_breath_arabic(self, client: TestClient) -> None:
        """شورتنس اوف بريث (shortness of breath) → shortness of breath
        — multi-word phrase match via phonetic.py."""
        resp = client.post("/api/correct", json={"text": "يعاني من شورتنس اوف بريث"})
        data = resp.json()
        assert "shortness of breath" in data["corrected_text"].lower()

    def test_aspirin_arabic(self, client: TestClient) -> None:
        """أسبرين (asbryn) → aspirin — skeleton 'sbrn' matches 'aspirin'
        'sprn'.

        NOTE: Uses أسبرين (with hamza) rather than اسبرين (without hamza)
        because the Arabic spelling corrector would intercept اسبرين,
        'fix' the hamza (ا→أ), and return أسبرين without ever reaching
        the English transliteration path. This test exercises the English
        matching path directly."""
        resp = client.post("/api/correct", json={"text": "يأخذ أسبرين يوميا"})
        data = resp.json()
        assert "aspirin" in data["corrected_text"].lower()


# ===========================================================================
# Group 4 — English medical misspellings (smoke tests)
# ===========================================================================
# Representative cases from the existing test suite to catch regressions
# at the integration level.  Not comprehensive — just enough to detect
# if the Arabic changes broke the English correction path.


class TestEnglishSmokeTests:
    """English correction path — smoke tests to verify no regression."""

    def test_amoxicilin_to_amoxicillin(self, client: TestClient) -> None:
        """Basic English deterministic correction still works."""
        resp = client.post(
            "/api/correct", json={"text": "Take amoxicilin 500 mg."}
        )
        data = resp.json()
        assert "amoxicillin" in data["corrected_text"].lower()

    def test_clopidogr_to_clopidogrel(self, client: TestClient) -> None:
        """Another basic English correction path."""
        resp = client.post(
            "/api/correct", json={"text": "Clopidogr 75 mg daily."}
        )
        data = resp.json()
        assert "clopidogrel" in data["corrected_text"].lower()

    def test_multiple_misspellings_all_corrected(self, client: TestClient) -> None:
        """Multiple English misspellings in one sentence: each fixed."""
        resp = client.post(
            "/api/correct",
            json={"text": "Needs clopidogr and amoxicilin."},
        )
        data = resp.json()
        ct = data["corrected_text"].lower()
        assert "clopidogrel" in ct
        assert "amoxicillin" in ct
        assert len(data["auto_corrections"]) >= 2

    def test_uppercase_still_corrects(self, client: TestClient) -> None:
        """Uppercase input still triggers corrections."""
        resp = client.post(
            "/api/correct", json={"text": "TAKE CLOPIDOGR DAILY"}
        )
        data = resp.json()
        assert "clopidogrel" in data["corrected_text"].lower()

    def test_already_correct_english(self, client: TestClient) -> None:
        """Clean English text should remain unchanged with zero corrections."""
        text = "The patient is stable and resting comfortably."
        resp = client.post("/api/correct", json={"text": text})
        data = resp.json()
        assert data["corrected_text"] == text
        assert len(data["auto_corrections"]) == 0
