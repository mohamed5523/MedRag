import asyncio
import json
import urllib.request
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

# --- COPY FROM MCP/matching_engine.py ---
ARABIC_TASHKEEL_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
ARABIC_TATWEEL_PATTERN = re.compile(r"[\u0640]")

TITLE_STOPWORDS = {
    "دكتور", "دكتورة", "د.", "د", "د/‏", "د/‏.",
    "دكتور.", "دكتوره",
    "dr", "dr.", "doctor",
}

def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = ARABIC_TASHKEEL_PATTERN.sub("", text)
    text = ARABIC_TATWEEL_PATTERN.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ء", "")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_mixed_text(text: str) -> str:
    text = text or ""
    text = text.strip()
    text = normalize_arabic(text)
    text = text.lower()
    text = text.replace("؟", " ").replace("،", " ").replace("؛", " ").replace("ـ", " ")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

CLINIC_STOPWORDS = {
    "عيادة", "عياده", "عيادات", "قسم", "مركز", "مجمع",
    "دكتور", "دكتورة", "دكتوره", "د", "د.", "dr", "dr.", "doctor",
    "طيب", "طب", "أنا", "انا", "هو", "هي", "احنا", "من فضلك", "لو سمحت", "معلش", 
    "بتاع", "بتاعة", "عاوز", "عايز", "محتاج", "عاوزة", "عايزة", "محتاجة", "روحت", "أروح", "اروح", "اشوف", "أشوف", "يا", "يا دكتور",
    "و", "في", "ب", "بال", "ال",
    "النهارده", "نهارده", "اليوم", "بكره", "بكرة", "غدا", "غداً",
    "مين", "موجود", "موجودين", "متاح", "مواعيد", "سعر", "كشف",
    "حد", "أحد", "اي", "أي", "دلوقتي", "دلوقت", "دي", "حاليا", "حالياً", "حالا", "الآن", "الان", "ده", "هذا",
    "clinic", "department", "center",
}

def tokenize_clinic(text: str) -> List[str]:
    norm = normalize_mixed_text(text)
    if not norm:
        return []
    raw_tokens = norm.split()
    tokens: List[str] = []
    for tok in raw_tokens:
        if tok in CLINIC_STOPWORDS:
            continue
        if tok.startswith("ال") and len(tok) > 3:
            tok = tok[2:]
        if tok == "و":
            continue
        if tok.startswith("و") and len(tok) > 3:
            tok = tok[1:]
        if len(tok) > 1 and tok not in CLINIC_STOPWORDS:
            tokens.append(tok)
    return list(dict.fromkeys(tokens))

def compute_order_score(query_tokens: List[str], target_tokens: List[str]) -> float:
    if len(query_tokens) < 2 or len(target_tokens) == 0:
        return 0.0
    if target_tokens[:len(query_tokens)] == query_tokens:
        return 1.0
    try:
        positions = [target_tokens.index(t) for t in query_tokens]
        if positions == sorted(positions):
            return 0.85
    except ValueError:
        pass
    return 0.0

# --- COPY FROM MCP/clinic_server.py ---
class ClinicMatch(BaseModel):
    clinic_id: str
    clinic_name: str
    score: float
    token_overlap: float
    fuzzy_name_score: float
    order_score: float
    matched_tokens: List[str] = []

class ClinicRecord(BaseModel):
    clinic_id: str
    clinic_name: str
    norm_name: str
    tokens: List[str]

def _parse_and_preprocess_clinics(data: List[Dict[str, Any]]) -> List[ClinicRecord]:
    clinics: List[ClinicRecord] = []
    seen: set[str] = set()
    for clinic in data:
        clinic_id = str(clinic.get("clinicId", "")).strip()
        clinic_name = str(clinic.get("clinicName", "")).strip()
        if not clinic_id or clinic_id in seen:
            continue
        seen.add(clinic_id)
        clinics.append(
            ClinicRecord(
                clinic_id=clinic_id,
                clinic_name=clinic_name,
                norm_name=normalize_mixed_text(clinic_name),
                tokens=tokenize_clinic(clinic_name),
            )
        )
    return clinics

def _match_clinic_multi_token(
    query_tokens: List[str],
    clinics: List[ClinicRecord],
    top_k: int,
    min_score: float,
):
    q_string = " ".join(query_tokens)
    scored: List[ClinicMatch] = []
    for c in clinics:
        overlap_sum = 0.0
        matched_tokens: List[str] = []
        for qt in query_tokens:
            best_token_score = 0.0
            for ct in c.tokens:
                s = fuzz.ratio(qt, ct) / 100.0
                if s > best_token_score:
                    best_token_score = s
            overlap_sum += best_token_score
            if best_token_score >= 0.70:
                matched_tokens.append(qt)
        token_overlap = (overlap_sum / len(query_tokens)) if query_tokens else 0.0
        order_score = compute_order_score(query_tokens, c.tokens)
        full = c.norm_name or c.clinic_name
        fuzzy_name_score = fuzz.WRatio(q_string, full) / 100.0 if full else 0.0
        final_score = 0.45 * token_overlap + 0.20 * order_score + 0.35 * fuzzy_name_score
        if final_score < min_score:
            continue
        scored.append(
            ClinicMatch(
                clinic_id=c.clinic_id,
                clinic_name=c.clinic_name,
                score=round(final_score, 4),
                token_overlap=round(token_overlap, 4),
                fuzzy_name_score=round(fuzzy_name_score, 4),
                order_score=round(order_score, 4),
                matched_tokens=matched_tokens,
            )
        )
    scored.sort(key=lambda m: (m.score, len(m.matched_tokens)), reverse=True)
    return scored[:top_k]

async def test_ortho_match():
    u='http://41.32.47.162:9091/api/clinicProviderlist/'
    req=urllib.request.Request(u, headers={'Authorization':'Basic bWlsbGVuOm1pbGxlbkA0MzIx'})
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            data = res.get('data', res)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    clinics = _parse_and_preprocess_clinics(data)
    
    queries = ["عظام", "دكتور عظام", "عيادة عظام", "طيب أنا عاوز دكتور عظام"]
    for query in queries:
        query_tokens = tokenize_clinic(query)
        print(f"--- Testing Query: '{query}' ---")
        print(f"Tokens: {query_tokens}")
        
        matches = _match_clinic_multi_token(query_tokens, clinics, top_k=5, min_score=0.4) # Lower min_score to see what it's close to
        if not matches:
            print("NO MATCHES FOUND (even with min_score=0.4)")
        else:
            for m in matches:
                print(f"Match: {m.clinic_name} (ID: {m.clinic_id}) - Score: {m.score}")
                print(f"  Overlap: {m.token_overlap}, Fuzzy: {m.fuzzy_name_score}, Order: {m.order_score}")
                print(f"  Matched Tokens: {m.matched_tokens}")
        
        # Exact check for "عظام" clinic record
        ortho = next((c for c in clinics if c.clinic_id == '1119'), None)
        if ortho:
            print(f"Record for ID 1119: name='{ortho.clinic_name}', tokens={ortho.tokens}")
        else:
            print("Clinic ID 1119 not found in data!")

if __name__ == "__main__":
    asyncio.run(test_ortho_match())
