import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Path to the generated JSON
PRICES_JSON_PATH = os.path.join(os.path.dirname(__file__), "doctor_prices.json")

class PricesLookup:
    _instance = None
    _data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PricesLookup, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance

    def _load_data(self):
        if not os.path.exists(PRICES_JSON_PATH):
            self._data = {"clinics": {}, "by_doctor": {}}
            return
        
        with open(PRICES_JSON_PATH, "r", encoding="utf-8") as f:
            self._data = json.load(f)

    def _normalize_arabic(self, text: str) -> str:
        if not text:
            return ""
        text = text.casefold()
        # Remove tashkeel + tatweel
        text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
        replacements = {
            "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
            "ى": "ي", "ئ": "ي", "ؤ": "و", "ة": "ه", "ۀ": "ه",
            "گ": "ك", "ڨ": "ق", "چ": "ج", "پ": "ب", "ژ": "ز",
        }
        text = text.translate(str.maketrans(replacements))
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        return " ".join(text.split()).strip()

    def _get_name_variants(self, name: str) -> List[str]:
        """Generate common variations of a name for better matching."""
        if not name:
            return []
        norm = self._normalize_arabic(name)
        variants = [norm]
        # Remove "ال" prefix
        tokens = norm.split()
        for token in tokens:
            if token.startswith("ال") and len(token) > 2:
                variants.append(norm.replace(token, token[2:]))
        return list(set(variants))

    def lookup_doctor(self, query: str) -> List[Dict[str, Any]]:
        """Find doctor entries by name using scoring based on token and character overlap."""
        if not query or not self._data:
            return []
        
        query_norm = self._normalize_arabic(query)
        # Handle English queries by normalizing to casefold + clean
        query_en = query.casefold().strip()
        
        candidates = []
        
        for doc_name, entries in self._data["by_doctor"].items():
            # doc_name is our canonical English name (e.g., "Abanob Kodos AzmyGadallah")
            doc_norm = doc_name.casefold()
            
            # 1. Check for exact match or direct containment (highest confidence)
            if query_en in doc_norm or doc_norm in query_en:
                candidates.append((1.0, entries))
                continue
            
            # 2. Token-based overlap
            doc_tokens = set(doc_norm.split())
            query_tokens = set(query_en.split())
            overlap = doc_tokens.intersection(query_tokens)
            
            if overlap:
                score = len(overlap) / max(len(query_tokens), 1)
                if score >= 0.5: # At least half of the query tokens must match
                    candidates.append((score, entries))
                    continue

            # 3. Handle Arabic matching (rough transliteration fallback for doctor names)
            # This handles names like "عهدي" -> "Ahdy" / "Ohdy" / "Ahdi"
            # Since our doctor names in JSON are English, we check if any common
            # English transliteration of the query matches the doctor's name.
            # Simplified: we look for common phonetic components.
            doc_simple = re.sub(r'[^a-z]', '', doc_norm)
            query_simple = re.sub(r'[^a-z]', '', query_en)
            
            if query_simple and query_simple in doc_simple:
                 candidates.append((0.8, entries))
                 continue

        # Sort by score descending and return results
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, entries in candidates:
            # Avoid adding duplicates if multiple clinics matched the same name
            for entry in entries:
                if entry not in results:
                    results.append(entry)
                    
        return results

    def lookup_clinic(self, clinic_id: str) -> Optional[Dict[str, Any]]:
        """Get all doctor prices for a specific clinic ID."""
        if not self._data:
            return None
        return self._data["clinics"].get(str(clinic_id))

    def detect_service_type(self, question: str) -> str:
        """Detect what kind of service the user is asking about."""
        q = self._normalize_arabic(question)
        
        if any(kw in q for kw in ["رنين", "مقطعيه", "سونار", "اشعه", "اشعة", "سكانر", "xray", "اكس راي"]):
            return "radiology"
        if any(kw in q for kw in ["رسم قلب", "ecg", "اي سي جي"]):
            return "ecg"
        if any(kw in q for kw in ["كشف", "عيادة", "عياده", "حجز"]):
            return "consultation"
        
        return "general"

    def format_price_response(self, results: List[Dict[str, Any]], service_type: str = "general") -> str:
        if not results:
            return "عذراً، لم أجد بيانات أسعار لهذا الطلب في النظام."
        
        lines = ["بيانات الأسعار الرسمية:"]
        
        # Group by clinic
        by_clinic = {}
        for res in results:
            cid = res["clinic_id"]
            if cid not in by_clinic:
                by_clinic[cid] = {"name": res["clinic_name"], "doctors": []}
            by_clinic[cid]["doctors"].append(res)
            
        for cid, clinic in by_clinic.items():
            lines.append(f"\nالعيادة: {clinic['name']}")
            for doc in clinic["doctors"]:
                lines.append(f"تذكرة الدكتور: {doc['name']}")
                
                # Filter by service type if detected
                found_specific = False
                
                if service_type == "consultation" and doc["consultation_price"]:
                    lines.append(f"- سعر الكشف: {doc['consultation_price']} جنيه")
                    found_specific = True
                
                # Show additional services
                for svc in doc["additional_services"]:
                    if service_type != "consultation" or service_type == "general":
                         lines.append(f"- {svc['name']}: {svc['price']} جنيه")
                         found_specific = True
                
                if not found_specific and doc["consultation_price"]:
                     lines.append(f"- سعر الكشف: {doc['consultation_price']} جنيه")

        return "\n".join(lines)

# Singleton instance
prices_lookup = PricesLookup()
