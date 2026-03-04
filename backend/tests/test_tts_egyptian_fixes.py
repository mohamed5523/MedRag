
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.tts_normalization import normalize_arabic_for_tts, normalize_medical_text, normalize_time_format

def test_egyptian_enhancements():
    # 1. Number pronunciation (No 'El' prefix for hours)
    # The time normalizer uses get_hour_word but it's internal to normalize_time_format
    # We test via normalize_time_format or full normalization if applicable
    
    # Test cases for numbers/time
    time_cases = [
        ("الساعة 11:00", "الساعة حداشر بالظبط"), # Should be "حداشر" not "الحداشر"
        ("الساعة 3:00", "الساعة تلاته بالظبط"), # Should be "تلاته" not "التلاته"
        ("الساعة 01:00", "الساعة واحده بالظبط"), # Should be "واحده" not "الواحدة"
    ]
    
    print("\n--- Testing Time/Number Pronunciation ---")
    for inputs, expected_part in time_cases:
        output = normalize_arabic_for_tts(inputs)
        print(f"Input: {inputs}")
        print(f"Output: {output}")
        if expected_part in output:
            print("✅ PASS")
        else:
            print(f"❌ FAIL - Expected to contain '{expected_part}'")

    # 2. Time expressions (صباحًا/مساءً)
    period_cases = [
        ("9:00 AM", "صباحًا"),
        ("9:00 ص", "صباحًا"),
        ("9:00 صباحا", "صباحًا"),
        ("9:00 PM", "مساءً"),
        ("9:00 م", "مساءً"),
        ("9:00 مساء", "مساءً"),
        ("الصبح", "صَباحًا"),
        ("بالليل", "مَساءً"),
    ]
    print("\n--- Testing Time Expressions ---")
    for inputs, expected_part in period_cases:
        output = normalize_arabic_for_tts(inputs)
        print(f"Input: {inputs}")
        print(f"Output: {output}")
        if expected_part in output:
            print("✅ PASS")
        else:
            print(f"❌ FAIL - Expected to contain '{expected_part}'")

    # 3. Medical Terms & Ma3ad
    medical_cases = [
        ("ميعاد", "مَعاد"),
        ("موعد", "مَعاد"),
        ("عيادة الجراحة", "عِيادِة العَمَلِية"), # or just surgery
        ("استشاري", "إستِشاري"),
        ("أخصائي", "أَخُصّائي"),
    ]
    print("\n--- Testing Medical Terms ---")
    for inputs, expected_part in medical_cases:
        output = normalize_medical_text(inputs)
        print(f"Input: {inputs}")
        print(f"Output: {output}")
        if expected_part in output:
            print("✅ PASS")
        else:
            print(f"❌ FAIL - Expected to contain '{expected_part}'")

if __name__ == "__main__":
    test_egyptian_enhancements()
