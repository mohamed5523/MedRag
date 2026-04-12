import json
import re
import os

def parse_markdown_prices(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by clinic sections
    sections = re.split(r"##\s+(.+?)\s+\(ID:\s+(\d+)\)", content)
    
    clinics = {}
    doctors_by_name = {}

    # sections[0] is the header (Create At, Source, etc.)
    # sections[1] is clinic name, sections[2] is ID, sections[3] is the table content
    for i in range(1, len(sections), 3):
        clinic_name = sections[i].strip()
        clinic_id = sections[i+1].strip()
        table_content = sections[i+2].strip()

        clinic_data = {
            "id": clinic_id,
            "name": clinic_name,
            "doctors": []
        }

        # Parse table rows
        rows = table_content.split("\n")
        for row in rows:
            if "|" not in row or "----" in row or "الطبيب" in row:
                continue
            
            # Extract parts from row | Name | Price | Extra |
            # Strip empty strings from split
            parts = [p.strip() for p in row.split("|") if p.strip()]
            if len(parts) < 2:
                continue
            
            doctor_name = parts[0]
            price_val = parts[1]
            extra_services = parts[2] if len(parts) > 2 else "—"

            # Normalize price
            consultation_price = None
            if price_val and price_val != "—":
                try:
                    consultation_price = float(price_val)
                except ValueError:
                    consultation_price = None

            # Parse extra services
            # Example: "كشف كمبيوتر: 90" or "رسم قلب: 80" or "—"
            additional_services = []
            if extra_services and extra_services != "—":
                # Splitting by common delimiters if multiple exist, though markdown usually has one per line
                service_parts = extra_services.split(",") # Assuming comma if multiple, adjusting if observed otherwise
                for sp in service_parts:
                    match = re.search(r"(.+?):\s*(\d+)", sp)
                    if match:
                        additional_services.append({
                            "name": match.group(1).strip(),
                            "price": float(match.group(2))
                        })

            doc_entry = {
                "name": doctor_name,
                "consultation_price": consultation_price,
                "additional_services": additional_services,
                "clinic_id": clinic_id,
                "clinic_name": clinic_name
            }

            clinic_data["doctors"].append(doc_entry)
            
            # Also index by doctor name for direct lookup
            if doctor_name not in doctors_by_name:
                doctors_by_name[doctor_name] = []
            doctors_by_name[doctor_name].append(doc_entry)

        clinics[clinic_id] = clinic_data

    return {
        "clinics": clinics,
        "by_doctor": doctors_by_name
    }

if __name__ == "__main__":
    input_file = "/home/morad/Projects/heal-query-hub/doctor_prices.md"
    output_file = "/home/morad/Projects/heal-query-hub/backend/app/core/doctor_prices.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Parsing {input_file}...")
    data = parse_markdown_prices(input_file)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Extraction complete. Saved to {output_file}")
    print(f"Found {len(data['clinics'])} clinics and {len(data['by_doctor'])} unique doctor names.")
