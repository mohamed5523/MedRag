import urllib.request, json, sys

def fetch_json(url):
    req = urllib.request.Request(url)
    req.add_header('Authorization', 'Basic bWlsbGVuOm1pbGxlbkA0MzIx')
    try:
        data = urllib.request.urlopen(req).read()
        return json.loads(data)
    except Exception as e:
        print(f"Error: {e}")
        return None

# Find surgery clinic ID
providers = fetch_json('http://192.0.0.192:3003/api/clinicProviderlist')
if not providers or 'data' not in providers:
    print("Failed to get providers")
    sys.exit(1)

surgery_clinic_id = None
for c in providers['data']:
    if 'جراح' in c.get('clinicName', ''):
        surgery_clinic_id = c['clinicId']
        print(f"Found surgery clinic: {c['clinicName']} (ID: {surgery_clinic_id})")
        break

if not surgery_clinic_id:
    print("Could not find surgery clinic")
    sys.exit(1)

# Fetch schedule
schedule = fetch_json(f'http://192.0.0.192:3002/api/clinicProviderschedule/?clinicid={surgery_clinic_id}&dateFrom=11/03/2026&dateTo=11/03/2026')
if not schedule:
    sys.exit(1)

print(json.dumps(schedule, ensure_ascii=False, indent=2))
