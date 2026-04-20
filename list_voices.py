import requests

url = "https://api.elevenlabs.io/v1/voices"
headers = {
    "Accept": "application/json",
    "xi-api-key": "sk_bbf09910d23504ec56a3a6e46b96ed86844ed8da7ee7ab7e"
}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    for voice in data.get('voices', []):
        if voice.get('category') == 'premade':
            print(f"- {voice['name']} (ID: {voice['voice_id']}, Gen: {voice['labels'].get('gender')})")
else:
    print(f"Error {response.status_code}: {response.text}")
