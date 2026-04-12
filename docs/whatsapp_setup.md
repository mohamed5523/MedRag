# WhatsApp Cloud API – Setup Guide

This document covers everything needed to connect MedRag to WhatsApp via
the **Meta WhatsApp Cloud API** (no Twilio, no third-party proxy).

---

## Architecture Overview

```
Patient WhatsApp
      │  (sends message)
      ▼
Meta Cloud API ──► POST /webhook/whatsapp  (FastAPI backend)
                         │
                         ├─ Mark message as READ  (✓✓ blue tick)
                         ├─ Send typing indicator (...)
                         │
                         ├─ Route to MCP Clinic Tools  (doctor/schedule/price)
                         └─ Route to RAG engine        (general Q&A)
                                   │
                               answer + TTS audio
                                   │
                         ◄─ POST Graph API /messages
      ▼
Patient received text + voice audio reply
```

---

## Step 1 – Meta App Setup (One-Time)

1. Go to [developers.facebook.com](https://developers.facebook.com) → **My Apps → Create App**
2. Choose **"Business"** app type
3. Add the **WhatsApp** product to the app
4. Under **WhatsApp → API Setup**:
   - Note your **Phone Number ID** → set `WHATSAPP_PHONE_NUMBER_ID` in `.env`
   - Note your **WhatsApp Business Account ID**

---

## Step 2 – Get a Permanent Token

> ⚠️ **Dev tokens from the App Dashboard expire every 24 hours.**  
> For production, always use a permanent System User token.

### Option A — Permanent System User Token (recommended for production)

1. Go to [business.facebook.com](https://business.facebook.com) → **Settings → Users → System Users**
2. Create a new System User (or select existing)
3. Click **"Add Assets"** → select your WhatsApp app → grant `FULL_CONTROL`
4. Click **"Generate New Token"** on the System User
5. Select your app → check **`whatsapp_business_messaging`** + **`whatsapp_business_management`**
6. Copy the token → paste into `backend/.env`:
   ```env
   WHATSAPP_TOKEN=<your-permanent-token>
   ```

### Option B — Short-lived Dev Token (for testing only, expires in 24h)

1. Go to **developers.facebook.com → Your App → WhatsApp → API Setup**
2. Click **"Generate Token"** under the temporary access token section
3. Copy it directly into `backend/.env` as `WHATSAPP_TOKEN`
4. This will stop working after 24 hours — you'll need to repeat this step

---

## Step 3 – Register the Webhook

Meta must be able to reach your server to deliver messages.

### For Production (VPS at `dsb-kairo.de`)

1. Go to **developers.facebook.com → Your App → WhatsApp → Configuration**
2. Under **Webhook**, click **Edit**:
   - **Callback URL**: `https://dsb-kairo.de/webhook/whatsapp`
   - **Verify Token**: `A123456789a`  ← must match `WHATSAPP_VERIFY_TOKEN` in `.env`
3. Click **Verify and Save**
   - Meta sends a `GET /webhook/whatsapp?hub.mode=subscribe&hub.verify_token=...&hub.challenge=...`
   - Your server responds with the challenge string → Meta confirms ✅
4. Under **Webhook Fields**, subscribe to **`messages`**

### For Local Development (with ngrok)

```bash
# Install ngrok (once): https://ngrok.com/download
ngrok http 8000
# Copy the HTTPS URL, e.g. https://abc123.ngrok.io
```
Then in Meta Dashboard set Callback URL to:
```
https://abc123.ngrok.io/webhook/whatsapp
```

---

## Step 4 – Verify Everything Works

### Health Check

```bash
curl https://dsb-kairo.de/webhook/whatsapp/health
```

Expected response:
```json
{
  "status": "healthy",
  "has_token": true,
  "has_phone_id": true,
  "has_verify_token": true,
  "token_type_hint": "likely_permanent",
  "features": {
    "read_receipts": true,
    "typing_indicator": true,
    "voice_transcription": true,
    "tts_audio_reply": true,
    "mcp_clinic_tools": true,
    "disambiguation": true,
    "symptom_triage": true
  }
}
```

> If `token_type_hint` returns `"likely_dev_short_lived"` you are using a temporary
> token that will expire in 24 hours. Follow Step 2 Option A to get a permanent one.

### Webhook Verification Test

```bash
curl "https://dsb-kairo.de/webhook/whatsapp?hub.mode=subscribe&hub.verify_token=A123456789a&hub.challenge=TESTCHALLENGE"
# Should return: TESTCHALLENGE
```

---

## WhatsApp Features Implemented

| Feature | Description |
|---------|-------------|
| 📨 **Text messages** | Full RAG + MCP processing |
| 🎤 **Voice messages** | Transcribed via Groq Whisper, then processed |
| 🔊 **Audio replies** | TTS response sent as audio after text |
| ✓✓ **Read receipts** | Incoming messages marked read instantly |
| ⌛ **Typing indicator** | "..." shown while bot is processing |
| 🏥 **Clinic tools** | Doctor lookup, schedule, pricing via MCP |
| 🔁 **Disambiguation** | Multi-turn doctor/clinic name resolution |
| 🩺 **Symptom triage** | Routes symptom questions to correct specialty |
| 🔁 **Deduplication** | Redis-based 24h dedup prevents duplicate processing |

---

## Troubleshooting

### Messages not arriving
- Check webhook is verified in Meta Dashboard (green ✓)
- Check `messages` field is subscribed in Webhook Fields
- Verify `WHATSAPP_TOKEN` is not expired (check health endpoint's `token_type_hint`)

### Bot responds with "مشكلة في الاتصال بنظام العيادات"
- The MCP server is unreachable — check `docker compose ps` and look for `mcp-server`
- Check `MCP_BASE_URL` env var is set correctly

### Typing indicator / read receipts not showing
- These features require **WhatsApp Business API v17+**
- They are non-critical: if they fail, the response is still sent normally
- Check backend logs: `docker compose logs backend | grep "non-critical"`

### Getting 403 on webhook verification
- Verify `WHATSAPP_VERIFY_TOKEN` in `.env` matches exactly what you typed in Meta Dashboard
- Token is case-sensitive and whitespace-sensitive
