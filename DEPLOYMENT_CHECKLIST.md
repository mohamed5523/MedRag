# 🚀 TTS Integration - Deployment Checklist

## Pre-Deployment Verification

### ✅ Backend Checklist

- [ ] **Environment Variables Set**
  - [ ] `ELEVENLABS_API_KEY` configured
  - [ ] `ELEVENLABS_VOICE_ID` configured
  - [ ] `ELEVENLABS_MODEL` set (default: eleven_flash_v2_5)
  - [ ] All values valid and tested

- [ ] **Dependencies Installed**
  - [ ] `elevenlabs>=2.16.0` installed
  - [ ] `pydantic-settings>=2.11.0` installed
  - [ ] `uv sync` completed successfully
  - [ ] No dependency conflicts

- [ ] **Code Files Present**
  - [ ] `app/core/tts_exceptions.py` exists
  - [ ] `app/core/tts_settings.py` exists
  - [ ] `app/core/text_to_speech.py` exists
  - [ ] `app/api/tts.py` exists
  - [ ] `app/api/chat.py` updated
  - [ ] `app/models/schemas.py` updated
  - [ ] `app/main.py` updated with TTS router

- [ ] **API Endpoints Working**
  - [ ] `GET /api/tts/health` returns 200
  - [ ] `GET /api/tts/voices` returns voice list
  - [ ] `GET /api/tts/audio?text=test` returns audio
  - [ ] `POST /api/tts/synthesize` works
  - [ ] `POST /api/chat/query-with-voice` works
  - [ ] All endpoints handle errors gracefully

- [ ] **Testing Complete**
  - [ ] Unit tests passing (if applicable)
  - [ ] Integration tests passing
  - [ ] Manual testing completed
  - [ ] Error scenarios tested
  - [ ] Performance acceptable (< 2s for TTS)

### ✅ Frontend Checklist

- [ ] **Code Files Updated**
  - [ ] `src/pages/PatientDashboard.tsx` updated
  - [ ] Message interface includes audio fields
  - [ ] Audio playback functions implemented
  - [ ] UI controls added (speaker icons)

- [ ] **API Integration**
  - [ ] Fetch calls `/api/chat/query-with-voice`
  - [ ] Response parsing works correctly
  - [ ] Base64 decoding works
  - [ ] Audio playback works

- [ ] **UI/UX Testing**
  - [ ] Audio auto-plays (or respects browser policy)
  - [ ] Speaker icons visible
  - [ ] Icons change during playback
  - [ ] Only one audio plays at a time
  - [ ] Error messages display correctly
  - [ ] Mobile responsive
  - [ ] Works on Chrome
  - [ ] Works on Firefox
  - [ ] Works on Safari
  - [ ] Works on mobile browsers

- [ ] **Performance**
  - [ ] No memory leaks (Object URLs cleaned)
  - [ ] No console errors
  - [ ] Smooth user experience
  - [ ] Audio loads quickly

### ✅ Documentation Checklist

- [ ] **README Updated**
  - [ ] Backend README includes TTS endpoints
  - [ ] Environment variables documented
  - [ ] Setup instructions clear

- [ ] **Integration Guides**
  - [ ] `QUICK_START_TTS.md` created
  - [ ] `TTS_INTEGRATION.md` created
  - [ ] `TTS_INTEGRATION_SUMMARY.md` created
  - [ ] `TTS_ARCHITECTURE.md` created
  - [ ] `DEPLOYMENT_CHECKLIST.md` (this file)

- [ ] **API Documentation**
  - [ ] OpenAPI/Swagger docs updated
  - [ ] Example requests documented
  - [ ] Example responses documented

## Production Deployment Steps

### 1. Pre-Deployment

```bash
# 1.1 Backup current system
git tag pre-tts-deployment
git push --tags

# 1.2 Create deployment branch
git checkout -b deploy/tts-integration

# 1.3 Final code review
git diff main...deploy/tts-integration

# 1.4 Update version numbers if applicable
# Edit pyproject.toml, package.json, etc.
```

### 2. Backend Deployment

```bash
# 2.1 Navigate to backend
cd heal-query-hub/backend

# 2.2 Set production environment variables
cat > .env.production << EOF
ELEVENLABS_API_KEY=your_production_key
ELEVENLABS_VOICE_ID=your_production_voice
ELEVENLABS_MODEL=eleven_flash_v2_5
# ... other production vars
EOF

# 2.3 Install dependencies
uv sync --frozen

# 2.4 Run database migrations (if any)
# alembic upgrade head

# 2.5 Start backend
uv run python run.py
# OR using systemd/supervisor/docker
```

### 3. Frontend Deployment

```bash
# 3.1 Navigate to frontend
cd heal-query-hub/frontend

# 3.2 Set production environment variables
cat > .env.production << EOF
VITE_API_URL=https://api.yourdomain.com
# ... other production vars
EOF

# 3.3 Build for production
npm run build

# 3.4 Deploy build artifacts
# Upload dist/ to your hosting (Vercel, Netlify, S3, etc.)
```

### 4. Post-Deployment Verification

```bash
# 4.1 Test health endpoint
curl https://api.yourdomain.com/api/tts/health

# 4.2 Test voices endpoint
curl https://api.yourdomain.com/api/tts/voices

# 4.3 Test audio generation
curl "https://api.yourdomain.com/api/tts/audio?text=Hello" -o test.mp3

# 4.4 Test chat with voice
curl -X POST https://api.yourdomain.com/api/chat/query-with-voice \
  -H "Content-Type: application/json" \
  -d '{"query": "What services do you offer?", "max_results": 5}'

# 4.5 Open frontend in browser
open https://yourdomain.com

# 4.6 Test complete user flow
# - Send message
# - Verify audio plays
# - Click speaker icon
# - Verify replay works
```

## Rollback Plan

### If Issues Occur

```bash
# 1. Quick rollback
git revert HEAD
git push

# 2. Or restore from tag
git checkout pre-tts-deployment
git push --force

# 3. Remove TTS environment variables
unset ELEVENLABS_API_KEY
unset ELEVENLABS_VOICE_ID

# 4. Restart services
systemctl restart backend
# OR
docker-compose restart backend

# 5. Clear frontend cache
# Instruct users to hard refresh (Ctrl+Shift+R)
```

### Rollback Verification

```bash
# 1. Check backend is running
curl https://api.yourdomain.com/health

# 2. Check frontend loads
curl https://yourdomain.com

# 3. Test basic chat (without voice)
curl -X POST https://api.yourdomain.com/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "max_results": 5}'
```

## Monitoring Setup

### Backend Monitoring

```python
# Add to backend monitoring

# 1. TTS Success Rate
tts_success_counter = Counter('tts_success_total')
tts_error_counter = Counter('tts_error_total')

# 2. TTS Latency
tts_latency_histogram = Histogram('tts_latency_seconds')

# 3. TTS Request Volume
tts_request_counter = Counter('tts_requests_total')

# 4. Audio Size Distribution
audio_size_histogram = Histogram('audio_size_bytes')
```

### Frontend Monitoring

```typescript
// Add to frontend analytics

// 1. Audio Playback Success Rate
analytics.track('audio_playback_success')
analytics.track('audio_playback_error')

// 2. User Interactions
analytics.track('audio_replay_clicked')
analytics.track('audio_autoplay_blocked')

// 3. Performance Metrics
analytics.track('audio_load_time', { duration: ms })
```

### Alerts to Set Up

- [ ] **Critical Alerts**
  - [ ] TTS API completely down
  - [ ] Error rate > 50%
  - [ ] Backend down
  - [ ] Frontend unreachable

- [ ] **Warning Alerts**
  - [ ] TTS error rate > 10%
  - [ ] TTS latency > 3s
  - [ ] Rate limit approaching
  - [ ] High memory usage

- [ ] **Info Alerts**
  - [ ] Unusual traffic spikes
  - [ ] New error types
  - [ ] Slow performance trends

## Security Checklist

- [ ] **API Keys**
  - [ ] Never committed to git
  - [ ] Stored in secure vault (e.g., AWS Secrets Manager)
  - [ ] Rotated regularly
  - [ ] Different keys for dev/staging/prod

- [ ] **Network Security**
  - [ ] HTTPS enabled
  - [ ] CORS configured correctly
  - [ ] Rate limiting enabled
  - [ ] API authentication in place

- [ ] **Input Validation**
  - [ ] Text length limited
  - [ ] Special characters handled
  - [ ] SQL injection prevented
  - [ ] XSS prevention in place

- [ ] **Error Handling**
  - [ ] No sensitive data in error messages
  - [ ] Proper logging (but not excessive)
  - [ ] User-friendly error messages
  - [ ] Stack traces hidden in production

## Performance Benchmarks

### Expected Performance

```
Backend:
- TTS generation: < 2 seconds
- Chat with voice: < 3 seconds
- TTS health check: < 100ms
- TTS voices list: < 200ms

Frontend:
- Audio decode: < 100ms
- Audio playback start: < 200ms
- UI updates: < 50ms
- Memory usage: < 50MB increase per session
```

### Load Testing

```bash
# Use Apache Bench or similar
ab -n 100 -c 10 https://api.yourdomain.com/api/tts/health

# Use k6 for more complex scenarios
k6 run load-test-tts.js
```

## Post-Deployment Tasks

### Week 1

- [ ] Monitor error rates daily
- [ ] Check user feedback
- [ ] Review performance metrics
- [ ] Adjust rate limits if needed
- [ ] Document any issues found

### Week 2-4

- [ ] Analyze usage patterns
- [ ] Optimize slow queries
- [ ] Review and update documentation
- [ ] Plan next iterations
- [ ] Gather user feedback

### Ongoing

- [ ] Monitor API costs (ElevenLabs usage)
- [ ] Track performance trends
- [ ] Update dependencies regularly
- [ ] Review security advisories
- [ ] Backup and disaster recovery testing

## Cost Monitoring

### ElevenLabs API Costs

```
Track:
- Characters per month
- Cost per 1000 characters
- Average cost per user
- Total monthly cost

Alert when:
- Daily cost > expected
- Monthly budget approaching limit
- Unusual spike in usage
```

### Optimization Tips

1. **Cache common phrases**
   - Store frequently used responses
   - Reduce API calls by 30-50%

2. **Optimize text length**
   - Summarize long responses
   - Split into chunks if needed

3. **Use appropriate model**
   - `eleven_flash_v2_5` for speed
   - `eleven_multilingual_v2` only when needed

4. **Monitor voice usage**
   - Track which voices are used most
   - Consider reducing voice options

## Success Criteria

### Day 1

- [ ] Zero critical errors
- [ ] All endpoints responding
- [ ] Users can access the system
- [ ] Basic functionality works

### Week 1

- [ ] < 5% error rate
- [ ] Average response time < 3s
- [ ] Positive user feedback
- [ ] No major bugs reported

### Month 1

- [ ] < 1% error rate
- [ ] Established usage patterns
- [ ] Cost within budget
- [ ] User satisfaction > 80%

## Emergency Contacts

```
Team:
- Backend Lead: [Name] [Email] [Phone]
- Frontend Lead: [Name] [Email] [Phone]
- DevOps: [Name] [Email] [Phone]

Services:
- ElevenLabs Support: support@elevenlabs.io
- Hosting Provider: [Contact Info]
- Database Provider: [Contact Info]
```

## Deployment Sign-Off

```
Deployment completed by: _________________
Date/Time: _________________
Version deployed: _________________
Rollback tested: ☐ Yes ☐ No

Verified by:
- Backend: _________________ Date: _______
- Frontend: _________________ Date: _______
- QA: _________________ Date: _______

Production URL: _________________________
Monitoring dashboard: _________________________
Incident response plan location: _________________________
```

---

## 🎉 Post-Deployment

Congratulations on deploying TTS integration!

### Next Steps

1. **Monitor closely** for the first 24-48 hours
2. **Gather user feedback** through surveys or support channels
3. **Document lessons learned** for future deployments
4. **Plan Phase 2** enhancements (WhatsApp integration!)

### Questions?

Refer to:

- `QUICK_START_TTS.md` for setup
- `TTS_INTEGRATION.md` for technical details
- `TTS_ARCHITECTURE.md` for architecture
- `TTS_INTEGRATION_SUMMARY.md` for overview

**Happy deploying! 🚀**
