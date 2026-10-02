# Deploy on Railway

1. Push this repo (fork) to GitHub, then Railway: New Project > Deploy from GitHub repo. `railway.json` selects the `Dockerfile` and the `/ready` healthcheck.
2. Variables:
   - `API_KEY`: required in practice. Without it the API is open. Generate one: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
   - `OPENPRONOUNCE_THREADS`: number of vCPUs given to the service
   - `OPENPRONOUNCE_MAX_CONCURRENCY`: simultaneous analyses (default 2; each can use ~1 GB extra RAM)
   - optional: `OPENPRONOUNCE_MAX_UPLOAD_MB` (10), `OPENPRONOUNCE_MAX_AUDIO_SECONDS` (60), `OPENPRONOUNCE_MAX_TEXT_CHARS` (1000)
3. Settings > Resources: start with 8 GB RAM and watch Metrics.
4. Settings > Networking > Generate Domain (HTTPS, needed for the microphone).
5. Open the domain. The UI asks for the API key on the first analysis and stores it in the browser (localStorage).

API calls: `curl -H "X-API-Key: $API_KEY" -F file=@a.wav -F expected_text="hello world" https://<domain>/pronunciation`

Public without a key: `/`, `/static`, `/languages`, `/health`, `/ready`.
