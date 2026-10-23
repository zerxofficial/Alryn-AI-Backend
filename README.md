# ALRYN AI Render gateway

This directory is deployed as a separate repository root in the user's `Alryn-AI-Backend` repository. Copy `main.py`, `providers.py`, `requirements.txt`, and this README to that repository's root. All AI/search credentials belong in Render Environment; never commit real values.

## Render settings

- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`
- Required environment variable: `ALRYN_BACKEND_SHARED_SECRET` (random value, minimum 32 characters; same exact value is configured in the bot's uncommitted `.env`)
- Provider/model variables: see `.env.example`

## Endpoints

- `GET /health`: public basic health check.
- `GET /ready`: checks that the shared secret is configured.
- `GET /v1/providers`: authenticated provider metadata; never returns key values.
- `POST /v1/chat`: authenticated text generation.
- `POST /v1/search`: authenticated Tavily/Exa search.
- `GET /internal/config-check`: authenticated presence-only diagnostics.

All `/v1/*` and internal routes require `Authorization: Bearer <ALRYN_BACKEND_SHARED_SECRET>`. The service validates request size and uses a basic in-memory requests-per-minute guard. This guard resets on restart and is not a durable quota database. The bot applies durable observed-usage limits in its local SQLite database if Pterodactyl storage persists.

Only text chat and Tavily/Exa search are wired in this build. Other provider capabilities are not automatically enabled because a provider key exists; image generation, vision, OCR, voice, video, and music need dedicated adapters and verified model access.
