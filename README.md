# Ziva

A lightweight personal assistant: chat with Gemini, save notes, set reminders. FastAPI, SQLite, and a single-page UI with no front-end build step.

## Run locally

    pip install -r requirements.txt
    cp config.example.json config.json   # add your Google API key
    python main.py

Open http://127.0.0.1:8000. The key can also come from the `GOOGLE_API_KEY` environment variable, which takes priority over `config.json`.

## Optional config.json settings

| Key | Purpose |
| --- | --- |
| `openai_compat` | `{base_url, model, timeout}` for Ollama or any OpenAI-compatible server as a fallback chat backend |
| `chat_backend` | `"gemini"` (default) or `"local"` |
| `llm_router` | `true` to route notes and reminders with a model instead of local rules |
| `models` | Override the Gemini model order |
| `doh` | `true` to resolve DNS over HTTPS |
| `tavily_api_key` | Key for Tavily web search (free tier). Also read from the `TAVILY_API_KEY` environment variable. Without it, Ziva falls back to keyless DuckDuckGo search |
| `web_search` | `false` to turn web search off |

## Public demo

Set `ZIVA_DEMO=1` to run it safely in public. Each visitor gets a private workspace tied to a cookie, data expires after a set time, and messages are rate limited per visitor, per IP and per day.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ZIVA_DEMO` | off | Turn demo mode on |
| `ZIVA_HOURLY` | 20 | Messages per visitor per hour |
| `ZIVA_DAILY` | 400 | Messages per day across all visitors |
| `ZIVA_TTL_HOURS` | 24 | Hours before chats, notes and reminders are deleted |

### Deploy on Render

1. Push this repo to GitHub.
2. In Render, choose New, then Blueprint, and select the repo. `render.yaml` does the rest.
3. Set `GOOGLE_API_KEY` and `TAVILY_API_KEY` in the service's environment settings. Never commit them.
