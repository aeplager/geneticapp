# LLM Chat API

This repository contains a simple Flask application that proxies requests to
multiple Large Language Model providers while keeping chat history between
requests. The available models include:

- **ChatGPT** via the OpenAI API
- **Claude** via the Anthropic API
- **Mistral** via the Mistral API
- **LLama** via a compatible OpenAI-style endpoint
- **Perplexity** via the Perplexity API

## Running locally

```bash
# install dependencies
pip install -r requirements.txt

# set the appropriate API keys as environment variables then start the app
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export MISTRAL_API_KEY=...
# optionally override the default endpoint
# export MISTRAL_API_BASE=https://api.mistral.ai/v1
export LLAMA_API_BASE=...
export LLAMA_API_KEY=...
export PERPLEXITY_API_KEY=...
export PERPLEXITY_MODEL=pplx-70b-online
python app/app.py
```

The server listens on port `5000`.

Open `http://localhost:5000/` in a browser to use the simple web interface. It
lets you choose a provider and select from the models defined in
`app/data/models.json`. Each model entry includes a `multimodal` flag indicating
whether the model can receive file uploads.

### Docker

Build and run using Docker:

```bash
docker build -t llm-chat .
docker run -p 5000:5000 --env-file .env llm-chat
```

Place your API keys in a `.env` file in the project root with values like:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
# MISTRAL_API_BASE=https://api.mistral.ai/v1
LLAMA_API_BASE=...
LLAMA_API_KEY=...
PERPLEXITY_API_KEY=...
PERPLEXITY_MODEL=pplx-70b-online
```

## Endpoints

- `GET /models` – return the available models for each provider.
- `POST /chat` – send a chat message.
- `POST /summarize` – summarize a block of text.
- `GET /history` – retrieve the last 20 prompt/response pairs for a session.

`/chat` and `/summarize` accept JSON with `session_id`, `provider` and
`model_name` along with the user content. The `/chat` endpoint also accepts
`multipart/form-data` with optional file uploads using the `files` parameter.
Files are only forwarded when the chosen model is marked as `multimodal` in
`app/data/models.json`. History for a session is stored in memory and
automatically included in subsequent calls for that session.
