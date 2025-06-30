from flask import Flask, request, jsonify, render_template
import os
import json

app = Flask(__name__)

# in-memory storage of conversations by session_id
conversations = {}

# load available models from json file
MODELS_PATH = os.path.join(os.path.dirname(__file__), "data", "models.json")
with open(MODELS_PATH, "r") as f:
    AVAILABLE_MODELS = json.load(f)


@app.route('/')
def index():
    """Simple web UI for chatting with different models."""
    return render_template('index.html')


@app.route('/models')
def list_models():
    """Return the available models for each provider."""
    return jsonify(AVAILABLE_MODELS)


def call_model(provider: str, messages, model_name: str):
    """Dispatch call to different LLM providers using a specific model."""
    if provider in ("chatgpt", "gpt-3.5-turbo", "openai"):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(model=model_name, messages=messages)
        return resp.choices[0].message.content
    elif provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
        )
        return resp.content[0].text
    elif provider == "mistral":
        from openai import OpenAI
        client = OpenAI(
            base_url=os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1"),
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        resp = client.chat.completions.create(model=model_name, messages=messages)
        return resp.choices[0].message.content
    elif provider == "llama":
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("LLAMA_API_BASE"), api_key=os.getenv("LLAMA_API_KEY"))
        resp = client.chat.completions.create(model=model_name, messages=messages)
        return resp.choices[0].message.content
    elif provider == "perplexity":
        from openai import OpenAI
        client = OpenAI(
            base_url=os.getenv("PERPLEXITY_API_BASE", "https://api.perplexity.ai"),
            api_key=os.getenv("PERPLEXITY_API_KEY"),
        )
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages)
            return resp.choices[0].message.content
        except Exception as e:
            return f"Perplexity error: {e}"
    else:
        return "Unknown model"


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    message = data.get('message', '')
    provider = data.get('provider', data.get('model', 'chatgpt'))
    model_name = data.get('model_name', data.get('model'))
    if not model_name:
        model_name = AVAILABLE_MODELS.get(provider, ["gpt-3.5-turbo"])[0]

    history = conversations.setdefault(session_id, [])
    history.append({'role': 'user', 'content': message})

    response_content = call_model(provider, history, model_name)

    history.append({'role': 'assistant', 'content': response_content})
    return jsonify({'response': response_content, 'history': history[-40:]})


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    text = data.get('text', '')
    provider = data.get('provider', data.get('model', 'chatgpt'))
    model_name = data.get('model_name', data.get('model'))
    if not model_name:
        model_name = AVAILABLE_MODELS.get(provider, ["gpt-3.5-turbo"])[0]

    summary_prompt = f"Summarize the following text:\n\n{text}"

    history = conversations.setdefault(session_id, [])
    history.append({'role': 'user', 'content': summary_prompt})

    response_content = call_model(provider, history, model_name)

    history.append({'role': 'assistant', 'content': response_content})
    return jsonify({'summary': response_content, 'history': history})


@app.route('/history', methods=['GET'])
def get_history():
    """Return the last 20 prompt/response pairs for a session."""
    session_id = request.args.get('session_id', 'default')
    history = conversations.get(session_id, [])
    # Grab the last 40 messages (20 pairs)
    recent = history[-40:]
    pairs = []
    for i in range(0, len(recent), 2):
        if i + 1 < len(recent):
            pairs.append({
                'prompt': recent[i]['content'],
                'response': recent[i + 1]['content']
            })
    return jsonify({'history': pairs})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
