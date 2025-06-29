from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# in-memory storage of conversations by session_id
conversations = {}


@app.route('/')
def index():
    """Simple web UI for chatting with different models."""
    return render_template('index.html')


def call_model(model, messages):
    """Dispatch call to different LLM providers."""
    if model in ("chatgpt", "gpt-3.5-turbo", "openai"):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return resp.choices[0].message.content
    elif model == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=messages,
            max_tokens=1024,
        )
        return resp.content[0].text
    elif model == "mistral":
        from mistralai.client import MistralClient
        client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))
        resp = client.chat(model="mistral-tiny", messages=messages)
        return resp.choices[0].message.content
    elif model == "llama":
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("LLAMA_API_BASE"), api_key=os.getenv("LLAMA_API_KEY"))
        resp = client.chat.completions.create(model="llama", messages=messages)
        return resp.choices[0].message.content
    elif model == "perplexity":
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("PERPLEXITY_API_BASE", "https://api.perplexity.ai"), api_key=os.getenv("PERPLEXITY_API_KEY"))
        resp = client.chat.completions.create(model="pplx-70b-online", messages=messages)
        return resp.choices[0].message.content
    else:
        return "Unknown model"


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    message = data.get('message', '')
    model = data.get('model', 'chatgpt')

    history = conversations.setdefault(session_id, [])
    history.append({'role': 'user', 'content': message})

    response_content = call_model(model, history)

    history.append({'role': 'assistant', 'content': response_content})
    return jsonify({'response': response_content, 'history': history})


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    text = data.get('text', '')
    model = data.get('model', 'chatgpt')

    summary_prompt = f"Summarize the following text:\n\n{text}"

    history = conversations.setdefault(session_id, [])
    history.append({'role': 'user', 'content': summary_prompt})

    response_content = call_model(model, history)

    history.append({'role': 'assistant', 'content': response_content})
    return jsonify({'summary': response_content, 'history': history})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
