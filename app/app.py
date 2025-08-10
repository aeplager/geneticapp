from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session,
)
import os
import json
import base64
import requests
from bs4 import BeautifulSoup
from tavus_agent import start_conversation, close_conversation

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "devkey")

# in-memory storage of conversations by session_id
conversations = {}


def login_required(func):
    """Decorator to require login for certain routes."""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return func(*args, **kwargs)

    return wrapper

# load available models from json file
MODELS_PATH = os.path.join(os.path.dirname(__file__), "data", "models.json")
with open(MODELS_PATH, "r") as f:
    AVAILABLE_MODELS = json.load(f)

# recipient specific base prompts used for gene variant explanations
ROLE_PROMPTS = {
    "child": (
        "You are a genetic counselor explaining genetic test results to parents about their child. "
        "Using the gene information, mutation details, and classification status provided, create a compassionate and clear summary that:"
        "- Explains what the gene does in simple terms a parent can understand\n"
        "- Describes the specific mutation found in language that avoids medical jargon\n"
        "- Clearly states whether the mutation is harmful (pathogenic), not harmful (benign), or uncertain (VUS - variant of uncertain significance)\n"
        "- Explains what this means for their child's health and development\n"
        "- Addresses immediate parental concerns about their child's wellbeing\n"
        "- Provides reassurance where appropriate while being honest about uncertainties\n"
        "- Mentions next steps or recommendations for the child's care\n"
        "- Uses a warm, supportive tone that acknowledges this may be overwhelming news\n"
        "Focus on what parents need to know to care for their child and make informed decisions."
    ),
    "self": (
        "You are a genetic counselor explaining genetic test results directly to the person who received them. "
        "Using the gene information, mutation details, and classification status provided, create a clear and empowering summary that:"
        "- Explains what the gene does and why it's important for your body\n"
        "- Describes the specific genetic change that was found in straightforward language\n"
        "- Clearly states whether this change is harmful (pathogenic), harmless (benign), or uncertain (VUS)\n"
        "- Explains what this means for your current and future health\n"
        "- Addresses how this might affect your daily life and medical care\n"
        "- Discusses any preventive measures or monitoring that might be recommended\n"
        "- Explains implications for family planning if relevant\n"
        "- Uses a respectful, direct tone that treats you as an active participant in your healthcare\n"
        "- Emphasizes your autonomy in making decisions about your health\n"
        "Focus on practical information you need to understand your results and take appropriate action."
    ),
    "spouse": (
        "You are a genetic counselor helping someone understand their partner's genetic test results. "
        "Using the gene information, mutation details, and classification status provided, create a supportive summary that:"
        "- Explains what the gene does and why the test was done\n"
        "- Describes the genetic change found in your partner in clear, non-technical language\n"
        "- States whether this mutation is harmful (pathogenic), harmless (benign), or uncertain (VUS)\n"
        "- Explains what this means for your partner's health and wellbeing\n"
        "- Discusses how this might affect your relationship and daily life together\n"
        "- Addresses implications for any children you might have or already have\n"
        "- Explains how you can be supportive while respecting your partner's autonomy\n"
        "- Mentions resources available for both of you\n"
        "- Uses a tone that acknowledges the emotional impact on relationships\n"
        "- Respects privacy while providing information you need as a supportive partner\n"
        "Focus on helping you understand how to be supportive while navigating this information together."
    ),
    "parent": (
        "You are a genetic counselor explaining genetic test results to an individual. "
        "Using the gene information, mutation details, and classification status provided, create a comprehensive yet accessible summary that:"
        "- Explains what the specific gene does in your body using everyday language\n"
        "- Describes the genetic variation that was found without complex scientific terms\n"
        "- Clearly states the classification: harmful (pathogenic), harmless (benign), or uncertain significance (VUS)\n"
        "- Explains what this result means for your health in practical terms\n"
        "- Discusses any recommended medical follow-up or lifestyle considerations\n"
        "- Addresses potential implications for blood relatives\n"
        "- Explains the difference between having a genetic variant and actually developing symptoms\n"
        "- Provides context about how common or rare this finding is\n"
        "- Uses clear, respectful language that doesn't talk down to you\n"
        "- Balances being thorough with being understandable\n"
        "Focus on giving you the complete picture while making the information accessible and actionable."
    ),
}


def fetch_omim_info(gene: str) -> str:
    """Fetch a short description of the gene from the OMIM API."""
    api_key = os.getenv("OMIM_API_KEY")
    if not api_key:
        return ""
    try:
        resp = requests.get(
            "https://api.omim.org/api/entry",
            params={
                "search": gene,
                "include": "geneMap",
                "format": "json",
                "apiKey": api_key,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        entry_list = data.get("omim", {}).get("entryList", [])
        if entry_list:
            entry = entry_list[0].get("entry", {})
            gene_map = entry.get("geneMap", {})
            return gene_map.get("text", "")
    except Exception:
        pass
    return ""


def fetch_medline_conditions(gene: str, variant: str) -> list[str]:
    """Search MedlinePlus for conditions related to the gene and variant."""
    query = f"{gene} {variant}".strip()
    try:
        resp = requests.get(
            "https://medlineplus.gov/search",
            params={"query": query},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for a in soup.select(".results-list a"):  # typical structure
            text = a.get_text(strip=True)
            if text:
                results.append(text)
            if len(results) >= 5:
                break
        return results
    except Exception:
        return []


def get_phenotypes_from_omim(gene: str, variant: str) -> list[str]:
    """Return phenotype names from OMIM for the given gene and variant by scraping the OMIM website."""
    query = f"{gene} {variant}".strip()
    if not query:
        return []
    try:
        # 1. Perform search on OMIM website
        search_url = (
            "https://omim.org/search?index=entry&search="
            + requests.utils.requote_uri(query)
        )
        headers = {
            "User-Agent": "Mozilla/5.0"
        }  # use a common User-Agent to avoid blocking
        resp = requests.get(search_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return []

        html = resp.text

        # 2. Parse HTML to find if it's an entry page or search results
        soup = BeautifulSoup(html, "html.parser")
        phenotypes = []

        # Try to find the Gene-Phenotype table directly
        table = None
        # Look for the header cell that indicates the phenotype table
        header_cell = soup.find(
            lambda tag: tag.name in ["th", "td"] and "Phenotype MIM number" in tag.text
        )
        if header_cell:
            # If we found the header, get the enclosing table
            table = header_cell.find_parent("table")

        if table is None:
            # If no table found, we might be on a search results page.
            # Find link to the gene entry (denoted by gene name or '*')
            link = None
            # Strategy: find a result link containing the gene symbol
            result_links = soup.find_all("a", href=True)
            for a in result_links:
                # Check if the link text or href suggests the gene entry
                # e.g., link text contains gene symbol or href is like "/entry/<OMIMID>"
                if gene.upper() in a.text.upper() or "/entry/" in a["href"]:
                    link = a["href"]
                    # If it looks like an /entry/ link, we take it and break
                    if "/entry/" in a["href"]:
                        break
            if not link:
                return []
            # Some links might be relative, ensure full URL
            if link.startswith("/"):
                link = "https://omim.org" + link
            # Fetch the gene entry page
            resp2 = requests.get(link, headers=headers, timeout=10)
            if resp2.status_code != 200:
                return []
            soup = BeautifulSoup(resp2.text, "html.parser")
            # Find the phenotype table on this page
            header_cell = soup.find(
                lambda tag: tag.name in ["th", "td"]
                and "Phenotype MIM number" in tag.text
            )
            if header_cell:
                table = header_cell.find_parent("table")

        if table:
            # 3. Extract phenotype names from the table
            rows = table.find_all("tr")
            # Usually first row is header
            for row in rows[1:]:
                cells = row.find_all("td")
                if not cells:
                    continue
                # Typically, Phenotype name is the second column (index 1)
                phenotype_name = cells[1].get_text(separator=" ", strip=True)
                if phenotype_name:
                    # Remove any leading '?' indicating provisional relationships
                    if phenotype_name.startswith("?"):
                        phenotype_name = phenotype_name[1:].strip()
                    phenotypes.append(phenotype_name)
            # Deduplicate while preserving order
            unique = []
            for p in phenotypes:
                if p not in unique:
                    unique.append(p)
            return unique[:5]
        else:
            # If we didn't find the table, return empty
            return []
    except Exception:
        return []


def get_first_mim_result_href(url):
    """
    Fetches HTML from a given URL, finds the first <span> with class "mim-result-font",
    and returns the href of the first <a> tag found within it.

    Args:
        url (str): The URL of the web page to parse.

    Returns:
        str or None: The href attribute value if found, otherwise None.
    """
    # Define headers to mimic a web browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, "html.parser")

        # Find the first <span> with the class "mim-result-font"
        target_span = soup.find("span", class_="mim-result-font")

        if target_span:
            # Within that span, find the first <a> tag
            link_tag = target_span.find("a")

            if link_tag and "href" in link_tag.attrs:
                return link_tag["href"]
            else:
                print("No <a> tag or href attribute found within the specified span.")
                return None
        else:
            print("No <span> with class 'mim-result-font' found on the page.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def get_html_as_plain_text(url):
    """
    Fetches HTML from a given URL and returns its plain text content.
    Includes a User-Agent header to mimic a browser.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        soup = BeautifulSoup(response.text, "html.parser")

        # Use get_text() to extract all text.
        # ' ' as separator often helps retain some readability between elements.
        # strip=True removes leading/trailing whitespace from each line/block.
        plain_text = soup.get_text(separator="\n", strip=True)
        return plain_text

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def is_multimodal(provider: str, model_name: str) -> bool:
    """Return True if the given provider/model supports file input."""
    models = AVAILABLE_MODELS.get(provider, [])
    for info in models:
        if isinstance(info, dict):
            if info.get("name") == model_name:
                return info.get("multimodal", False)
        elif info == model_name:
            return False
    return False


def default_model(provider: str) -> str:
    models = AVAILABLE_MODELS.get(provider, [])
    if models:
        first = models[0]
        if isinstance(first, dict):
            return first.get("name")
        return first
    return "gpt-3.5-turbo"


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if email == "ariel.brautbar@gmail.com" and password == "Gilat#12":
            session["logged_in"] = True
            return redirect(url_for("gene_page"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))




@app.route("/")
def root():
    """Landing page - redirect to login."""
    return redirect(url_for("login"))

@app.route("/chat")
@login_required
def index():
    """Simple web UI for chatting with different models."""
    return render_template("index.html")


@app.get("/introduction", endpoint="introduction")
def introduction_get():
    classification_options = [
        {"value": "pathogenic", "label": "Pathogenic"},
        {"value": "likely_pathogenic", "label": "Likely Pathogenic"},
        {"value": "vus", "label": "Variant of Uncertain Significance"},
        {"value": "likely_benign", "label": "Likely Benign"},
        {"value": "benign", "label": "Benign"},
    ]
    model_options = [
        {"value": "gpt-4o", "label": "GPT-4o"},
        {"value": "gpt-5", "label": "GPT-5"},
    ]
    return render_template(
        "introduction.html",
        form_data={},
        errors={},
        classification_options=classification_options,
        model_options=model_options,
    )


@app.route("/gene")
@login_required
def gene_page():
    """Page for looking up gene/variant information."""
    return render_template("gene.html")


@app.route("/conditions")
@login_required
def conditions_page():
    gene = request.args.get("gene", "")
    variant = request.args.get("variant", "")
    provider = request.args.get("provider", "chatgpt")
    model_name = request.args.get("model_name") or default_model(provider)

    # Build the OMIM search URL using the gene and variant
    url = (
        "https://www.omim.org/search?index=entry&start=1&limit=10"
        "&sort=score+desc%2C+prefix_sort+desc&search="
        f"{gene}++++{variant}"
    )

    # Get the first result from the search results page
    href = get_first_mim_result_href(url)
    if not href:
        conditions = []
        summary = ""
    else:
        entry_url = "https://www.omim.org" + href
        text = get_html_as_plain_text(entry_url) or ""
        # Ask the model to extract the phenotype names from the page text
        prompt = "Please tell me the phenotypes listed in this text:\n\n" + text[:10000]
        response = call_model(
            provider, [{"role": "user", "content": prompt}], model_name
        )

        # Convert the model response into a simple list of conditions
        conditions = [
            line.strip("- ")
            for line in (response or "").splitlines()
            if line.strip()
        ]

        summary = ""
        if conditions:
            prompt = (
                "Summarize the following medical conditions for a patient:\n"
                + "\n".join(conditions)
            )
            summary = call_model(
                provider, [{"role": "user", "content": prompt}], model_name
            )

    if (
        request.args.get("json") == "1"
        or request.accept_mimetypes["application/json"]
        > request.accept_mimetypes["text/html"]
    ):
        return jsonify({"conditions": conditions, "summary": summary})

    return render_template(
        "conditions.html",
        conditions=conditions,
        summary=summary,
        gene=gene,
        variant=variant,
    )


@app.route("/chatpage")
@login_required
def chat_page():
    gene = request.args.get("gene", "")
    variant = request.args.get("variant", "")
    status = request.args.get("status", "")
    recipient = request.args.get("recipient", "self")
    return render_template(
        "chat.html", gene=gene, variant=variant, status=status, recipient=recipient
    )


@app.route("/models")
def list_models():
    """Return the available models for each provider."""
    return jsonify(AVAILABLE_MODELS)


@app.route("/tavus")
@login_required
def tavus_page():
    """Render the Tavus conversational agent page."""
    return render_template("tavus.html")


@app.route("/tavus/start")
@login_required
def tavus_start():
    """Start a new Tavus conversation."""
    force_new = request.args.get("new") == "1"
    gene = request.args.get("gene", "")
    variant = request.args.get("variant", "")
    status = request.args.get("status", "")
    try:
        data = start_conversation(gene=gene, variant=variant, status=status, force_new=force_new)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tavus/close")
@login_required
def tavus_close():
    """End the Tavus conversation and go back to the home page."""
    close_conversation()
    return redirect(url_for("gene_page"))


def call_model(provider: str, messages, model_name: str, files=None):
    """Dispatch call to different LLM providers using a specific model."""
    if provider in ("chatgpt", "gpt-3.5-turbo", "openai"):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OpenAI API key not configured"
        client = OpenAI(api_key=api_key)

        if files:
            if not is_multimodal(provider, model_name):
                return "Model does not support file input"

            user_msg = messages[-1] if messages else {"role": "user", "content": ""}
            content = []
            if isinstance(user_msg.get("content"), str):
                content.append({"type": "text", "text": user_msg["content"]})
            elif isinstance(user_msg.get("content"), list):
                content.extend(user_msg["content"])

            for f in files:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                mime = getattr(f, "mimetype", "application/octet-stream")
                content.append(
                    {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"}
                )

            user_msg["content"] = content
            messages = messages[:-1] + [user_msg]

        try:
            resp = client.chat.completions.create(model=model_name, messages=messages)
            return resp.choices[0].message.content
        except Exception as e:
            return f"OpenAI error: {e}"
    elif provider == "claude":
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "Anthropic API key not configured"
        client = anthropic.Anthropic(api_key=api_key)
        if files:
            if not is_multimodal(provider, model_name):
                return "Model does not support file input"

            user_msg = messages[-1] if messages else {"role": "user", "content": ""}
            content = []
            if isinstance(user_msg.get("content"), str):
                content.append({"type": "text", "text": user_msg["content"]})
            elif isinstance(user_msg.get("content"), list):
                content.extend(user_msg["content"])

            for f in files:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                mime = getattr(f, "mimetype", "application/octet-stream")
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": b64},
                    }
                )

            user_msg["content"] = content
            messages = messages[:-1] + [user_msg]

        system_prompt = None
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content", "")
            messages = messages[1:]

        try:
            params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 1024,
            }
            if system_prompt is not None:
                params["system"] = system_prompt
            resp = client.messages.create(**params)
            return resp.content[0].text
        except Exception as e:
            return f"Anthropic error: {e}"
    elif provider == "mistral":
        from openai import OpenAI

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return "Mistral API key not configured"
        client = OpenAI(
            base_url=os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1"),
            api_key=api_key,
        )
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages)
            return resp.choices[0].message.content
        except Exception as e:
            return f"Mistral error: {e}"
    elif provider == "llama":
        from openai import OpenAI

        api_key = os.getenv("LLAMA_API_KEY")
        if not api_key:
            return "LLama API key not configured"
        client = OpenAI(base_url=os.getenv("LLAMA_API_BASE"), api_key=api_key)
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages)
            return resp.choices[0].message.content
        except Exception as e:
            return f"LLama error: {e}"
    elif provider == "perplexity":
        from openai import OpenAI

        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            return "Perplexity API key not configured"
        client = OpenAI(
            base_url=os.getenv("PERPLEXITY_API_BASE", "https://api.perplexity.ai"),
            api_key=api_key,
        )
        try:
            resp = client.chat.completions.create(model=model_name, messages=messages)
            return resp.choices[0].message.content
        except Exception as e:
            return f"Perplexity error: {e}"
    else:
        return "Unknown model"


@app.route("/gene_chat", methods=["POST"])
def gene_chat():
    """Handle gene queries by pulling info from OMIM before calling the LLM."""
    data = request.json or {}
    session_id = data.get("session_id", "gene")
    gene = data.get("gene", "")
    variant = data.get("variant", "")
    status = data.get("status", "")
    recipient = data.get("recipient", "self")
    provider = data.get("provider", "chatgpt")
    model_name = data.get("model_name") or default_model(provider)
    question = data.get("question", "")

    base_prompt = ROLE_PROMPTS.get(recipient, ROLE_PROMPTS["self"])
    gene_info = fetch_omim_info(gene)

    history = conversations.setdefault(session_id, [])
    if not history:
        # only include the role instructions once per session
        history.append({"role": "system", "content": base_prompt})

    prompt = (
        f"Gene: {gene}\nVariant: {variant}\n"
        f"Classification Status: {status}\n"
        f"Information from OMIM: {gene_info}"
    )
    if question:
        prompt += f"\nQuestion: {question}"

    history.append({"role": "user", "content": prompt})

    response_content = call_model(provider, history, model_name)

    history.append({"role": "assistant", "content": response_content})
    return jsonify({"response": response_content, "history": history[-40:]})


@app.route("/chat", methods=["POST"])
def chat():
    files = None
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        data = request.form
        files = request.files.getlist("files")
    else:
        data = request.json or {}

    session_id = data.get("session_id", "default")
    message = data.get("message", "")
    provider = data.get("provider", data.get("model", "chatgpt"))
    model_name = data.get("model_name", data.get("model"))
    if not model_name:
        model_name = default_model(provider)

    history = conversations.setdefault(session_id, [])
    history.append({"role": "user", "content": message})

    response_content = call_model(provider, history, model_name, files=files)

    history.append({"role": "assistant", "content": response_content})
    return jsonify({"response": response_content, "history": history[-40:]})


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.json or {}
    session_id = data.get("session_id", "default")
    text = data.get("text", "")
    provider = data.get("provider", data.get("model", "chatgpt"))
    model_name = data.get("model_name", data.get("model"))
    if not model_name:
        model_name = default_model(provider)

    summary_prompt = f"Summarize the following text:\n\n{text}"

    history = conversations.setdefault(session_id, [])
    history.append({"role": "user", "content": summary_prompt})

    response_content = call_model(provider, history, model_name)

    history.append({"role": "assistant", "content": response_content})
    return jsonify({"summary": response_content, "history": history})


@app.route("/history", methods=["GET"])
def get_history():
    """Return the last 20 prompt/response pairs for a session."""
    session_id = request.args.get("session_id", "default")
    history = conversations.get(session_id, [])
    # Grab the last 40 messages (20 pairs)
    recent = history[-40:]
    pairs = []
    for i in range(0, len(recent), 2):
        if i + 1 < len(recent):
            pairs.append(
                {"prompt": recent[i]["content"], "response": recent[i + 1]["content"]}
            )
    return jsonify({"history": pairs})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=5000)
