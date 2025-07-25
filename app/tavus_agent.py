import os
import requests

TAVUS_API_KEY = os.getenv("TAVUS_API_KEY")
TAVUS_REPLICA_ID = os.getenv("TAVUS_REPLICA_ID", "r4317e64d25a")
TAVUS_PERSONA_ID = os.getenv("TAVUS_PERSONA_ID", "p70ec11f62ec")

# Optional configuration for saving conversation recordings to S3
TAVUS_RECORDING_S3_BUCKET_NAME = os.getenv("TAVUS_RECORDING_S3_BUCKET_NAME")
TAVUS_RECORDING_S3_BUCKET_REGION = os.getenv("TAVUS_RECORDING_S3_BUCKET_REGION")
TAVUS_AWS_ASSUME_ROLE_ARN = os.getenv("TAVUS_AWS_ASSUME_ROLE_ARN")

HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": TAVUS_API_KEY,
}

DATA = {
    "replica_id": TAVUS_REPLICA_ID,
    "conversation_name": "Olivia",
    "persona_id": TAVUS_PERSONA_ID,
    "custom_greeting": "Hello, Welcome to your genetic counseling session.",
    "conversational_context": "",
    "properties": {
        "enable_closed_captions": False,
        "enable_recording": True,
        "recording_s3_bucket_name": TAVUS_RECORDING_S3_BUCKET_NAME,
        "recording_s3_bucket_region": TAVUS_RECORDING_S3_BUCKET_REGION,
        "aws_assume_role_arn": TAVUS_AWS_ASSUME_ROLE_ARN,
    },
}

API_URL = os.getenv("TAVUS_API_URL", "https://tavusapi.com/v2/conversations")

_CONVERSATION_CACHE = None


def start_conversation(gene: str = "", variant: str = "", status: str = "", force_new: bool = False):
    """Start a conversation using the Tavus API."""
    global _CONVERSATION_CACHE
    if _CONVERSATION_CACHE is not None and not force_new:
        return _CONVERSATION_CACHE

    if not TAVUS_API_KEY:
        raise RuntimeError("TAVUS_API_KEY not set")

    context = (
        "You are a genetic counsellor, empathetic AI host whose job is to answer questions and answers about the gene, variant and status. "
        f"The gene is {gene}, the variant is {variant}, and the status is {status}. "
        "Opening Greeting:\n\nHello! I’m your genetic counselor. I’m happy to answer any questions you may have.   Please feel free to ask me questions.\n"
    )

    payload = {**DATA, "conversational_context": context, "properties": dict(DATA.get("properties", {}))}

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    _CONVERSATION_CACHE = response.json()
    return _CONVERSATION_CACHE


def close_conversation():
    """Clear any cached conversation info."""
    global _CONVERSATION_CACHE
    _CONVERSATION_CACHE = None
