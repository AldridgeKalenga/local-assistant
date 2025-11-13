# llm.py
#
# This module is the "LLM brain":
# - tracks which local model tag we're using (MODEL_ref)
# - builds the initial system messages for a persona (seed_chat_history)
# - sends each chat turn to Ollama (chat_turn)

import ollama
from personas import PERSONAS

# Keep the model name in a mutable dict so other files can "change" it
# without needing to declare global everywhere.
MODEL_ref = {
    "model": "llama3.2:1b"
}


def set_model(new_model_name: str):
    """
    Update which local Ollama model/tag we call.
    repl.py uses this for the /model command.
    """
    if not new_model_name:
        return
    MODEL_ref["model"] = new_model_name


STYLE_REMINDER = (
    "Follow the persona style strictly. If the user requests a different tone, "
    "politely keep the assigned style."
)


def seed_chat_history(identity: str):
    """
    Build the starting system messages for a given identity/persona.

    identity is usually "Aldridge" or "Professor".
    If identity is something else (LOCKED, guest, etc.), repl.py will still
    call this using a fallback identity like "Aldridge" so we get a sane style.

    Returns: a list of message dicts:
    [
      {"role": "system", "content": "... persona system prompt ..."},
      {"role": "system", "content": "Style guide: ..."},
      {"role": "system", "content": STYLE_REMINDER},
    ]
    """
    # safety: if identity isn't in PERSONAS (e.g. guest),
    # fall back to Aldridge persona for tone
    if identity not in PERSONAS:
        identity = "Aldridge"

    persona = PERSONAS[identity]

    return [
        {"role": "system", "content": persona["system"]},
        {"role": "system", "content": f"Style guide: {persona['style']}"},
        {"role": "system", "content": STYLE_REMINDER},
    ]


def chat_turn(history, user_text: str, identity: str):
    """
    Send one user message + the running history to Ollama and get assistant reply.

    - history is the running list of messages (system + past user/assistant turns)
    - user_text is the new user message string
    - identity lets us ensure style stays consistent in case we later want to
      do per-identity routing or extra safety rules

    Returns: assistant_text (string)
    """
    # We append the new user message to a temporary copy
    # to send to ollama.chat.
    messages = history + [{"role": "user", "content": user_text}]

    # Call local model through Ollama
    resp = ollama.chat(
        model=MODEL_ref["model"],
        messages=messages
    )

    # Ollama returns a dict; resp["message"]["content"] is the assistant text.
    assistant_reply = resp["message"]["content"]

    return assistant_reply
