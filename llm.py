# llm.py
#
# This module is the "LLM brain":
# - tracks which local model tag we're using (MODEL_ref)
# - builds the initial system messages for a persona (seed_chat_history)
# - sends each chat turn to Ollama (chat_turn)

import ollama
from personas import PERSONAS
from datetime import datetime
from zoneinfo import ZoneInfo
from config import LOCAL_TZ_NAME

# Keep the model name in a mutable dict so other files can "change" it
# without needing to declare global everywhere.
MODEL_ref = {
    "model": "llama3.2:3b"
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


def build_grounding_context(identity: str, profiles: dict | None) -> str | None:
    """
    Build a grounding blob that lists known structured facts (places, permissions, etc.)
    so the model can answer questions like "what is my home" without guessing.
    """
    tz = ZoneInfo(LOCAL_TZ_NAME)
    now_local = datetime.now(tz).strftime("%A %b %d %Y, %I:%M %p")
    profile = profiles.get(identity, {}) if profiles and identity in profiles else {}
    lines = [
        "CRITICAL RULES - You must follow these strictly:",
        "1. Ground every factual answer ONLY in the profile data below. If data is missing, say 'I don't know' and suggest commands like /setplace or /agenda.",
        "2. DO NOT invent meetings, tasks, locations, contacts, schedules, or calendar events.",
        "3. DO NOT invent personal activities, daily routines, or recent events (like 'just got back from breakfast' or 'we have a meeting').",
        "4. DO NOT make up calendar details. If asked about schedule, use /agenda command or say you don't have that information.",
        "5. You are a personal assistant. For greeting requests, respond warmly and naturally as if speaking to an audience, but don't claim to actually send messages or sync with devices.",
        "6. If asked about capabilities you don't have (like live location/GPS, actual broadcasting to other devices), politely explain your limitations.",
        "7. Never claim features that don't exist. Stay within your actual capabilities.",
        f"Current authenticated user: {identity}.",
        f"Current local time ({LOCAL_TZ_NAME}): {now_local}."
    ]

    places = profile.get("places", {})
    if places:
        lines.append("Saved places:")
        for key, addr in places.items():
            lines.append(f"- {key}: {addr}")
    else:
        lines.append("No saved places are stored for this user.")

    perms = profile.get("permissions", {})
    if perms:
        allowed = [p for p, allowed in perms.items() if allowed]
        denied = [p for p, allowed in perms.items() if not allowed]
        if allowed:
            lines.append(f"Granted permissions: {', '.join(sorted(allowed))}.")
        if denied:
            lines.append(f"Restricted permissions: {', '.join(sorted(denied))}.")

    lines.append("When referencing these facts, quote them exactly and avoid hallucinating.")
    return "\n".join(lines)


def chat_turn(history, user_text: str, identity: str, *, context: str | None = None):
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
    messages = history[:]
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": user_text})

    # Call local model through Ollama
    resp = ollama.chat(
        model=MODEL_ref["model"],
        messages=messages
    )

    # Ollama returns a dict; resp["message"]["content"] is the assistant text.
    assistant_reply = resp["message"]["content"]

    return assistant_reply
