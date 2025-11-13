# profiles.py
# Load/save profile data, manage permissions, TTS prefs, saved places.

import json
import os
from config import (
    PROFILES_PATH,
    TTS_ENABLED_DEFAULT,
    TTS_RATE_DEFAULT,
    TTS_VOICE_INDEX_DEFAULT,
)

def load_profiles():
    if not os.path.exists(PROFILES_PATH):
        return {}
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_profiles(profiles):
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

def ensure_identity_struct(profiles, identity):
    """
    Make sure profiles[identity] exists with sane defaults:
    - places
    - tts settings
    - permissions (calendar only auto-True for Aldridge)
    """
    profiles.setdefault(identity, {})
    profiles[identity].setdefault("places", {})
    profiles[identity].setdefault("tts", {
        "enabled": TTS_ENABLED_DEFAULT,
        "voice_index": TTS_VOICE_INDEX_DEFAULT,
        "rate": TTS_RATE_DEFAULT
    })
    profiles[identity].setdefault(
        "permissions",
        {"calendar": (identity == "Aldridge")}
    )

def has_calendar_permission(identity, profiles):
    """
    Check if this identity is allowed to access calendar.
    Guest should always return False unless explicitly changed (which we won't).
    """
    return profiles.get(identity, {}).get("permissions", {}).get("calendar", False)
