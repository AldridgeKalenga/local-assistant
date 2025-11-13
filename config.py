# config.py
import os

# -------- Core assistant identity / model --------
ASSISTANT_NAME = "Assistant"

# path where we store user profiles (places, TTS prefs, permissions, etc.)
PROFILES_PATH = "profiles.json"

# local timezone string used for calendar queries and natural language time parsing
LOCAL_TZ_NAME = "America/New_York"

# -------- Google Calendar config --------
GCAL_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
GCAL_CREDENTIALS_PATH = "credentials.json"  # fallback creds path

# -------- TTS defaults --------
TTS_ENABLED_DEFAULT = True
TTS_RATE_DEFAULT = None
TTS_VOICE_INDEX_DEFAULT = None

# -------- Auth / automation knobs --------
# STRICT_AUTH:
#   1 => lock sensitive features until a recognized face logs in
#   0 => don't lock, just pick last identity or Aldridge
STRICT_AUTH = os.getenv("STRICT_AUTH", "1") == "1"

# We try auto face recognition ON START before dropping into the REPL.
AUTO_RECOG_ON_START = os.getenv("FACE_AUTORECOG", "1") == "1"

# -------- Voice mode knobs --------
VOICE_MODE_DEFAULT = os.getenv("VOICE_MODE", "1") == "1"        # start in hands-free mode if unlocked
VOICE_PHRASE_LIMIT = int(os.getenv("VOICE_PHRASE_LIMIT", "30")) # max seconds to record per utterance
VOICE_END_SILENCE = float(os.getenv("VOICE_END_SILENCE", "2.0"))# silence that ends the utterance
VOICE_MIN_LISTEN = float(os.getenv("VOICE_MIN_LISTEN", "0.8"))  # always listen at least this long

# -------- Face quality / security knobs --------
FACE_MIN_SIZE    = int(os.getenv("FACE_MIN_SIZE", "80"))        # px width/height min
FACE_BLUR_THRESH = float(os.getenv("FACE_BLUR_THRESH", "100"))  # Laplacian variance lower bound
FACE_MAX_BLUR    = float(os.getenv("FACE_MAX_BLUR", "600"))     # Laplacian variance upper bound (too bright)
FACE_DIST_THRESH = float(os.getenv("FACE_DIST_THRESH", "2600")) # max kNN embedding distance allowed

# -------- Camera selection knobs --------
# CAMERA_INDEX:
#   -1 = auto-pick (pref external if PREFER_EXTERNAL=1)
#    0 = force internal webcam
#    1 = force external / iPhone / USB cam
#    2 = etc
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "-1"))

# If True, when auto-picking, we try index 1 first (iPhone/USB) then 0.
# If False, we try 0 first.
PREFER_EXTERNAL = os.getenv("PREFER_EXTERNAL", "1") == "1"

# Hard overrides for each phase:
# ENROLL_CAMERA_INDEX: which cam to use when capturing a new face
# RECOG_CAMERA_INDEX:  which cam to use for recognition/unlock
ENROLL_CAMERA_INDEX = os.getenv("ENROLL_CAMERA_INDEX")  # string or None
RECOG_CAMERA_INDEX  = os.getenv("RECOG_CAMERA_INDEX")   # string or None


# -------- User help text --------
HELP_TEXT = """
Commands:
/recognize
    - Try face recognition again to unlock.

/setup_profile <Name>
    - Enroll a new face for <Name>.
      We'll grab samples from the camera and save them under that name.

/lock
    - Lock the assistant again. Requires face or bypass to unlock.

/login <Name>
    - Dev bypass login. Only for testing with known personas (like Aldridge).
      Guest is not allowed to /login as Aldridge or Professor.

/switch
    - Switch persona AFTER you're unlocked.
      Toggles between Aldridge and Professor.
    - If you are LOCKED, you can use '/switch guest' to enter Guest mode
      (chat only, no calendar, no navigation to saved places).

/model <name>
    - Change which local LLM we're using (the Ollama model tag).

/clear
    - Clear the chat history context for the current persona so it "forgets"
      previous turns in this session.

/nav <place_key>
    - Open Apple Maps / Google Maps navigation to a saved place
      (ex: /nav home). Requires you to be unlocked and have that place saved.

/setplace <key> = <address or 'lat,lon'>
    - Save/update a named destination for navigation. Unlocked profiles only.

/places
    - List all saved named destinations for the current unlocked profile.

/agenda
    - Read your next 10 upcoming calendar events. Only if this profile
      has calendar permission.

/voice on | off | status
    - Turn voice/auto-listen mode on or off, or check status.

/tts on | off
    - Enable or disable text-to-speech playback of the assistant responses.

/voices
    - List available TTS voices on this machine.

/voiceidx <index>
    - Pick which TTS voice you want by index.

/rate <number>
    - Set TTS speaking rate (words per minute-ish).

/mic
    - Manually capture a single voice message and use it as input
      (for when voice mode is OFF).

/help
    - Show this help.

/exit
    - Quit the assistant.
""".strip()
+