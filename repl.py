# repl.py

import re
import datetime

from config import (
    ASSISTANT_NAME,
    LOCAL_TZ_NAME,
    STRICT_AUTH,
    VOICE_MODE_DEFAULT,
    VOICE_PHRASE_LIMIT,
    VOICE_END_SILENCE,
    VOICE_MIN_LISTEN,
    HELP_TEXT,
)

from personas import PERSONAS
from profiles import (
    load_profiles,
    save_profiles,
    ensure_identity_struct,
    has_calendar_permission,
)

from llm import (
    MODEL_ref,
    set_model,
    seed_chat_history,
    chat_turn,
)

from tts_stt import (
    tts,
    stt,
)

from nav import (
    open_maps_destination,
)

from calendar_tools import (
    gcal_list_upcoming,
    gcal_format_event,
    handle_calendar_text,
)

from face_auth import (
    HAS_CV2,
    recognize_quick,
    capture_profile,
)

# ---------- helpers ----------

def print_header(identity: str):
    print("\n----------------------------------------")
    print(f" User: {identity}  |  Model: {MODEL_ref['model']}  |  Assistant: {ASSISTANT_NAME}")
    print("----------------------------------------")


def do_voice_listen(voice_mode_active):
    """
    Listen for voice input through STT.
    Returns (transcribed_text or None, voice_mode_active_after).
    """
    if not stt.available:
        print("(STT unavailable)")
        return None, voice_mode_active

    spoken = stt.listen_once()
    if not spoken:
        if voice_mode_active:
            print("(Heard nothing, listening again...)")
            return None, True
        return None, voice_mode_active

    print(f"(You said) {spoken}")
    low_spoken = spoken.lower().strip()
    exit_phrases = ["exit", "quit", "goodbye", "bye", "stop", "pause"]
    if voice_mode_active and any(p in low_spoken for p in exit_phrases):
        print("(Voice mode paused. Type to continue or /voice on to resume.)")
        return None, False

    return spoken, voice_mode_active


def attempt_auto_identity(profiles):
    """
    Decide initial identity when we launch.
    Logic:
    - If STRICT_AUTH = True:
        * Try face recognition.
        * If face matches known persona => return that persona (authed=True)
        * Else => return "LOCKED", authed=False
    - If STRICT_AUTH = False:
        * Try face recognition.
        * If match => that persona
        * else => fallback last identity or 'Aldridge'
    """
    who = None
    if HAS_CV2:
        who = recognize_quick(timeout_frames=250, need_votes=5, k=5)

    if STRICT_AUTH:
        if who and who in PERSONAS:
            return who, True
        return "LOCKED", False

    # non-strict mode fallback
    if who and who in PERSONAS:
        return who, True
    last = profiles.get("_last_identity")
    if last in PERSONAS:
        return last, True
    return "Aldridge", True


def run_repl():
    profiles = load_profiles()

    # pick startup identity
    identity, authed = attempt_auto_identity(profiles)

    # If STRICT_AUTH and we didn't auto-auth, start LOCKED (not guest yet).
    # Guest will be allowed via '/switch guest'.
    if identity not in PERSONAS and identity != "LOCKED":
        # safety fallback
        identity = "LOCKED"
        authed = False

    # make sure that profile structs exist
    seed_key = identity if identity in PERSONAS else "Aldridge"
    ensure_identity_struct(profiles, seed_key)

    # build starting chat history for that persona (or Aldridge while locked)
    history = seed_chat_history(seed_key)

    print_header(identity)

    # load TTS prefs (even if locked we grab Aldridge's bucket
    # so we have something consistent to speak with)
    tts_bucket = identity if identity in PERSONAS else "Aldridge"
    ensure_identity_struct(profiles, tts_bucket)
    tts_enabled = profiles[tts_bucket]["tts"]["enabled"]
    tts_voice_index = profiles[tts_bucket]["tts"]["voice_index"]
    tts_rate = profiles[tts_bucket]["tts"]["rate"]

    # greeting logic
    if identity == "LOCKED":
        if not HAS_CV2:
            print("(Camera / OpenCV not available.)")
        # check if face_dataset exists
        import os
        has_faces = os.path.isdir("face_dataset") and any(
            f.endswith(".npy") for f in os.listdir("face_dataset")
        )
        if not has_faces:
            print("(No enrolled faces yet.)")
        print("→ Use `/setup_profile <Name>` to enroll.")
        print("→ Use `/recognize` after that to unlock.")
        print("→ You can still use guest mode with `/switch guest` (no calendar / no saved places).")
        print("Note: All sensitive features are restricted while locked.")
    else:
        greet = f"Hey {identity}, how can I help you today?"
        print(greet)
        if tts_enabled:
            tts.speak(greet, True, tts_voice_index, tts_rate)

    # voice mode starts active only if:
    # - unlocked identity
    # - STT actually available
    voice_mode = (
        VOICE_MODE_DEFAULT
        and stt.available
        and identity != "LOCKED"
    )
    if voice_mode:
        print("(Voice mode active - speak naturally, say 'pause' to stop)")

    # --------------- MAIN LOOP ---------------
    while True:

        # -------- input stage --------
        if voice_mode:
            spoken, voice_mode = do_voice_listen(voice_mode)
            if spoken is None:
                # either keep listening or voice_mode just went False
                if not voice_mode:
                    # voice mode turned off by saying "pause" etc.
                    # fall through to typed input next loop
                    pass
                continue
            user = spoken
            low = user.lower()
        else:
            try:
                user = input(f"{identity}> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            low = user.lower()

        # -------- global commands --------
        if low == "/exit":
            break

        if low == "/help":
            print(HELP_TEXT)
            continue

        # ---------- AUTH / LOCK / LOGIN ----------
        if low == "/recognize":
            if not HAS_CV2:
                print("(Camera / OpenCV not available.)")
                continue

            who = recognize_quick(timeout_frames=250, need_votes=5, k=5)

            # Case 0: we didn't confidently match any face at all
            if not who:
                print("(No authorized face recognized.)")
                print("→ You can still use guest mode with `/switch guest` (limited features).")

                if tts_enabled:
                    tts.speak(
                        "No authorized face recognized. "
                        "You can still use guest mode with slash switch guest.",
                        True,
                        tts_voice_index,
                        tts_rate
                    )
                continue

            # Case 1: we matched a name, but that name is NOT an approved persona
            if who not in PERSONAS:
                print(f"(I see you as '{who}', but you don't have a profile with permissions yet.)")
                print("→ You can still use guest mode: `/switch guest` (no calendar / no saved places).")
                print("→ Ask Aldridge to add you as a persona if you should have your own profile.")

                if tts_enabled:
                    tts.speak(
                        f"I see you as {who}, but you don't have a profile yet. "
                        "You can still use guest mode with slash switch guest.",
                        True,
                        tts_voice_index,
                        tts_rate
                    )
                continue

            # Case 2: we matched an approved persona (like Aldridge / Professor)
            identity = who
            profiles["_last_identity"] = identity
            save_profiles(profiles)

            # reset chat with that persona's style/system prompts
            history = seed_chat_history(identity)
            print_header(identity)

            greet = f"Hey {identity}, how can I help you?"
            print(greet)

            # Make sure this persona has its TTS prefs, etc.
            ensure_identity_struct(profiles, identity)
            tts_enabled = profiles[identity]["tts"]["enabled"]
            tts_voice_index = profiles[identity]["tts"]["voice_index"]
            tts_rate = profiles[identity]["tts"]["rate"]

            # Speak greeting if TTS is enabled
            if tts_enabled:
                tts.speak(greet, True, tts_voice_index, tts_rate)

            # After successful recognition, turn on voice mode if allowed
            if stt.available and VOICE_MODE_DEFAULT:
                voice_mode = True
                print("(Voice mode active - speak naturally, say 'pause' to stop)")
            continue

        if low.startswith("/setup_profile"):
            parts = user.split(maxsplit=1)
            name = parts[1].strip() if len(parts) > 1 else ""
            if not name:
                print("Usage: /setup_profile <Name>")
                continue
            ok = capture_profile(name)
            if ok:
                print(f"Enrolled '{name}'. Say `/recognize` to unlock.")
            else:
                print("Enrollment failed or no samples captured.")
            continue

        if low == "/lock":
            identity = "LOCKED"
            print("Locked. Use `/recognize` or `/setup_profile <Name>`.")
            # when locked, we kill voice mode because we don't want
            # passive /nav etc. in a locked state
            voice_mode = False
            continue

        if low.startswith("/login "):
            devname = user.split(maxsplit=1)[1].strip()
            # dev bypass ONLY for known personas (not guest)
            if devname in PERSONAS and devname.lower() != "guest":
                identity = devname
                profiles["_last_identity"] = identity
                save_profiles(profiles)

                history = seed_chat_history(identity)
                print_header(identity)
                print(f"(DEV BYPASS) Logged in as {identity}.")

                ensure_identity_struct(profiles, identity)
                tts_enabled = profiles[identity]["tts"]["enabled"]
                tts_voice_index = profiles[identity]["tts"]["voice_index"]
                tts_rate = profiles[identity]["tts"]["rate"]

                # allow voice mode in bypass too
                if stt.available and VOICE_MODE_DEFAULT:
                    voice_mode = True
                    print("(Voice mode active - speak naturally, say 'pause' to stop)")
            else:
                print(f"'{devname}' is not an allowed /login target.")
            continue

        # ---------- persona switching ----------
        if low == "/switch":
            # normal toggle between Aldridge and Professor IF you're not locked and not guest
            if identity in ("Aldridge", "Professor"):
                identity = "Professor" if identity == "Aldridge" else "Aldridge"
                ensure_identity_struct(profiles, identity)
                history = seed_chat_history(identity)

                # refresh TTS prefs for new persona
                tts_enabled = profiles[identity]["tts"]["enabled"]
                tts_voice_index = profiles[identity]["tts"]["voice_index"]
                tts_rate = profiles[identity]["tts"]["rate"]

                print_header(identity)
                profiles["_last_identity"] = identity
                save_profiles(profiles)
            elif identity == "LOCKED":
                # allow switching to guest from LOCKED
                identity = "guest"
                # guest uses Aldridge style/persona seed for now
                history = seed_chat_history("Aldridge")
                print_header("guest")
                print("Guest mode active. You can chat, but calendar and saved places are restricted.")
                # voice mode for guest? we can allow STT but still no sensitive stuff
                if stt.available and VOICE_MODE_DEFAULT:
                    voice_mode = True
                    print("(Voice mode active - speak naturally, say 'pause' to stop)")
            elif identity == "guest":
                # guest cannot just /switch into Aldridge/Professor without auth
                print("Guest cannot switch to a protected profile. Use /recognize or /login (dev).")
            else:
                # some weird state
                print("Can't switch personas from here.")
            continue

        # ---------- BLOCK SENSITIVE COMMANDS IF LOCKED OR GUEST ----------
        locked_like = (identity == "LOCKED" or identity == "guest")

        # /clear always allowed (just local memory wipe)
        if low == "/clear":
            # seed history for current identity if it's a known persona
            if identity in PERSONAS:
                history = seed_chat_history(identity)
            else:
                # LOCKED or guest -> just seed Aldridge so style is sane
                history = seed_chat_history("Aldridge")
            print("(Cleared chat context.)")
            continue

        # model switch
        if low.startswith("/model "):
            new_model = user.split(maxsplit=1)[1].strip()
            set_model(new_model)
            print(f"(Model set to {new_model})")
            continue

        # TTS commands (/tts, /voices, /voiceidx, /rate)
        if low.startswith("/tts "):
            arg = user.split(maxsplit=1)[1].strip().lower()
            if arg in ("on","off"):
                tts_enabled = (arg == "on")
                # only store prefs if identity is a real persona
                if identity in PERSONAS:
                    ensure_identity_struct(profiles, identity)
                    profiles[identity]["tts"]["enabled"] = tts_enabled
                    save_profiles(profiles)
                print(f"(TTS {'enabled' if tts_enabled else 'disabled'})")
            continue

        if low == "/voices":
            voices = tts.list_voices()
            if not voices:
                print("(No voices found or TTS unavailable.)")
            else:
                for i, name, lang in voices:
                    lang_str = f" {lang}" if lang else ""
                    print(f"{i}: {name}{lang_str}")
            continue

        if low.startswith("/voiceidx "):
            try:
                idx = int(user.split(maxsplit=1)[1].strip())
                if identity in PERSONAS:
                    ensure_identity_struct(profiles, identity)
                    profiles[identity]["tts"]["voice_index"] = idx
                    save_profiles(profiles)
                tts_voice_index = idx
                print(f"(Voice set to {idx})")
            except Exception:
                print("Usage: /voiceidx <index>")
            continue

        if low.startswith("/rate "):
            try:
                rate_val = int(user.split(maxsplit=1)[1].strip())
                if identity in PERSONAS:
                    ensure_identity_struct(profiles, identity)
                    profiles[identity]["tts"]["rate"] = rate_val
                    save_profiles(profiles)
                tts_rate = rate_val
                print(f"(Rate set to {rate_val})")
            except Exception:
                print("Usage: /rate <number>")
            continue

        # voice mode toggle / status
        if low.startswith("/voice "):
            arg = user.split(maxsplit=1)[1].strip().lower()
            if arg == "on":
                if not stt.available:
                    print("(STT unavailable. Install STT deps.)")
                else:
                    voice_mode = True
                    print("(Voice mode ON - speak naturally, say 'pause' to stop)")
            elif arg == "off":
                voice_mode = False
                print("(Voice mode OFF - type commands normally)")
            elif arg == "status":
                status = "ON" if voice_mode else "OFF"
                print(f"(Voice mode: {status})")
                if voice_mode:
                    print(f"  - Phrase limit: {VOICE_PHRASE_LIMIT}s")
                    print(f"  - End silence: {VOICE_END_SILENCE}s")
                    print(f"  - Min listen: {VOICE_MIN_LISTEN}s")
            else:
                print("Usage: /voice on|off|status")
            continue

        # calendar permission management (only Aldridge can change permissions)
        if low.startswith("/grant_calendar "):
            if identity != "Aldridge":
                print("(Only Aldridge can grant calendar permissions.)")
                continue
            target = user.split(maxsplit=1)[1].strip()
            if target not in PERSONAS:
                print(f"('{target}' is not a defined persona.)")
                continue
            ensure_identity_struct(profiles, target)
            profiles[target]["permissions"]["calendar"] = True
            save_profiles(profiles)
            print(f"(Calendar access granted to {target}.)")
            continue

        if low.startswith("/revoke_calendar "):
            if identity != "Aldridge":
                print("(Only Aldridge can revoke calendar permissions.)")
                continue
            target = user.split(maxsplit=1)[1].strip()
            if target not in PERSONAS:
                print(f"('{target}' is not a defined persona.)")
                continue
            ensure_identity_struct(profiles, target)
            profiles[target]["permissions"]["calendar"] = False
            save_profiles(profiles)
            print(f"(Calendar access revoked from {target}.)")
            continue

        # ---------- Sensitive commands that require unlocked REAL identity ----------

        # Calendar commands
        if low == "/agenda":
            if locked_like:
                print("(Locked/Guest cannot access calendar.)")
                if tts_enabled:
                    tts.speak("Calendar is locked.", True, tts_voice_index, tts_rate)
                continue
            if not has_calendar_permission(identity, profiles):
                print("(No calendar access for this profile.)")
                if tts_enabled:
                    tts.speak("You don't have calendar access.", True, tts_voice_index, tts_rate)
                continue
            try:
                events = gcal_list_upcoming(n=10, tzname=LOCAL_TZ_NAME, identity=identity)
                if not events:
                    print("(No upcoming events.)")
                    if tts_enabled:
                        tts.speak("You have no upcoming events.", True, tts_voice_index, tts_rate)
                else:
                    print("Upcoming events:")
                    if tts_enabled:
                        tts.speak("Here are your next events.", True, tts_voice_index, tts_rate)
                    for ev in events:
                        print(gcal_format_event(ev, tzname=LOCAL_TZ_NAME))
            except Exception as e:
                print(f"(Calendar error: {e})")
            continue

        # Natural language calendar questions
        if not locked_like and has_calendar_permission(identity, profiles):
            handled, tts_summary = handle_calendar_text(user, tzname=LOCAL_TZ_NAME, identity=identity)
            if handled:
                if tts_enabled and tts_summary:
                    tts.speak(tts_summary, True, tts_voice_index, tts_rate)
                continue
        else:
            # sniff if they asked about calendar while locked/guest/no permission
            cal_words = [
                "calendar", "schedule", "am i free", "what do i have",
                "what's on", "whats on", "do i have anything", "busy"
            ]
            if any(w in low for w in cal_words):
                if locked_like:
                    print("(Locked/Guest cannot access calendar.)")
                else:
                    print("(No calendar access for this profile.)")
                if tts_enabled:
                    tts.speak("Calendar is locked.", True, tts_voice_index, tts_rate)
                continue

        # Navigation command /nav
        if low.startswith("/nav "):
            if locked_like:
                print("(Locked/Guest cannot open saved navigation.)")
                continue
            key = user.split(maxsplit=1)[1].strip().lower()
            dest = profiles.get(identity, {}).get("places", {}).get(key)
            if not dest:
                print(f"(No place '{key}' for {identity}. Use `/setplace {key} = <address or lat,lon>`.)")
                continue
            url = open_maps_destination(dest)
            print(f"Opening navigation to '{key}' → {dest}\nURL: {url}")
            continue

        # Save a place (only unlocked real identity)
        if low.startswith("/setplace "):
            if locked_like:
                print("(Locked/Guest cannot save places.)")
                continue
            m = re.match(r"/setplace\s+([A-Za-z0-9_\-\. ]+)\s*=\s*(.+)$", user)
            if not m:
                print("Usage: /setplace <key> = <address or lat,lon>")
                continue
            k = m.group(1).strip().lower()
            v = m.group(2).strip()
            ensure_identity_struct(profiles, identity)
            profiles[identity]["places"][k] = v
            save_profiles(profiles)
            print(f"(Saved place '{k}' for {identity}.)")
            continue

        if low == "/places":
            if locked_like:
                print("(Locked/Guest cannot view saved places.)")
                continue
            places = profiles.get(identity, {}).get("places", {})
            if not places:
                print("(No saved places.)")
            else:
                for k, v in places.items():
                    print(f"- {k}: {v}")
            continue

        # STT one-shot mic capture
        if low == "/mic":
            if not stt.available:
                print("(STT unavailable. Install STT deps.)")
                continue
            spoken, _ = do_voice_listen(False)
            if not spoken:
                continue
            user = spoken
            low = user.lower()

            # After /mic capture, try calendar intent if allowed
            if (not locked_like) and has_calendar_permission(identity, profiles):
                handled, tts_summary = handle_calendar_text(user, tzname=LOCAL_TZ_NAME, identity=identity)
                if handled:
                    if tts_enabled and tts_summary:
                        tts.speak(tts_summary, True, tts_voice_index, tts_rate)
                    continue

        # Natural-language nav ("take me to work")
        nav_match = re.search(
            r"(navigate|directions|route|drive|take me|go to|lead me)\s+(to\s+)?(?P<place>[A-Za-z0-9 _\-\.,]+)$",
            user.strip(),
            re.IGNORECASE
        )
        if nav_match:
            if locked_like:
                print("(Locked/Guest cannot open saved navigation.)")
                continue
            place_key = re.sub(
                r"^to\s+",
                "",
                nav_match.group("place").strip(),
                flags=re.IGNORECASE
            ).strip().lower()
            dest = profiles.get(identity, {}).get("places", {}).get(place_key)
            if not dest:
                print(f"(I don't have '{place_key}' saved for {identity}. Use `/setplace {place_key} = <address or lat,lon>`.)")
                continue
            url = open_maps_destination(dest)
            print(f"Opening navigation to '{place_key}' → {dest}\nURL: {url}")
            continue

        # ---------- Chat fallback (LLM) ----------
        try:
            assistant_text = chat_turn(history, user, identity)
        except Exception as e:
            print(f"(LLM error: {e})")
            continue

        # Keep convo context
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": assistant_text})

        print(f"\n{ASSISTANT_NAME}: {assistant_text}\n")
        if tts_enabled:
            to_say = assistant_text if len(assistant_text) <= 600 else assistant_text[:600] + " ..."
            tts.speak(to_say, True, tts_voice_index, tts_rate)
