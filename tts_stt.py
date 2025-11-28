# tts_stt.py
# Text-to-speech (TTS) and speech-to-text (STT) helpers.
# We expose ready-to-use singletons: `tts` and `stt`.

import os
import sys
import subprocess
import queue

from config import (
    VOICE_END_SILENCE,
    VOICE_MIN_LISTEN,
    VOICE_PHRASE_LIMIT,
    STT_BACKEND,
    VOICE_THOUGHT_PADDING,
    WAKE_WORDS,
    WAKE_WORD_PHRASE_LIMIT,
    WAKE_WORD_END_SILENCE,
    WAKE_WORD_MIN_LISTEN,
)


class TTS:
    """
    Handles spoken responses.
    Priority:
    - pyttsx3 (works offline, cross-platform)
    - macOS 'say' command (fallback on mac)
    """
    def __init__(self):
        self.debug = os.getenv("TTS_DEBUG", "0") == "1"
        self.has_pyttsx3 = self._check_pyttsx3()
        self.has_say = (sys.platform == "darwin" and self._has_cmd("say"))
        self.available = self.has_pyttsx3 or self.has_say

    def _check_pyttsx3(self):
        try:
            import pyttsx3  # noqa
            return True
        except Exception:
            return False

    def _has_cmd(self, cmd):
        from shutil import which
        return which(cmd) is not None

    def list_voices(self):
        """
        Returns [(index, name, lang), ...]
        """
        if self.has_pyttsx3:
            try:
                import pyttsx3
                eng = pyttsx3.init()
                voices = eng.getProperty("voices") or []
                out = []
                for i, v in enumerate(voices):
                    name = getattr(v, "name", "")
                    lang = getattr(v, "languages", [""])[0] if hasattr(v, "languages") else ""
                    out.append((i, name, lang))
                try:
                    eng.stop()
                except Exception:
                    pass
                del eng
                return out
            except Exception:
                pass

        if self.has_say:
            try:
                res = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
                lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
                return [(i, ln, "") for i, ln in enumerate(lines)]
            except Exception:
                pass

        return []

    def speak(self, text, enabled=True, voice_index=None, rate=None):
        """
        Say `text` out loud if enabled.
        voice_index/rate currently only affect pyttsx3 path.
        """
        if not enabled or not text:
            return

        # Try pyttsx3 first
        if self.has_pyttsx3:
            try:
                import pyttsx3
                if self.debug:
                    print("(TTS: using pyttsx3)")
                eng = pyttsx3.init()

                if voice_index is not None:
                    try:
                        voices = eng.getProperty("voices") or []
                        v = voices[int(voice_index)]
                        eng.setProperty("voice", v.id)
                    except Exception as e:
                        if self.debug:
                            print(f"(TTS: couldn't set voice {voice_index}: {e})")

                if rate is not None:
                    try:
                        eng.setProperty("rate", int(rate))
                    except Exception as e:
                        if self.debug:
                            print(f"(TTS: couldn't set rate {rate}: {e})")

                eng.say(text)
                eng.runAndWait()
                try:
                    eng.stop()
                except Exception:
                    pass
                del eng
                return
            except Exception as e:
                if self.debug:
                    print(f"(TTS: pyttsx3 failed: {e})")

        # Fallback to macOS 'say'
        if self.has_say:
            try:
                if self.debug:
                    print("(TTS: using macOS 'say')")
                subprocess.run(["say", text])
                return
            except Exception as e:
                if self.debug:
                    print(f"(TTS: say failed: {e})")

        if self.debug:
            print("(TTS: no available backend)")


class STT:
    """
    Handles voice input → text.
    Supports:
    - vosk (offline local model) if VOSK_MODEL is set
    - SpeechRecognition + Google Web Speech as fallback
    """
    def __init__(self):
        self.vosk_model_path = os.getenv("VOSK_MODEL")
        self.has_sr = self._check_sr()
        self.has_vosk = self._check_vosk()
        self.preferred_backend = STT_BACKEND  # "auto", "vosk", or "google"
        self.available = self.has_sr or (self.has_vosk and self.vosk_model_path)

    def _check_sr(self):
        try:
            import speech_recognition  # noqa
            return True
        except Exception:
            return False

    def _check_vosk(self):
        try:
            import vosk  # noqa
            import sounddevice  # noqa
            return True
        except Exception:
            return False

    def listen_once(
        self,
        *,
        end_silence=VOICE_END_SILENCE,
        min_listen=VOICE_MIN_LISTEN,
        phrase_time_limit=VOICE_PHRASE_LIMIT,
        samplerate=16000,
        blocksize=8000,
        allow_wake_prefix=False,
        quiet=False
    ):
        """
        Record one utterance, return the recognized text (string),
        or None if we got nothing.
        """
        # Offline Vosk path
        use_vosk = (
            self.has_vosk and self.vosk_model_path and
            (self.preferred_backend in ("auto", "vosk"))
        )
        if use_vosk:
            try:
                import vosk, sounddevice as sd, json as _json, time as _time
                model = vosk.Model(self.vosk_model_path)

                q = queue.Queue()

                def _cb(indata, frames, time, status):
                    if status:
                        pass  # ignore warnings
                    q.put(bytes(indata))

                with sd.RawInputStream(
                    samplerate=samplerate,
                    blocksize=blocksize,
                    dtype="int16",
                    channels=1,
                    callback=_cb
                ):
                    rec = vosk.KaldiRecognizer(model, samplerate)
                    if not quiet:
                        print("(Listening… speak now)")
                    start_t = _time.time()
                    last_voice_t = start_t
                    heard_anything = False

                    while True:
                        try:
                            data = q.get(timeout=0.1)
                        except queue.Empty:
                            data = None

                        now = _time.time()

                        if data:
                            if rec.AcceptWaveform(data):
                                res = _json.loads(rec.Result())
                                text = (res.get("text") or "").strip()
                                if text:
                                    # remove optional wake prefix "slash "
                                    if allow_wake_prefix and text.startswith("slash "):
                                        text = text[6:].lstrip()
                                    return text
                                last_voice_t = now
                            else:
                                part = _json.loads(rec.PartialResult()).get("partial", "")
                                if part:
                                    heard_anything = True
                                    last_voice_t = now

                        elapsed = now - start_t
                        since_voice = now - last_voice_t

                        dynamic_end = end_silence + VOICE_THOUGHT_PADDING

                        # silence after we've heard something (with thought padding)
                        if elapsed >= min_listen and since_voice >= dynamic_end:
                            final = _json.loads(rec.FinalResult()).get("text", "").strip()
                            return final or (None if not heard_anything else "")

                        # hard cap
                        if elapsed >= phrase_time_limit:
                            final = _json.loads(rec.FinalResult()).get("text", "").strip()
                            return final or (None if not heard_anything else "")

            except Exception as e:
                print(f"(Vosk STT error: {e})")

        # SpeechRecognition fallback (uses Google Web Speech API)
        use_google = (
            self.has_sr and
            (self.preferred_backend in ("auto", "google"))
        )
        if use_google:
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                # Align pause/min-listen timings with our config knobs so we don't cut users off.
                r.pause_threshold = max(0.1, VOICE_END_SILENCE)
                r.non_speaking_duration = max(0.1, VOICE_END_SILENCE)
                r.phrase_threshold = max(0.1, VOICE_MIN_LISTEN)
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    if not quiet:
                        print("(Listening… speak now)")
                    audio = r.listen(
                        source,
                        timeout=None,
                        phrase_time_limit=phrase_time_limit
                    )

                try:
                    text = r.recognize_google(audio)
                    if text:
                        text = text.strip()
                        if allow_wake_prefix and text.lower().startswith("slash "):
                            text = text[6:].lstrip()
                        return text
                    return None
                except sr.UnknownValueError:
                    return None
                except sr.RequestError as e:
                    print(f"(SpeechRecognition request error: {e})")
                    return None

            except Exception as e:
                print(f"(SpeechRecognition error: {e})")
                return None

        print("(STT unavailable: install SpeechRecognition + PyAudio, or set VOSK_MODEL and install vosk + sounddevice.)")
        return None

    def detect_wake_word(self, quiet=True):
        """
        Wake word detection using full STT (not true keyword spotting).
        Listens for a short phrase and checks if it contains any wake word.
        Returns True if wake word detected, False otherwise.
        
        Note: This still uses full STT transcription, just with shorter timeouts.
        For true CPU savings, would need dedicated keyword spotting (KWS) model.
        Current benefit: shorter listens (3s vs 15s) = less processing of long utterances.
        
        Args:
            quiet: If True, suppress "Listening..." messages (default: True)
        """
        if not self.available or not WAKE_WORDS:
            return False

        # Use shorter timeouts for wake word detection (3s max vs 15s for full mode)
        # This reduces CPU by limiting how long we process audio, but still does full STT
        text = self.listen_once(
            end_silence=WAKE_WORD_END_SILENCE,
            min_listen=WAKE_WORD_MIN_LISTEN,
            phrase_time_limit=WAKE_WORD_PHRASE_LIMIT,
            allow_wake_prefix=False,
            quiet=quiet
        )

        if not text:
            return False

        # Normalize and check for wake words
        text_lower = text.lower().strip()
        
        # Check if any wake word appears in the transcribed text
        # This handles cases like "hey xai" or "xai what's the weather"
        for wake_word in WAKE_WORDS:
            if wake_word in text_lower:
                return True

        return False


# create singletons that repl.py can import
tts = TTS()
stt = STT()
