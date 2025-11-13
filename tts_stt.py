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
        blocksize=8000
    ):
        """
        Record one utterance, return the recognized text (string),
        or None if we got nothing.
        """
        # Offline Vosk path
        if self.has_vosk and self.vosk_model_path:
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
                                    return text
                                last_voice_t = now
                            else:
                                part = _json.loads(rec.PartialResult()).get("partial", "")
                                if part:
                                    heard_anything = True
                                    last_voice_t = now

                        elapsed = now - start_t
                        since_voice = now - last_voice_t

                        # silence after we've heard something
                        if elapsed >= min_listen and since_voice >= end_silence:
                            final = _json.loads(rec.FinalResult()).get("text", "").strip()
                            return final or (None if not heard_anything else "")

                        # hard cap
                        if elapsed >= phrase_time_limit:
                            final = _json.loads(rec.FinalResult()).get("text", "").strip()
                            return final or (None if not heard_anything else "")

            except Exception as e:
                print(f"(Vosk STT error: {e})")

        # SpeechRecognition fallback (uses Google Web Speech API)
        if self.has_sr:
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    print("(Listening… speak now)")
                    audio = r.listen(source, timeout=10, phrase_time_limit=phrase_time_limit)

                try:
                    text = r.recognize_google(audio)
                    return text.strip() if text else None
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


# create singletons that repl.py can import
tts = TTS()
stt = STT()
