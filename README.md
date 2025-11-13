# Local Assistant

A voice-enabled, on-device AI assistant with face recognition, calendar integration, and navigation features.

## Features

- **Voice Interface**: Speech-to-text (STT) and text-to-speech (TTS) support
- **Face Recognition**: Secure authentication using face recognition
- **Calendar Integration**: Google Calendar access for agenda queries
- **Navigation**: Save and navigate to frequently visited places
- **Multi-Persona Support**: Different personas (Aldridge, Professor, Guest) with distinct styles
- **On-Device LLM**: Uses local Ollama models for privacy

## Prerequisites

- Python 3.11 or higher
- Ollama installed and running locally
- Google Cloud Project with Calendar API enabled (for calendar features)
- Camera access (for face recognition)
- Microphone access (for voice input)

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd SeniorDesign
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r ../requirements.txt
```

4. Install additional dependencies for voice features:
```bash
pip install pyttsx3 speechrecognition pyaudio
```

For offline STT (optional):
- Download a Vosk model and set `VOSK_MODEL` environment variable

## Configuration

### Google Calendar Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Calendar API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download the credentials JSON file
6. Rename it to `credentials.json` and place it in the `SeniorDesign/` directory
7. On first run, the app will open a browser for authentication and save tokens

**Note**: The `credentials.json` file is not included in this repository for security. Each collaborator needs their own Google Cloud project credentials.

### Face Recognition Setup

1. Run the assistant
2. Use `/setup_profile <Name>` to enroll a new face
3. The face data will be saved in `face_dataset/` (not tracked in git)

### Environment Variables (Optional)

- `STRICT_AUTH`: Set to `"1"` for strict face recognition (default: `"1"`)
- `FACE_AUTORECOG`: Auto-recognize face on startup (default: `"1"`)
- `VOICE_MODE`: Enable voice mode by default (default: `"1"`)
- `CAMERA_INDEX`: Force specific camera index (default: `"-1"` for auto)
- `VOSK_MODEL`: Path to Vosk model for offline STT (optional)

## Usage

Run the assistant:
```bash
python main.py
```

### Commands

- `/recognize` - Try face recognition to unlock
- `/setup_profile <Name>` - Enroll a new face
- `/login <Name>` - Dev bypass login (testing only)
- `/switch` - Switch between personas
- `/agenda` - View upcoming calendar events
- `/nav <place>` - Navigate to a saved place
- `/setplace <key> = <address>` - Save a navigation destination
- `/voice on|off|status` - Control voice mode
- `/tts on|off` - Enable/disable text-to-speech
- `/model <name>` - Change Ollama model
- `/help` - Show all commands
- `/exit` - Quit the assistant

## Project Structure

```
SeniorDesign/
├── main.py              # Entry point
├── repl.py              # Main REPL loop and command handling
├── config.py            # Configuration constants
├── llm.py               # LLM integration (Ollama)
├── personas.py          # Persona definitions
├── profiles.py          # Profile management
├── tts_stt.py           # Text-to-speech and speech-to-text
├── face_auth.py         # Face recognition and authentication
├── calendar_tools.py    # Google Calendar integration
├── nav.py               # Navigation helpers
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Security Notes

- **Never commit**:
  - `token*.json` files (OAuth tokens)
  - `credentials.json` (Google OAuth credentials)
  - `profiles.json` (personal data)
  - `face_dataset/` (biometric data)

- Each collaborator should:
  1. Create their own Google Cloud project
  2. Generate their own `credentials.json`
  3. Run the app once to authenticate and generate tokens
  4. Keep their credentials secure and never share them

## Troubleshooting

### Camera not working
- Close other apps using the camera (Zoom, Teams, etc.)
- Check camera permissions in system settings
- Try setting `CAMERA_INDEX` environment variable

### Voice recognition not working
- Install `pyaudio`: `pip install pyaudio`
- On macOS, you may need: `brew install portaudio`
- For offline STT, download a Vosk model and set `VOSK_MODEL`

### Calendar errors
- Ensure `credentials.json` is in the correct location
- Check that Google Calendar API is enabled in your project
- Re-authenticate by deleting `token*.json` files and running again


