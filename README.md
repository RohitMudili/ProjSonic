# Sonic - AI Voice Assistant

A real-time voice conversation application with AI capabilities, built using Streamlit and various AI services.

## Features

- Real-time voice recording and transcription
- AI-powered conversation using OpenAI's GPT-3.5
- Text-to-speech response using ElevenLabs
- LiveKit integration for real-time audio streaming
- Latency metrics tracking
- Session history logging

## Prerequisites

- Python 3.8 or higher
- Deepgram API key
- OpenAI API key
- ElevenLabs API key
- LiveKit server (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RohitMudili/ProjSonic.git
cd ProjSonic
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
TTS_API_KEY=your_elevenlabs_api_key
TTS_API_BASE_URL=https://api.elevenlabs.io/v1
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Click "Start Recording" to begin a voice conversation with the AI assistant

## Project Structure

- `app.py`: Main application file
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked in git)
- `transcriptions.txt`: Log of conversations (generated during use)

## License

MIT License - see LICENSE file for details 