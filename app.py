import streamlit as st
import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import requests
from pydub import AudioSegment
import io
import base64
import asyncio
from livekit_client import LiveKitClient

# Load environment variables
load_dotenv()

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
RECORD_SECONDS = 5
TOKEN_SERVER_URL = "http://localhost:5000"  # Update this if your token server is hosted elsewhere

class VoiceAssistant:
    def __init__(self):
        # Load and validate API keys
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.deepgram_api_key:
            st.error("Deepgram API key not found. Please check your .env file.")
            st.stop()
            
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            st.error("OpenAI API key not found. Please check your .env file.")
            st.stop()
            
        self.tts_api_key = os.getenv("TTS_API_KEY")
        self.tts_api_base_url = os.getenv("TTS_API_BASE_URL")
        
        # Initialize LiveKit client (will be set up when connecting)
        self.livekit_client = None
        self.livekit_connected = False

    async def connect_livekit(self):
        """Connect to LiveKit server using token from our token server"""
        try:
            # Get token from token server
            response = requests.get(f"{TOKEN_SERVER_URL}/generate-token")
            if response.status_code != 200:
                st.error(f"Failed to get LiveKit token: {response.text}")
                return
                
            token_data = response.json()
            
            # Ensure URL is properly formatted
            livekit_url = token_data["url"]
            if not livekit_url.startswith(('ws://', 'wss://')):
                livekit_url = f"wss://{livekit_url}"
            
            # Initialize LiveKit client with the token
            self.livekit_client = LiveKitClient(livekit_url, token_data["token"])
            await self.livekit_client.connect()
            self.livekit_connected = True
            st.success("Connected to LiveKit server")
        except Exception as e:
            st.error(f"Failed to connect to LiveKit: {e}")
            self.livekit_connected = False

    def transcribe_audio(self, audio_data):
        """Send audio to Deepgram for transcription"""
        try:
            # Use REST API for transcription
            url = "https://api.deepgram.com/v1/listen"
            headers = {
                "Authorization": f"Token {self.deepgram_api_key}",
                "Content-Type": "audio/wav"
            }
            
            st.write("Sending request to Deepgram...")
            response = requests.post(url, headers=headers, data=audio_data)
            
            if response.status_code != 200:
                st.error(f"Deepgram API error: {response.status_code} - {response.text}")
                return None
                
            result = response.json()
            st.write("Received response from Deepgram")
            
            # Debug: Print response
            st.write("Response:", result)
            
            if "results" in result and result["results"].get("channels"):
                channels = result["results"]["channels"]
                if channels and channels[0].get("alternatives"):
                    return channels[0]["alternatives"][0].get("transcript", "")
            
            st.error("Could not find transcript in response")
            return None
            
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            return None

    def get_llm_response(self, text):
        """Get response from OpenAI"""
        if not text:
            return "I couldn't understand the audio. Please try again."
            
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": text}
                ],
                "max_tokens": 150
            }
            
            st.write("Sending request to OpenAI...")
            st.write("Using endpoint: https://api.openai.com/v1/chat/completions")
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            st.write(f"OpenAI Response Status: {response.status_code}")
            st.write(f"OpenAI Response Text: {response.text[:200]}...")  # Show first 200 chars for debugging
            
            if response.status_code != 200:
                st.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return "Sorry, I encountered an error while processing your request."
            
            try:
                response_json = response.json()
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"]
                else:
                    st.error(f"Unexpected OpenAI response format: {response_json}")
                    return "Sorry, I couldn't process the response properly."
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse OpenAI response: {str(e)}")
                st.error(f"Raw response: {response.text[:200]}...")
                return "Sorry, I couldn't understand the response from the AI."
                
        except Exception as e:
            st.error(f"Error getting OpenAI response: {str(e)}")
            return "Sorry, I encountered an error while processing your request."

    def text_to_speech(self, text):
        """Convert text to speech using ElevenLabs"""
        if not text:
            return None
            
        try:
            headers = {
                "xi-api-key": self.tts_api_key,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            response = requests.post(
                f"{self.tts_api_base_url}/text-to-speech/21m00Tcm4TlvDq8ikWAM",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.content
            else:
                st.error(f"TTS Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            st.error(f"Error in text-to-speech: {str(e)}")
            return None

async def record_audio(assistant):
    """Record audio from microphone and stream to LiveKit if available"""
    st.write("Recording...")
    
    if assistant.livekit_client and assistant.livekit_connected:
        # Use LiveKit for real-time streaming
        audio_data = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16,
            callback=lambda indata, frames, time, status: 
                asyncio.run(assistant.livekit_client.publish_audio(indata))
        )
    else:
        # Fallback to regular recording
        audio_data = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16
        )
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Update progress bar during recording
    for i in range(RECORD_SECONDS):
        progress_bar.progress((i + 1) / RECORD_SECONDS)
        status_text.text(f"Recording... {RECORD_SECONDS - i} seconds remaining")
        await asyncio.sleep(1)
    
    sd.wait()  # Wait until recording is finished
    status_text.text("Recording complete!")
    progress_bar.progress(1.0)
    
    return audio_data

def save_audio(audio_data, filename):
    """Save audio data to WAV file"""
    try:
        write(filename, SAMPLE_RATE, audio_data)
        st.write(f"Audio saved to {filename}")
        
        # Debug: Print audio file details
        st.write(f"Audio shape: {audio_data.shape}")
        st.write(f"Sample rate: {SAMPLE_RATE}")
        st.write(f"Data type: {audio_data.dtype}")
    except Exception as e:
        st.error(f"Error saving audio: {str(e)}")

def save_transcription(transcript, response):
    """Save transcription and response to a file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("transcriptions.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== {timestamp} ===\n")
            f.write(f"Transcription: {transcript}\n")
            f.write(f"Response: {response}\n")
            f.write("=" * 50 + "\n")
        st.success("Transcription saved successfully!")
    except Exception as e:
        st.error(f"Error saving transcription: {str(e)}")

async def main():
    st.title("AI Voice Assistant")
    st.write("Real-time voice conversation with AI")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = VoiceAssistant()
    if 'transcript' not in st.session_state:
        st.session_state.transcript = None
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'audio_response' not in st.session_state:
        st.session_state.audio_response = None
    if 'latencies' not in st.session_state:
        st.session_state.latencies = {
            'transcription': None,
            'llm': None,
            'tts': None,
            'total': None
        }
    
    # Display API key status
    st.sidebar.write("API Key Status:")
    
    # Deepgram
    deepgram_key = st.session_state.assistant.deepgram_api_key
    if deepgram_key:
        st.sidebar.success(f"Deepgram API key: {deepgram_key[:8]}...")
    else:
        st.sidebar.error("No Deepgram API key found")
    
    # OpenAI
    openai_key = st.session_state.assistant.openai_api_key
    if openai_key:
        st.sidebar.success(f"OpenAI API key: {openai_key[:8]}...")
    else:
        st.sidebar.error("No OpenAI API key found")
    
    # TTS
    tts_key = st.session_state.assistant.tts_api_key
    tts_url = st.session_state.assistant.tts_api_base_url
    if tts_key and tts_url:
        st.sidebar.success(f"TTS API configured")
    else:
        st.sidebar.error("TTS API not configured")
        
    # LiveKit
    if st.sidebar.button("Connect to LiveKit"):
        await st.session_state.assistant.connect_livekit()
    st.sidebar.write("LiveKit Status:", 
                    "Connected" if st.session_state.assistant.livekit_connected else "Disconnected")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input")
        if st.button("Start Recording"):
            try:
                start_time = time.time()
                st.write("Starting recording...")
                audio_data = await record_audio(st.session_state.assistant)
                
                # Save audio
                save_audio(audio_data, "temp.wav")
                
                # Process audio
                with open("temp.wav", "rb") as f:
                    audio_bytes = f.read()
                    st.write(f"Audio file size: {len(audio_bytes)} bytes")
                
                # Transcribe
                st.write("Starting transcription...")
                transcription_start = time.time()
                st.session_state.transcript = st.session_state.assistant.transcribe_audio(audio_bytes)
                st.session_state.latencies['transcription'] = time.time() - transcription_start
                
                if st.session_state.transcript:
                    st.write("Transcription:", st.session_state.transcript)
                    
                    # Get LLM response
                    st.write("Getting AI response...")
                    llm_start = time.time()
                    st.session_state.response = st.session_state.assistant.get_llm_response(st.session_state.transcript)
                    st.session_state.latencies['llm'] = time.time() - llm_start
                    
                    # Convert to speech
                    st.write("Converting to speech...")
                    tts_start = time.time()
                    st.session_state.audio_response = st.session_state.assistant.text_to_speech(st.session_state.response)
                    st.session_state.latencies['tts'] = time.time() - tts_start
                    
                    # Calculate total time
                    st.session_state.latencies['total'] = time.time() - start_time
                    
                    # Save transcription and response
                    save_transcription(st.session_state.transcript, st.session_state.response)
            except Exception as e:
                st.error(f"Error in main processing: {str(e)}")
    
    with col2:
        st.header("Output")
        if st.session_state.transcript:
            st.subheader("Transcription")
            st.write(st.session_state.transcript)
            
            if st.session_state.response:
                st.subheader("AI Response")
                st.write(st.session_state.response)
                
                if st.session_state.audio_response:
                    st.subheader("Audio Response")
                    st.audio(st.session_state.audio_response, format="audio/mp3")
        
        # Display latencies
        if st.session_state.latencies['total'] is not None:
            st.subheader("Latency Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Transcription Latency", f"{st.session_state.latencies['transcription']:.2f}s")
                st.metric("LLM Response Latency", f"{st.session_state.latencies['llm']:.2f}s")
            
            with col2:
                st.metric("TTS Generation Latency", f"{st.session_state.latencies['tts']:.2f}s")
                st.metric("Total Processing Latency", f"{st.session_state.latencies['total']:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())