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
import re
import queue

# Load environment variables
load_dotenv()

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
ENERGY_THRESHOLD = 0.01  # Adjust this value based on your microphone
SILENCE_DURATION = 1.0  # Stop after 1 second of silence
TOKEN_SERVER_URL = "http://localhost:5000"

class VoiceAssistant:
    def __init__(self):
        # Load and validate API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            st.error("OpenAI API key not found. Please check your .env file.")
            st.stop()
            
        self.tts_api_key = os.getenv("TTS_API_KEY")
        self.tts_api_base_url = os.getenv("TTS_API_BASE_URL")
        
        self.alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        
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
        """Send audio to OpenAI Whisper for transcription"""
        try:
            # Ensure audio_data is a numpy array
            if not isinstance(audio_data, np.ndarray):
                st.error("Audio data must be a numpy array")
                return None
            
            # Normalize audio data to int16 range (-32768 to 32767)
            if audio_data.dtype != np.int16:
                # First normalize to float between -1 and 1
                audio_data = audio_data.astype(np.float32)
                if np.abs(audio_data).max() > 1.0:
                    audio_data = audio_data / np.abs(audio_data).max()
                # Then convert to int16
                audio_data = (audio_data * 32767).astype(np.int16)
            
            # Create a BytesIO object to store the WAV file
            wav_buffer = io.BytesIO()
            
            # Write the audio data to the buffer as a WAV file
            write(wav_buffer, SAMPLE_RATE, audio_data)
            
            # Reset buffer position
            wav_buffer.seek(0)
            
            # Debug information
            st.write(f"Audio data shape: {audio_data.shape}")
            st.write(f"Audio data type: {audio_data.dtype}")
            st.write(f"Audio data min/max: {audio_data.min()}/{audio_data.max()}")
            st.write(f"WAV buffer size: {len(wav_buffer.getvalue())} bytes")
            
            # Verify WAV buffer is not empty
            if len(wav_buffer.getvalue()) == 0:
                st.error("Generated WAV file is empty")
                return None
            
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            files = {
                "file": ("audio.wav", wav_buffer, "audio/wav"),
                "model": (None, "whisper-1")
            }
            
            # Debug: Print request details
            st.write("Sending request to OpenAI API...")
            
            response = requests.post(url, headers=headers, files=files)
            
            if response.status_code != 200:
                st.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return None
                
            result = response.json()
            if "text" in result:
                return result["text"]
            st.error("Could not find transcript in response")
            return None
            
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            st.error(f"Error type: {type(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None

    def get_stock_price(self, symbol):
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.alphavantage_api_key
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            try:
                price = data["Global Quote"]["05. price"]
                return price
            except KeyError:
                return None
        else:
            return None

    def get_llm_response(self, text):
        """Get response from OpenAI, with Alpha Vantage integration for stock prices."""
        if not text:
            return "I couldn't understand the audio. Please try again."
        stock_match = re.search(r"stock price of ([A-Za-z]+)", text, re.IGNORECASE)
        if stock_match and self.alphavantage_api_key:
            symbol = stock_match.group(1).upper()
            price = self.get_stock_price(symbol)
            if price:
                text = f"The current price of {symbol} is {price} USD. {text}"
            else:
                text = f"I could not retrieve the price for {symbol}. {text}"
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a financial assistant. Always provide market data in numbers when possible."},
                    {"role": "user", "content": text}
                ],
                "max_tokens": 100
            }
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
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
                return "Sorry, I couldn't understand the response from the AI."
        except Exception as e:
            st.error(f"Error getting OpenAI response: {str(e)}")
            return "Sorry, I encountered an error while processing your request."

    def text_to_speech(self, text):
        """Convert text to speech using OpenAI TTS (voice: alloy)"""
        if not text:
            return None
        try:
            headers = {
                "Authorization": f"Bearer {self.tts_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "tts-1",
                "input": text,
                "voice": "alloy"
            }
            response = requests.post(
                self.tts_api_base_url,
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
    """Record audio from microphone with energy-based voice activity detection"""
    st.write("Recording... (Speak to start, silence to stop)")
    
    # Create a queue for audio chunks
    audio_queue = queue.Queue()
    recording = True
    audio_chunks = []
    silence_start = None
    
    def audio_callback(indata, frames, time, status):
        if status:
            st.error(f"Audio callback error: {status}")
        audio_queue.put(indata.copy())
    
    # Start recording
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )
    
    with stream:
        while recording:
            # Get audio chunk
            chunk = audio_queue.get()
            
            # Calculate energy (RMS) of the chunk
            energy = np.sqrt(np.mean(chunk**2))
            
            # Check if chunk contains speech
            is_speech = energy > ENERGY_THRESHOLD
            
            if is_speech:
                audio_chunks.append(chunk)
                silence_start = None
            else:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    recording = False
                
                # Still add the chunk to maintain continuity
                audio_chunks.append(chunk)
            
            # Update progress
            st.write(f"Recording... {'Speaking' if is_speech else 'Silence'} (Energy: {energy:.4f})")
            await asyncio.sleep(0.01)  # Small delay to prevent UI freezing
    
    # Combine all chunks and convert to int16
    if audio_chunks:
        audio_data = np.concatenate(audio_chunks)
        # Convert to int16
        audio_data = (audio_data * 32767).astype(np.int16)
        st.write("Recording complete!")
        return audio_data
    else:
        st.error("No audio recorded")
        return None

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
                
                if audio_data is not None:
                    # Process audio
                    st.write("Starting transcription...")
                    transcription_start = time.time()
                    st.session_state.transcript = st.session_state.assistant.transcribe_audio(audio_data)
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
                    audio_bytes = st.session_state.audio_response
                    b64 = base64.b64encode(audio_bytes).decode()
                    md = f'''
                        <audio controls autoplay>
                            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                    '''
                    st.markdown(md, unsafe_allow_html=True)
        
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