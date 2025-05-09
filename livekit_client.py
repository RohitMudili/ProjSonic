import os
import asyncio
from livekit import rtc
import numpy as np
from typing import Optional, Callable

class LiveKitClient:
    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self.room = rtc.Room()
        self.audio_callback = None
        self.is_connected = False
        
        # Set up event handlers
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)
        self.room.on("disconnected", self._on_disconnected)
        
    async def connect(self):
        """Connect to LiveKit room"""
        try:
            await self.room.connect(self.url, self.token)
            self.is_connected = True
            print("Connected to LiveKit room")
        except Exception as e:
            print(f"Failed to connect to LiveKit: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from LiveKit room"""
        if self.is_connected:
            await self.room.disconnect()
            self.is_connected = False
            
    def set_audio_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for received audio frames"""
        self.audio_callback = callback
        
    def _on_track_subscribed(self, track, publication, participant):
        """Handle new track subscription"""
        if track.kind == rtc.TrackKind.AUDIO:
            track.on("frame", self._on_audio_frame)
                
    def _on_track_unsubscribed(self, track, publication, participant):
        """Handle track unsubscription"""
        if track.kind == rtc.TrackKind.AUDIO:
            track.off("frame", self._on_audio_frame)
                
    def _on_audio_frame(self, frame):
        """Process incoming audio frames"""
        if self.audio_callback:
            # Convert audio frame to numpy array
            samples = np.frombuffer(frame.data, dtype=np.int16)
            self.audio_callback(samples)
            
    def _on_disconnected(self):
        """Handle disconnection"""
        self.is_connected = False
        print("Disconnected from LiveKit room")
        
    async def publish_audio(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Publish audio data to the room"""
        if not self.is_connected:
            raise Exception("Not connected to LiveKit room")
            
        # Create audio track
        audio_track = rtc.LocalAudioTrack.create_audio_track(
            "microphone",
            rtc.AudioSourceOptions(sample_rate=sample_rate)
        )
        
        # Add track to room
        await self.room.local_participant.publish_track(audio_track)
        
        # Send audio data
        audio_track.write(audio_data.tobytes()) 