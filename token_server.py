from flask import Flask, jsonify
import os
from dotenv import load_dotenv
import uuid
import jwt
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get LiveKit credentials from environment variables
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")

def generate_test_token():
    """Generate a test JWT token for LiveKit"""
    # Generate a unique user ID
    user_id = str(uuid.uuid4())
    
    # Create token payload
    payload = {
        "sub": user_id,
        "name": f"User-{user_id[:8]}",
        "iss": LIVEKIT_API_KEY,
        "nbf": int(time.time()),
        "exp": int(time.time()) + 3600,  # Token expires in 1 hour
        "video": {
            "roomJoin": True,
            "room": "voice-assistant"
        }
    }
    
    # Generate token
    token = jwt.encode(
        payload,
        LIVEKIT_API_SECRET,
        algorithm="HS256"
    )
    
    return token

@app.route('/generate-token', methods=['GET'])
def get_token():
    try:
        token = generate_test_token()
        
        # Ensure URL is properly formatted
        if not LIVEKIT_URL.startswith(('ws://', 'wss://')):
            livekit_url = f"wss://{LIVEKIT_URL}"
        else:
            livekit_url = LIVEKIT_URL
            
        return jsonify({
            "token": token,
            "url": livekit_url,
            "room": "voice-assistant"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 