"""
Speech-to-Text processing using Faster Whisper.
Handles wake word detection and command transcription.
"""
import os
import torch
from faster_whisper import WhisperModel
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Global model instance
_model: Optional[WhisperModel] = None

# Wake word configuration
WAKE_WORD = "hey daniel"
WAKE_WORD_CONFIDENCE = 0.5

# Model size - smaller for speed, larger for accuracy
# Options: tiny, base, small, medium, large
MODEL_SIZE = "base"


def get_model() -> WhisperModel:
    """
    Get or initialize the Whisper model.
    Uses GPU if available, otherwise falls back to CPU.
    """
    global _model
    
    if _model is not None:
        return _model
    
    # Determine compute type based on hardware
    if torch.cuda.is_available():
        compute_type = "float16"
        logger.info("Using GPU for Whisper inference")
    else:
        compute_type = "int8"
        logger.info("Using CPU for Whisper inference")
    
    # Download and load model
    logger.info(f"Loading Whisper {MODEL_SIZE} model...")
    _model = WhisperModel(
        MODEL_SIZE,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type=compute_type
    )
    logger.info("Whisper model loaded successfully")
    
    return _model


def transcribe_audio(audio_data: bytes) -> Tuple[str, float]:
    """
    Transcribe audio data to text.
    
    Args:
        audio_data: Raw audio bytes
        
    Returns:
        Tuple of (transcribed_text, confidence_score)
    """
    import tempfile
    import numpy as np
    import wave
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data)
        tmp_path = tmp_file.name
    
    try:
        model = get_model()
        
        # Transcribe with word-level timestamps disabled for speed
        segments, info = model.transcribe(
            tmp_path,
            language="en",
            beam_size=1,  # Faster decoding
            vad_filter=True,  # Voice activity detection
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Get full transcription
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        full_text = " ".join(text_parts).strip()
        confidence = info.language_probability if info else 0.0
        
        logger.info(f"Transcription: '{full_text}' (confidence: {confidence:.2f})")
        
        return full_text, confidence
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return "", 0.0
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def check_wake_word(text: str) -> bool:
    """
    Check if the transcribed text contains the wake word.
    
    Args:
        text: Transcribed text to check
        
    Returns:
        True if wake word is detected
    """
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    # Check for wake word
    if WAKE_WORD in text_lower:
        logger.info(f"Wake word '{WAKE_WORD}' detected!")
        return True
    
    return False


def extract_command(text: str) -> str:
    """
    Extract the command after the wake word.
    
    Args:
        text: Full transcribed text
        
    Returns:
        Command portion after wake word
    """
    if not text:
        return ""
    
    text_lower = text.lower().strip()
    
    # Remove wake word from the beginning
    if WAKE_WORD in text_lower:
        command = text_lower.replace(WAKE_WORD, "", 1).strip()
        return command
    
    return text_lower


def parse_intent(command: str) -> Tuple[str, str]:
    """
    Parse command into intent and parameters.
    
    Args:
        command: The command text to parse
        
    Returns:
        Tuple of (intent, parameters)
    """
    command = command.lower().strip()
    
    # Define intent patterns
    intents = {
        "room_status": [
            "what's happening in my room",
            "what is happening in my room",
            "what's in my room",
            "describe my room",
            "describe room",
            "what do you see",
            "what do you see in my room"
        ],
        "check_dirty": [
            "check if my room dirty",
            "check if my room is dirty",
            "is my room dirty",
            "is my room clean",
            "how messy is my room",
            "is my room messy",
            "room dirty",
            "room clean"
        ],
        "stop_camera": [
            "stop camera",
            "turn off camera",
            "close camera",
            "disable camera"
        ],
        "start_camera": [
            "start camera",
            "turn on camera",
            "open camera",
            "enable camera"
        ],
        "mute": [
            "mute",
            "mute microphone",
            "mute mic"
        ],
        "unmute": [
            "unmute",
            "unmute microphone",
            "unmute mic"
        ],
        "help": [
            "help",
            "what can you do",
            "commands",
            "list commands"
        ],
        "stop_listening": [
            "stop listening",
            "stop",
            "go to sleep",
            "goodbye",
            "bye"
        ]
    }
    
    # Match command to intent
    for intent, patterns in intents.items():
        for pattern in patterns:
            if pattern in command:
                logger.info(f"Matched intent: {intent}")
                return intent, command.replace(pattern, "", 1).strip()
    
    # Default to room_status if unclear
    if command:
        logger.info(f"No specific intent matched, treating as room_status")
        return "room_status", command
    
    return "unknown", ""


def get_response_text(intent: str, room_description: str = "", messiness: dict = None) -> str:
    """
    Generate response text based on intent.
    Uses casual Nigerian Pidgin English style.
    
    Args:
        intent: The recognized intent
        room_description: Description of the room (for room_status)
        messiness: Messiness detection results
        
    Returns:
        Response text to speak
    """
    import random
    
    responses = {
        "room_status": [
            "Yes boss, I dey. {description}",
            "Boss, {description}",
            "I see {description}",
            "Your room be like {description}"
        ],
        "check_dirty": {
            "clean": [
                "Boss your room clean abeg! No stress.",
                "I no see wahala. Room fine oo.",
                "Your room tidy well well! Good boy.",
                "Room clean like new. You tried!"
            ],
            "messy": [
                "Boss your room messy oo, you for clean am.",
                "E choke! Room too dirty. Clean am na!",
                "Your room need attention oo. Too much things for floor.",
                "Abeg, room messy well. You go clean am now?"
            ],
            "moderate": [
                "Room nor clean nor dirty. E be like moderate.",
                "Room dey half clean. You fit tidy am small.",
                "E get as e be. Room need small cleaning."
            ]
        },
        "stop_camera": [
            "Camera don close boss.",
            "Camera off. I no go look again.",
            "Done! Camera dey off now."
        ],
        "start_camera": [
            "Camera don open boss.",
            "Camera on! I go watch your room now.",
            "Ready! Camera dey work."
        ],
        "mute": [
            "Mic don mute boss.",
            "I go deaf now. Muted!",
            "Done! I no go hear anything."
        ],
        "unmute": [
            "Mic don unmute boss.",
            "I dey hear now! Unmuted!",
            "Ready! I go listen for you."
        ],
        "help": [
            "Boss, you fit tell me: 'what's happening in my room' for room description, 'check if my room dirty' for messiness check, 'stop camera' or 'start camera' for camera control, 'mute' or 'unmute' for mic control, or 'help' to see this message.",
            "I understand these commands: room description, check dirty, camera on/off, mute/unmute, and help.",
            "You fit ask me about your room, check if e dirty, control camera, or mute. Just say the word boss!"
        ],
        "stop_listening": [
            "Goodbye boss! I go sleep now.",
            "Bye bye! Call me when you need me.",
            "I go rest. Say 'Hey Daniel' when you ready."
        ],
        "unknown": [
            "Sorry boss, I no understand. Try again.",
            "E dey confuse me. Repeat am please.",
            "I no get am. Help me understand better."
        ]
    }
    
    # Get response based on intent
    if intent == "room_status":
        description = room_description if room_description else "nothing special dey happen"
        response = random.choice(responses["room_status"]).format(description=description)
        
    elif intent == "check_dirty" and messiness:
        level = messiness.get("level", "moderate")
        response = random.choice(responses["check_dirty"].get(level, responses["check_dirty"]["moderate"]))
        
    else:
        response = random.choice(responses.get(intent, responses["unknown"]))
    
    return response


def initialize_model():
    """
    Initialize the model on startup.
    Call this when the app starts.
    """
    try:
        get_model()
        logger.info("Speech model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize speech model: {e}")
        raise
