"""
Daniel Voice Assistant - FastAPI Backend
Main application entry point with API routes.
"""
import os
import io
import base64
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np

# Import our modules
from speech import (
    transcribe_audio,
    check_wake_word,
    extract_command,
    parse_intent,
    get_response_text,
    initialize_model as init_speech
)
from vision import (
    process_frame,
    initialize_model as init_vision,
    get_model as get_vision_model
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application state
app_state = {
    "camera_enabled": False,
    "mic_muted": False,
    "last_room_description": "",
    "last_messiness": None,
    "initialized": False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting Daniel Voice Assistant...")
    
    try:
        # Initialize speech model
        init_speech()
        logger.info("Speech model ready")
    except Exception as e:
        logger.warning(f"Speech model init failed: {e}")
    
    try:
        # Initialize vision model
        init_vision()
        logger.info("Vision model ready")
    except Exception as e:
        logger.warning(f"Vision model init failed: {e}")
    
    app_state["initialized"] = True
    logger.info("Daniel is ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Daniel...")


# Create FastAPI app
app = FastAPI(
    title="Daniel Voice Assistant",
    description="Local voice assistant with camera vision",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Daniel Voice Assistant",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "camera_enabled": app_state["camera_enabled"],
        "mic_muted": app_state["mic_muted"],
        "initialized": app_state["initialized"]
    }


@app.post("/stt")
async def speech_to_text(
    audio: UploadFile = File(...),
    wake_word_only: bool = Form(False)
):
    """
    Speech-to-Text endpoint.
    
    Args:
        audio: Audio file upload
        wake_word_only: If true, only check for wake word
    
    Returns:
        JSON with transcription, wake_word_detected, and command
    """
    try:
        # Read audio data
        audio_data = await audio.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio data")
        
        # Transcribe
        text, confidence = transcribe_audio(audio_data)
        
        if not text:
            return JSONResponse({
                "success": True,
                "text": "",
                "confidence": 0,
                "wake_word_detected": False,
                "command": "",
                "intent": "unknown"
            })
        
        # Check for wake word
        wake_word_detected = check_wake_word(text)
        
        # Extract command if wake word found
        command = ""
        intent = "unknown"
        
        if wake_word_detected:
            command = extract_command(text)
            intent, params = parse_intent(command)
            
            # Handle camera-dependent intents
            response_text = ""
            if intent in ["room_status", "check_dirty"]:
                # These need camera - respond that we're processing
                if app_state["camera_enabled"]:
                    response_text = "Boss, give me small time to check your room..."
                else:
                    response_text = "Boss, you need to turn on camera first. Say 'start camera'."
            else:
                response_text = get_response_text(intent)
            
            # Store last room info if available
            if app_state["last_room_description"]:
                response_text = get_response_text(
                    intent,
                    app_state["last_room_description"],
                    app_state["last_messiness"]
                )
        
        return {
            "success": True,
            "text": text,
            "confidence": confidence,
            "wake_word_detected": wake_word_detected,
            "command": command,
            "intent": intent,
            "response": response_text if wake_word_detected else ""
        }
        
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/command")
async def process_command(
    command: str = Form(...),
    frame_data: Optional[str] = Form(None)
):
    """
    Process a voice command with optional frame data.
    
    Args:
        command: The text command
        frame_data: Optional base64 encoded frame
    
    Returns:
        JSON with response and analysis results
    """
    try:
        # Parse intent
        intent, params = parse_intent(command)
        
        logger.info(f"Processing command: '{command}' -> intent: {intent}")
        
        # Handle state changes
        response_text = ""
        
        if intent == "stop_camera":
            app_state["camera_enabled"] = False
            response_text = get_response_text(intent)
            
        elif intent == "start_camera":
            app_state["camera_enabled"] = True
            response_text = get_response_text(intent)
            
        elif intent == "mute":
            app_state["mic_muted"] = True
            response_text = get_response_text(intent)
            
        elif intent == "unmute":
            app_state["mic_muted"] = False
            response_text = get_response_text(intent)
            
        elif intent == "help":
            response_text = get_response_text(intent)
            
        elif intent == "stop_listening":
            response_text = get_response_text(intent)
            
        elif intent in ["room_status", "check_dirty"]:
            # Process frame if available
            if frame_data:
                try:
                    # Decode base64 image
                    frame_bytes = base64.b64decode(frame_data)
                    
                    # Process frame
                    result = process_frame(frame_bytes)
                    
                    # Store results
                    app_state["last_room_description"] = result["description"]
                    app_state["last_messiness"] = result["messiness"]
                    
                    # Generate response
                    response_text = get_response_text(
                        intent,
                        result["description"],
                        result["messiness"]
                    )
                    
                    return {
                        "success": True,
                        "intent": intent,
                        "response": response_text,
                        "description": result["description"],
                        "messiness": result["messiness"],
                        "detections": result["detections"]
                    }
                    
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    response_text = "Boss, I get problem来分析 your room. Make you try again."
            else:
                # No frame - use last known state
                if app_state["last_room_description"]:
                    response_text = get_response_text(
                        intent,
                        app_state["last_room_description"],
                        app_state["last_messiness"]
                    )
                else:
                    response_text = "Boss, you need to turn on camera first. Say 'start camera'."
        else:
            response_text = get_response_text(intent)
        
        return {
            "success": True,
            "intent": intent,
            "response": response_text
        }
        
    except Exception as e:
        logger.error(f"Command processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Analyze a single frame for object detection and messiness.
    
    Args:
        file: Image file upload
    
    Returns:
        JSON with detections, messiness, and description
    """
    try:
        # Read image data
        image_data = await file.read()
        
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image data")
        
        # Process frame
        result = process_frame(image_data)
        
        # Store for later use
        app_state["last_room_description"] = result["description"]
        app_state["last_messiness"] = result["messiness"]
        
        return {
            "success": True,
            "detections": result["detections"],
            "messiness": result["messiness"],
            "description": result["description"],
            "annotated_image": result["annotated_image"]
        }
        
    except Exception as e:
        logger.error(f"Frame analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/camera/toggle")
async def toggle_camera(enabled: bool = Form(...)):
    """
    Toggle camera state.
    
    Args:
        enabled: True to enable, False to disable
    
    Returns:
        JSON with new state
    """
    app_state["camera_enabled"] = enabled
    
    response_text = get_response_text(
        "start_camera" if enabled else "stop_camera"
    )
    
    return {
        "success": True,
        "camera_enabled": enabled,
        "response": response_text
    }


@app.post("/mic/toggle")
async def toggle_mic(muted: bool = Form(...)):
    """
    Toggle microphone state.
    
    Args:
        muted: True to mute, False to unmute
    
    Returns:
        JSON with new state
    """
    app_state["mic_muted"] = muted
    
    response_text = get_response_text(
        "mute" if muted else "unmute"
    )
    
    return {
        "success": True,
        "mic_muted": muted,
        "response": response_text
    }


@app.get("/status")
async def get_status():
    """
    Get current assistant status.
    
    Returns:
        JSON with current state
    """
    return {
        "camera_enabled": app_state["camera_enabled"],
        "mic_muted": app_state["mic_muted"],
        "last_room_description": app_state["last_room_description"],
        "last_messiness": app_state["last_messiness"],
        "initialized": app_state["initialized"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
