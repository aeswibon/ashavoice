import logging
import os
import tempfile
from contextlib import asynccontextmanager  # Import asynccontextmanager
from pathlib import Path

import torch
import whisper
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

app = FastAPI()


# Class to manage the Whisper model state
class WhisperModelManager:
    def __init__(self):
        self.model = None


whisper_model_manager = WhisperModelManager()


# Define the lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    whisper_model_manager.model = None
    """""
    Handles startup and shutdown events for the application.
    Loads the Whisper model on startup and performs cleanup on shutdown (if any).
    """
    try:
        model_name = os.getenv("WHISPER_MODEL_NAME", "base")
        model_dir = os.getenv("WHISPER_MODEL_DIR", str(MODEL_DIR))

        logger.info("Loading Whisper model: %s from cache: %s", model_name, model_dir)

        # Ensure the cache directory exists and is writable
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s for Whisper model.", device)

        whisper_model_manager.model = whisper.load_model(
            model_name, download_root=model_dir, device=device
        )

        logger.info("Whisper model '%s' loaded successfully.", model_name)
    except Exception as e:
        logger.exception("Error loading Whisper model during startup: %s", e)
        error_message = (
            "Failed to load Whisper model. Ensure the model name is correct and "
            "the model cache directory is accessible. Check logs for more details."
        )
        raise RuntimeError(error_message) from e

    # Yield control to the application, allowing it to serve requests
    yield

    if whisper_model_manager.model:
        del whisper_model_manager.model
        whisper_model_manager.model = None
        logger.info("Whisper model resources released.")


# Initialize the FastAPI app with the lifespan context manager
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    if whisper_model_manager.model:
        return {"status": "ok", "model_loaded": True}
    return {
        "status": "ok",
        "model_loaded": False,
        "message": "Model not yet loaded or failed to load",
    }


@app.post("/transcribe/")
async def transcribe_audio(audio_file: UploadFile):
    """Transcribes an audio file."""

    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    if not whisper_model_manager.model:
        raise HTTPException(status_code=503, detail="Whisper model not loaded yet.")

    # Check if it's an audio file (basic validation)
    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        # Allow some common audio file extensions even if content-type is not set properly
        allowed_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
        file_ext = Path(audio_file.filename or "").suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file.",
            )

    try:
        # Read the audio file into memory
        audio_bytes = await audio_file.read()
        logger.info(
            "Received audio file: %s, size: %d bytes",
            audio_file.filename,
            len(audio_bytes),
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        try:
            # Perform transcription
            # Use fp16=False for CPU, fp16=True for GPU if available
            fp16_enabled = torch.cuda.is_available()

            result = whisper_model_manager.model.transcribe(
                tmp_file_path,
                fp16=fp16_enabled,
                verbose=False,  # Reduce verbose output
            )

            # Ensure result is a dictionary and contains 'text'
            if not isinstance(result, dict) or "text" not in result:
                raise ValueError(
                    "Whisper transcription did not return expected format."
                )

            transcript = result["text"].strip()

            logger.info(
                "Transcription complete for %s. Transcript length: %d characters",
                audio_file.filename,
                len(transcript),
            )

            # Return additional metadata if needed
            response_data = {
                "transcript": transcript,
                "language": result.get("language", "unknown"),
                "segments": len(result.get("segments", [])),
                "filename": audio_file.filename,
            }

            return JSONResponse(response_data)

        finally:
            try:
                Path(tmp_file_path).unlink(missing_ok=True)
                logger.info("Temporary file deleted: %s", tmp_file_path)
            except OSError:
                logger.warning("Failed to delete temporary file: %s", tmp_file_path)

    except Exception as e:
        logger.exception(
            "Error during transcription for %s: %s", audio_file.filename, e
        )
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e


@app.get("/models")
async def get_available_models():
    """Returns list of available Whisper models."""
    try:
        models = whisper.available_models()
        return JSONResponse({"available_models": list(models)})
    except Exception as e:
        logger.exception("Error getting available models: %s", e)
        error_message = (
            "Failed to retrieve available models. Ensure the Whisper library is installed "
            "and the model cache directory is accessible."
        )
        raise HTTPException(status_code=500, detail=error_message) from e


@app.get("/model-info")
async def get_model_info():
    """Returns information about the currently loaded model."""
    if not whisper_model_manager.model:
        raise HTTPException(status_code=503, detail="No model loaded.")

    try:
        model_name = os.getenv("WHISPER_MODEL_NAME", "base")
        device = next(whisper_model_manager.model.parameters()).device

        return JSONResponse(
            {
                "model_name": model_name,
                "device": str(device),
                "is_cuda_available": torch.cuda.is_available(),
                "model_loaded": True,
            }
        )
    except Exception as e:
        logger.exception("Error getting model info: %s", e)
        error_message = (
            "Failed to retrieve model information. Ensure the Whisper model is loaded "
            "and accessible."
        )
        raise HTTPException(status_code=500, detail=error_message) from e
