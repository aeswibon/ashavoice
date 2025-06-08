from pathlib import Path

from app.core.clients.stt.whisper import WhisperClient
from app.utils.logging import logger


class STTService:
    """
    Service responsible for Speech-to-Text operations.
    Utilizes WhisperClient to interact with the Whisper microservice.
    """

    def __init__(self, whisper_client: WhisperClient):
        self.whisper_client = whisper_client
        logger.info("STTService initialized.")

    async def transcribe_audio_file(self, audio_file_path: Path) -> str:
        """
        Transcribes an audio file into text.
        ... (rest of method remains the same) ...
        """
        logger.info(f"STTService: Starting transcription for {audio_file_path.name}")
        try:
            transcript = await self.whisper_client.transcribe_audio(audio_file_path)
            if not transcript.strip():
                logger.warning("STTService: Transcription returned empty text.")
                raise ValueError("Transcription resulted in empty text.")
            logger.info("STTService: Transcription complete.")
            return transcript
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"STTService: Error during transcription: {e}", exc_info=True)
            error_message = f"Failed to transcribe audio: {e}"
            raise RuntimeError(error_message) from e

    async def list_models(self) -> list[str]:
        """
        Lists available STT models from the underlying Whisper client.
        """
        logger.info("STTService: Listing available STT models.")
        try:
            models = await self.whisper_client.list_models()
            logger.info(f"STTService: Found {len(models)} STT models.")
            return models
        except Exception as e:
            logger.error(f"STTService: Failed to list STT models: {e}", exc_info=True)
            error_message = f"Failed to retrieve STT models: {e}"
            raise RuntimeError(error_message) from e
