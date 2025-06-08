import asyncio
from http import HTTPStatus
from pathlib import Path

import aiofiles
import httpx

from app.utils.config import settings
from app.utils.logging import logger


class WhisperClient:
    """
    Client for interacting with the Whisper STT microservice.
    Encapsulates request logic, retries, and error handling.
    """

    def __init__(self):
        self.service_url = settings.WHISPER_SERVICE_URL
        self.max_retries = settings.WHISPER_MAX_RETRIES
        self.timeout = settings.WHISPER_TIMEOUT
        self.client = httpx.AsyncClient(timeout=self.timeout)
        logger.info(f"WhisperClient initialized for service URL: {self.service_url}")

    async def transcribe_audio(self, audio_file_path: Path) -> str:
        """
        Sends an audio file to the Whisper microservice for transcription.

        Args:
            audio_file_path (Path): Path to the audio file.

        Returns:
            str: The transcribed text.

        Raises:
            FileNotFoundError: If the local audio file does not exist.
            RuntimeError: If transcription fails after max retries or due to unexpected errors.
        """
        if not audio_file_path.exists():
            logger.error("Audio file not found: %s", audio_file_path)
            error_message = f"Audio file not found: {audio_file_path}"
            raise FileNotFoundError(error_message)

        transcribe_url = f"{self.service_url}/transcribe/"

        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Whisper transcription attempt {attempt + 1}/{self.max_retries} for {audio_file_path.name}"
                )

                async with aiofiles.open(audio_file_path, mode="rb") as f:
                    file_content = await f.read()

                files = {
                    "audio_file": (audio_file_path.name, file_content, "audio/mpeg")
                }
                response = await self.client.post(transcribe_url, files=files)
                response.raise_for_status()

                result = response.json()
                transcript = result.get("transcript")

                if not transcript:
                    logger.warning(
                        "Whisper service returned empty transcript or unexpected format on attempt %d: %s",
                        attempt + 1,
                        result,
                    )
                    raise ValueError(
                        "Whisper service returned an empty or invalid transcript."
                    )

                logger.info("Transcription received successfully.")
                return transcript

            except httpx.TimeoutException as e:
                logger.error(
                    f"Whisper service request timed out on attempt {attempt + 1}: {e}"
                )
            except httpx.RequestError as e:
                logger.error(
                    f"Network error with Whisper service on attempt {attempt + 1}: {e}"
                )
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error from Whisper service on attempt {attempt + 1}: {e.response.status_code} - {e.response.text}"
                )
                if e.response.status_code < HTTPStatus.INTERNAL_SERVER_ERROR:
                    break
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred with Whisper client on attempt {attempt + 1}: {e}",
                    exc_info=True,
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2**attempt)

        logger.critical(
            f"Failed to transcribe audio after {self.max_retries} attempts for {audio_file_path.name}."
        )
        error_message = f"Failed to transcribe audio after {self.max_retries} attempts."
        raise RuntimeError(error_message)

    async def list_models(self) -> list[str]:
        """
        Mocks listing available Whisper models.
        In a real scenario, if the Whisper service had an API to list models,
        you would make an HTTP call here.
        """
        logger.info("WhisperClient: Listing models (mock)...")
        # For simplicity, we'll return a hardcoded list of common Whisper models
        # that might be used by the external service.
        await asyncio.sleep(0.05)  # Simulate a small delay
        return ["base", "small", "medium", "large-v3", "distil-large-v3"]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
