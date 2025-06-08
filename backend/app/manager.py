import logging

from fastapi import HTTPException

from app.core.clients.llm import LLMClient
from app.core.clients.stt import WhisperClient
from app.core.services import LLMService, STTService, VoiceProcessorService
from app.utils.logging import logger


class Manager:
    def __init__(self):
        self._llm_client: LLMClient | None = None
        self._whisper_client: WhisperClient | None = None
        self._stt_service: STTService | None = None
        self._llm_service: LLMService | None = None
        self._voice_processor_service: VoiceProcessorService | None = None

    def _log(self, level: int, message: str):
        """
        Logs messages at the specified level.
        """
        logger.log(level, message)

    def get_llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._log(
                logging.ERROR,
                message="LLMClient is not initialized. Please check the application startup.",
            )
            raise HTTPException(status_code=500, detail="LLMClient not initialized.")
        return self._llm_client

    def set_llm_client(self, client: LLMClient):
        if not isinstance(client, LLMClient):
            raise TypeError("Expected an instance of LLMClient.")
        self._llm_client = client
        self._log(logging.INFO, "LLMClient has been set successfully.")

    def get_whisper_client(self) -> WhisperClient:
        if self._whisper_client is None:
            self._log(
                logging.ERROR,
                message="WhisperClient is not initialized. Please check the application startup.",
            )
            raise HTTPException(
                status_code=500, detail="WhisperClient not initialized."
            )
        return self._whisper_client

    def set_whisper_client(self, client: WhisperClient):
        if not isinstance(client, WhisperClient):
            raise TypeError("Expected an instance of WhisperClient.")
        self._whisper_client = client
        self._log(logging.INFO, "WhisperClient has been set successfully.")

    def get_stt_service(self) -> STTService:
        if self._stt_service is None:
            self._log(
                logging.ERROR,
                message="STTService is not initialized. Please check the application startup.",
            )
            raise HTTPException(status_code=500, detail="STTService not initialized.")
        return self._stt_service

    def set_stt_service(self, service: STTService):
        if not isinstance(service, STTService):
            raise TypeError("Expected an instance of STTService.")
        self._stt_service = service
        self._log(logging.INFO, "STTService has been set successfully.")

    def get_llm_service(self) -> LLMService:
        if self._llm_service is None:
            self._log(
                logging.ERROR,
                message="LLMService is not initialized. Please check the application startup.",
            )
            raise HTTPException(status_code=500, detail="LLMService not initialized.")
        return self._llm_service

    def set_llm_service(self, service: LLMService):
        if not isinstance(service, LLMService):
            raise TypeError("Expected an instance of LLMService.")
        self._llm_service = service
        self._log(logging.INFO, "LLMService has been set successfully.")

    def get_voice_processor_service(self) -> VoiceProcessorService:
        if self._voice_processor_service is None:
            self._log(
                logging.ERROR,
                message="VoiceProcessorService is not initialized. Please check the application startup.",
            )
            raise HTTPException(
                status_code=500, detail="VoiceProcessorService not initialized."
            )
        return self._voice_processor_service

    def set_voice_processor_service(self, service: VoiceProcessorService):
        if not isinstance(service, VoiceProcessorService):
            raise TypeError("Expected an instance of VoiceProcessorService.")
        self._voice_processor_service = service
        self._log(logging.INFO, "VoiceProcessorService has been set successfully.")

    async def cleanup(self):
        """
        Cleans up all resources managed by this manager.
        This should be called during application shutdown.
        """
        self._log(logging.INFO, "Cleaning up resources...")
        if self._llm_client:
            await self._llm_client.__aexit__(
                None, None, None
            )  # Call aexit for graceful shutdown
            self._llm_client = None
        if self._whisper_client:
            await self._whisper_client.__aexit__(
                None, None, None
            )  # Call aexit for graceful shutdown
            self._whisper_client = None
        self._stt_service = None
        self._llm_service = None
        self._voice_processor_service = None
        self._log(logging.INFO, "All resources cleaned up successfully.")


manager = Manager()
