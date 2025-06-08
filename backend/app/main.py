from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.api.v1.routes import router as v1_router
from app.core.clients.llm import OllamaClient
from app.core.clients.stt import WhisperClient
from app.core.services import LLMService, STTService, VoiceProcessorService
from app.manager import manager
from app.utils.config import settings
from app.utils.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Handles FastAPI application startup and shutdown events.
    Initializes and cleans up global resources (clients and services).
    """
    logger.info("Application startup: Initializing services...")
    try:
        # Initialize LLM Client
        if settings.LLM_PROVIDER.lower() == "ollama":
            manager.set_llm_client(OllamaClient())
            logger.info("Using Ollama as LLM provider.")
        else:
            error_message = f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}"
            raise ValueError(error_message)

        if not await manager.get_llm_client().test_connection():
            error_message = f"LLM client ({settings.LLM_PROVIDER}) failed to connect."
            raise RuntimeError(error_message)

        manager.set_whisper_client(WhisperClient())

        # Initialize services
        manager.set_stt_service(
            STTService(whisper_client=manager.get_whisper_client())
        )  # Corrected call
        manager.set_llm_service(
            LLMService(llm_client=manager.get_llm_client())
        )  # Corrected call
        manager.set_voice_processor_service(
            VoiceProcessorService(  # Corrected call
                stt_service=manager.get_stt_service(),
                llm_service=manager.get_llm_service(),
            )
        )
        logger.info("All services and clients initialized successfully.")

    except Exception as e:
        logger.critical(
            f"Failed to initialize critical services during startup: {e}", exc_info=True
        )
        raise RuntimeError(
            "Application failed to start due to service initialization issues."
        ) from e

    yield  # Application is ready to serve requests

    logger.info("Application shutdown: Closing clients and cleaning up...")

    # Cleanup resources
    await manager.cleanup()


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AshaVoice Backend API for medical conversation processing (STT, LLM Summary, SOAP Note)",
    lifespan=lifespan,
)

app.dependency_overrides[LLMService] = manager.get_llm_service
app.dependency_overrides[STTService] = manager.get_stt_service
app.dependency_overrides[VoiceProcessorService] = manager.get_voice_processor_service

app.include_router(v1_router, prefix="/api/v1", tags=["v1"])


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# --- Health Check ---
@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Checks the status of the backend API and its core services."""
    status = {"api_status": "healthy"}
    try:
        # Use the manager's getters to access the initialized services
        llm_client = manager.get_llm_client()
        whisper_client = manager.get_whisper_client()
        stt_service = manager.get_stt_service()
        llm_service = manager.get_llm_service()
        voice_processor_service = manager.get_voice_processor_service()

        if llm_client:
            status["llm_client_initialized"] = True
            status["llm_provider"] = settings.LLM_PROVIDER
            status["llm_service_connection"] = await llm_client.test_connection()
        else:
            status["llm_client_initialized"] = False

        if whisper_client:
            status["whisper_client_initialized"] = True
            status["whisper_service_connection"] = (
                True  # Placeholder for now, refine if WhisperClient has a health check
            )
        else:
            status["whisper_client_initialized"] = False

        status["services_initialized"] = all(
            [stt_service, llm_service, voice_processor_service]
        )

        if not status["llm_service_connection"] or not status["services_initialized"]:
            raise HTTPException(status_code=503, detail="Core service(s) not ready.")

    except HTTPException:
        raise  # Re-raise FastAPI HTTP exceptions directly
    except Exception as e:
        status["message"] = f"Health check failed: {e}"
        logger.error(f"Health check encountered an error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal health check error."
        ) from e

    return status


if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server for {settings.APP_NAME}")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level=settings.LOG_LEVEL.lower())
