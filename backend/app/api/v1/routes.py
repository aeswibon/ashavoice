import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from app import manager
from app.core.services import LLMService, STTService, VoiceProcessorService
from app.utils.config import settings
from app.utils.logging import logger

router = APIRouter()


# ----------------------------------------------------------------------
# Pydantic models for request/response validation
# ----------------------------------------------------------------------
class ModelRequest(BaseModel):
    model_name: str


# ----------------------------------------------------------------------
# Dependency functions to integrate with the app_manager
# ----------------------------------------------------------------------
# These functions will be used in Depends() to fetch the services
# managed by app_manager in main.py
def get_current_voice_processor_service() -> VoiceProcessorService:
    return manager.get_voice_processor_service()


def get_current_llm_service() -> LLMService:
    return manager.get_llm_service()


def get_current_stt_service() -> STTService:
    return manager.get_stt_service()


# ----------------------------------------------------------------------
# Module-level dependency singletons to avoid B008 warning
# ----------------------------------------------------------------------
voice_processor_dependency = Depends(get_current_voice_processor_service)
llm_service_dependency = Depends(get_current_llm_service)
stt_service_dependency = Depends(get_current_stt_service)


# ----------------------------------------------------------------------
# Audio Processing Route
# ----------------------------------------------------------------------


@router.post(
    "/process_audio",
    summary="Process audio file for transcription, summarization, and SOAP note generation",
)
async def process_audio_file(
    audio_file: UploadFile,
    voice_processor: VoiceProcessorService = voice_processor_dependency,
):
    """
    Receives an audio file, processes it through the entire AshaVoice pipeline
    (STT, LLM summarization, SOAP note generation), and returns the results.
    """

    if not audio_file:
        logger.warning("API: No audio file provided in the request.")
        raise HTTPException(
            status_code=400,
            detail="No audio file provided. Please upload an audio file.",
        )

    logger.info(f"API: Received audio file '{audio_file.filename}' for processing.")

    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        logger.warning(f"API: Invalid file type received: {audio_file.content_type}")
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an audio file."
        )

    temp_dir = Path(tempfile.gettempdir()) / "ashavoice_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_path = temp_dir / audio_file.filename

    try:
        with temp_audio_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"API: Saved uploaded audio to temporary path: {temp_audio_path}")

        # Call the appropriate method on the VoiceProcessorService
        pipeline_results = await voice_processor.run_full_pipeline(str(temp_audio_path))

        return JSONResponse(
            content={
                "message": "Audio processed successfully",
                "results": pipeline_results,
                "output_directory": str(settings.OUTPUT_DIR),
            },
            status_code=status.HTTP_200_OK,
        )

    except FileNotFoundError as e:
        logger.error(f"API Error: File not found during processing: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        logger.error(f"API Error: Invalid data during processing: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        logger.error(f"API Error: Pipeline runtime failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal service error during processing: {e}"
        ) from e
    except Exception as e:
        logger.critical(
            f"API Error: Unhandled exception during audio processing: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        ) from e
    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink()
            logger.info(f"API: Cleaned up temporary audio file: {temp_audio_path}")


@router.get(
    "/results",
    summary="List all processed results (transcripts, summaries, SOAP notes)",
)
async def list_processed_results():
    """
    Lists all files (transcripts, summaries, SOAP notes) found in the output directory.
    """
    output_files = []
    try:
        if not settings.OUTPUT_DIR.is_dir():
            logger.warning(f"API: Output directory not found: {settings.OUTPUT_DIR}")
            raise HTTPException(status_code=404, detail="Output directory not found.")

        for f in settings.OUTPUT_DIR.iterdir():
            if f.is_file():
                output_files.append({"name": f.name, "size_bytes": f.stat().st_size})
        logger.info(f"API: Listed {len(output_files)} processed files.")
        return JSONResponse(
            content={"files": output_files}, status_code=status.HTTP_200_OK
        )
    except PermissionError as e:
        logger.error(f"API Error: Permission denied accessing output directory: {e}")
        raise HTTPException(
            status_code=500, detail="Permission denied to access output directory."
        ) from e
    except Exception as e:
        logger.critical(
            f"API Error: Failed to list processed results: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while listing results: {e}",
        ) from e


@router.get(
    "/results/{filename}",
    summary="Retrieve content of a specific processed result file",
)
async def get_processed_result(filename: str):
    """
    Retrieves the content of a specific processed file by its filename.
    """
    file_path = settings.OUTPUT_DIR / filename

    try:
        resolved_file_path = file_path.resolve(strict=True)
        if not resolved_file_path.is_relative_to(settings.OUTPUT_DIR):
            logger.warning(f"API: Attempted directory traversal: {filename}")
            raise HTTPException(status_code=400, detail="Invalid file path.")
    except FileNotFoundError as e:
        logger.warning(f"API: Requested file not found: {filename}")
        raise HTTPException(status_code=404, detail="File not found.") from e
    except Exception as e:
        logger.error(
            f"API Error: Error resolving file path for {filename}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail="Error resolving file path.") from e

    try:
        if resolved_file_path.suffix == ".json":
            with resolved_file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
            return JSONResponse(content=content, status_code=status.HTTP_200_OK)
        if resolved_file_path.suffix == ".txt":
            with resolved_file_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return PlainTextResponse(content=content, status_code=status.HTTP_200_OK)
        logger.warning(f"API: Unsupported file type requested: {filename}")
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    except PermissionError as e:
        logger.error(f"API Error: Permission denied accessing file {filename}: {e}")
        raise HTTPException(
            status_code=500, detail="Permission denied to access file."
        ) from e
    except Exception as e:
        logger.critical(
            f"API Error: Failed to retrieve file {filename}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while retrieving file: {e}",
        ) from e


@router.get(
    "/models/llm",
    summary="List available LLM models",
)
async def list_llm_models(llm_service: LLMService = llm_service_dependency):
    """
    Lists all available LLM models from the configured LLM provider (e.g., Ollama, OpenAI).
    """
    try:
        models = await llm_service.list_models()
        return JSONResponse(
            content={"llm_models": models}, status_code=status.HTTP_200_OK
        )
    except RuntimeError as e:
        logger.error(f"API Error: Failed to list LLM models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve LLM models: {e}"
        ) from e
    except Exception as e:
        logger.critical(
            f"API Error: Unhandled exception listing LLM models: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        ) from e


@router.post(
    "/models/llm/pull",
    summary="Pull a specific LLM model (Ollama-specific functionality)",
)
async def pull_llm_model(
    model_request: dict[str, str], llm_service: LLMService = llm_service_dependency
):
    """
    Initiates pulling a specified LLM model from the configured provider.
    Note: This functionality is primarily relevant for local LLM providers like Ollama.
    """
    name = model_request.get("model_name")
    if not name:
        raise HTTPException(
            status_code=400, detail="'model_name' is required in the request body."
        )

    try:
        if not hasattr(llm_service.llm_client, "pull_model") or not callable(
            llm_service.llm_client.pull_model
        ):
            raise HTTPException(
                status_code=405,
                detail=f"Model pulling not supported by the current LLM provider ({type(llm_service.llm_client).__name__}).",
            )

        success = await llm_service.pull_model(name)
        if success:
            return JSONResponse(
                content={
                    "message": f"Model '{name}' pull initiated/completed successfully."
                },
                status_code=status.HTTP_200_OK,
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to pull model '{name}'. Check logs for details.",
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"API Error: Failed to pull LLM model {name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to pull LLM model: {e}"
        ) from e
    except Exception as e:
        logger.critical(
            f"API Error: Unhandled exception pulling LLM model {name}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        ) from e


@router.delete(
    "/models/llm/delete",
    summary="Delete a specific LLM model (Ollama-specific functionality)",
)
async def delete_llm_model(
    model_request: dict[str, str], llm_service: LLMService = llm_service_dependency
):
    """
    Deletes a specified LLM model from the configured provider.
    Note: This functionality is primarily relevant for local LLM providers like Ollama.
    """
    name = model_request.get("model_name")
    if not name:
        raise HTTPException(
            status_code=400, detail="'model_name' is required in the request body."
        )

    try:
        if not hasattr(llm_service.llm_client, "delete_model") or not callable(
            llm_service.llm_client.delete_model
        ):
            raise HTTPException(
                status_code=405,
                detail=f"Model deletion not supported by the current LLM provider ({type(llm_service.llm_client).__name__}).",
            )

        success = await llm_service.delete_model(name)
        if success:
            return JSONResponse(
                content={"message": f"Model '{name}' deleted successfully."},
                status_code=status.HTTP_200_OK,
            )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model '{name}'. Check logs for details.",
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"API Error: Failed to delete LLM model {name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete LLM model: {e}"
        ) from e
    except Exception as e:
        logger.critical(
            f"API Error: Unhandled exception deleting LLM model {name}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        ) from e


@router.get(
    "/models/stt",
    summary="List available STT models",
)
async def list_stt_models(stt_service: STTService = stt_service_dependency):
    """
    Lists all available STT models from the configured STT provider (e.g., Whisper).
    """
    try:
        models = await stt_service.list_models()
        return JSONResponse(
            content={"stt_models": models}, status_code=status.HTTP_200_OK
        )
    except RuntimeError as e:
        logger.error(f"API Error: Failed to list STT models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve STT models: {e}"
        ) from e
    except Exception as e:
        logger.critical(
            f"API Error: Unhandled exception listing STT models: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        ) from e
