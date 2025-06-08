import logging
import mimetypes
from pathlib import Path

from app.core.models import SOAPNote, SymptomSummary
from app.core.services import LLMService, STTService
from app.utils.config import settings
from app.utils.logging import logger


class VoiceProcessorService:
    """
    Service responsible for orchestrating the entire AshaVoice pipeline:
    Audio Validation -> STT -> LLM Summarization -> LLM SOAP Note Generation -> Result Saving.
    This class encapsulates the end-to-end business workflow.
    """

    def __init__(self, stt_service: STTService, llm_service: LLMService):
        self.stt_service = stt_service
        self.llm_service = llm_service
        logger.info("VoiceProcessor initialized.")

    def _log(self, level: int, message: str):
        """
        Custom logging method to ensure consistent logging format.
        """
        logger.log(level, f"{self.__class__.__name__}: {message}")

    async def validate_audio_file(self, audio_file_path: Path) -> Path:
        """
        Validates the audio file path and type.
        """
        audio_file_path = audio_file_path.resolve()
        if not audio_file_path.is_file():
            error_message = f"Audio file not found: {audio_file_path}"
            self._log(logging.ERROR, error_message)
            raise FileNotFoundError(error_message)

        mime_type, _ = mimetypes.guess_type(audio_file_path)
        if mime_type is None or not mime_type.startswith("audio/"):
            error_message = (
                f"Invalid audio file type: {mime_type} for file {audio_file_path}"
            )
            self._log(logging.ERROR, error_message)
            raise ValueError(error_message)

        return audio_file_path

    async def run_full_pipeline(self, audio_file_path_str: str) -> dict:
        """
        Executes the full AshaVoice pipeline for a given audio file.

        Args:
            audio_file_path_str (str): Path to the audio file.

        Returns:
            dict: A dictionary containing the transcript, summary, and SOAP note.

        Raises:
            FileNotFoundError: If the audio file is not found.
            ValueError: If input is invalid or transcription is empty.
            RuntimeError: For failures in STT or LLM processing.
            Exception: For any unexpected errors.
        """
        self._log(logging.INFO, f"Starting full pipeline for: {audio_file_path_str}")

        try:
            audio_file_path = await self.validate_audio_file(Path(audio_file_path_str))
            self._log(logging.INFO, f"Validated audio file: {audio_file_path}")
            transcript = await self.stt_service.transcribe_audio_file(audio_file_path)
            self._log(logging.INFO, f"Transcription completed for: {audio_file_path}")
            self._log(logging.INFO, f"Transcript length: {len(transcript)} characters")

            self._log(
                logging.INFO, "Step 2: Processing symptoms and generating summary..."
            )
            summary: SymptomSummary = (
                await self.llm_service.process_symptoms_and_summarize(transcript)
            )

            self._log(logging.INFO, "Symptom summary generated successfully.")
            self._log(
                logging.INFO,
                f"Summary length: {len(summary.model_dump_json(indent=2))} characters",
            )

            self._log(
                logging.INFO,
                "Step 3: Generating SOAP Note from summary...",
            )
            soap_note: SOAPNote = await self.llm_service.generate_soap_note(summary)
            self._log(logging.INFO, "SOAP Note generated successfully.")
            self._log(
                logging.INFO,
                f"SOAP Note length: {len(soap_note.model_dump_json(indent=2))} characters",
            )

            # Save results locally
            await self._save_results(audio_file_path, transcript, summary, soap_note)

            self._log(logging.INFO, "Results saved successfully.")

            return {
                "transcript": transcript,
                "summary": summary.model_dump(),
                "soap_note": soap_note.model_dump(),
            }

        except FileNotFoundError:
            self._log(
                logging.ERROR,
                f"Audio file not found: {audio_file_path_str}",
            )
            raise
        except ValueError:
            self._log(
                logging.ERROR,
                f"Invalid input or transcription failed for: {audio_file_path_str}",
            )
            raise
        except RuntimeError:
            self._log(
                logging.ERROR,
                f"Runtime error during STT or LLM processing for: {audio_file_path_str}",
            )
            raise
        except Exception as e:
            self._log(
                logging.CRITICAL,
                f"An unexpected error occurred during pipeline execution: {e}",
            )
            error_message = f"An unexpected error occurred: {e}"
            raise RuntimeError(error_message) from e

    async def _save_results(
        self,
        audio_file_path: Path,
        transcript: str,
        summary: SymptomSummary,
        soap_note: SOAPNote,
    ):
        """
        Saves the pipeline results to the output directory.
        """
        try:
            output_base_path = Path(settings.OUTPUT_DIR)
            output_base_path.mkdir(parents=True, exist_ok=True)

            base_name = audio_file_path.stem
            summary_output_path = output_base_path.joinpath(f"{base_name}_summary.json")
            soap_output_path = output_base_path.joinpath(f"{base_name}_soap.json")
            transcript_output_path = output_base_path.joinpath(
                f"{base_name}_transcript.txt"
            )

            with summary_output_path.open("w", encoding="utf-8") as f:
                f.write(summary.model_dump_json(indent=2))
            with soap_output_path.open("w", encoding="utf-8") as f:
                f.write(soap_note.model_dump_json(indent=2))
            with transcript_output_path.open("w", encoding="utf-8") as f:
                f.write(transcript)

            self._log(
                logging.INFO,
                f"Files saved: {summary_output_path}, {soap_output_path}, {transcript_output_path}",
            )

        except Exception as e:
            self._log(
                logging.CRITICAL,
                f"Error saving results: {e}",
            )
            error_message = f"Failed to save results: {e}"
            raise RuntimeError(error_message) from e
