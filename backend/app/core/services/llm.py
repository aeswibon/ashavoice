import json

from app.core.clients.llm.base import LLMClient
from app.core.models import SOAPNote, SymptomSummary
from app.utils.config import settings
from app.utils.logging import logger


class LLMService:
    """
    Service responsible for LLM-based text processing,
    including symptom summarization and SOAP note generation, and model management.
    Utilizes an LLMClient (e.g., OllamaClient, OpenAIClient) to interact with the LLM.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.llm_model_name = settings.LLM_MODEL_NAME
        self.temperature = settings.LLM_TEMPERATURE
        self.top_p = settings.LLM_TOP_P
        logger.info("LLMService initialized with client: %s", type(llm_client).__name__)

    async def process_symptoms_and_summarize(self, transcript: str) -> SymptomSummary:
        """
        Analyzes the transcribed text to extract symptoms and generate a summary using LLM.
        ... (method remains the same) ...
        """
        logger.info("LLMService: Starting symptom processing and summarization.")

        prompt = f"""You are an AI assistant designed to extract key medical symptoms and provide a concise summary from a patient-provider conversation.
        Identify the main symptoms and their associated details.
        Summarize the overall narrative of the conversation in a clear and objective manner.

        Conversation Transcript:
        "{transcript}"

        Please provide the output in a JSON format with the following keys:
        {{
            "symptoms": [
                {{"name": "string", "details": "string"}},
                // ... more symptoms
            ],
            "summary_narrative": "string"
        }}
        Ensure the output is valid JSON.
        """

        try:
            json_output = await self.llm_client.generate_completion(
                prompt,
                model=self.llm_model_name,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            summary_data = json.loads(json_output)
            summary = SymptomSummary(**summary_data)
            logger.info("LLMService: Symptom summary generated successfully.")
            return summary
        except json.JSONDecodeError as e:
            logger.error(
                f"LLMService: LLM output was not valid JSON for symptom summary: {json_output[:500]}... Error: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                "LLM failed to provide valid JSON for symptom summary."
            ) from e
        except Exception as e:
            logger.error(
                f"LLMService: Error processing symptoms with LLM: {e}", exc_info=True
            )
            error_message = f"Failed to summarize symptoms: {e}"
            raise RuntimeError(error_message) from e

    async def generate_soap_note(self, summary: SymptomSummary) -> SOAPNote:
        """
        Generates a SOAP (Subjective, Objective, Assessment, Plan) note based on the symptom summary.
        ... (method remains the same) ...
        """
        logger.info("LLMService: Starting SOAP Note generation.")

        summary_json_str = summary.model_dump_json(indent=2)

        prompt = f"""You are an AI assistant specialized in generating concise and accurate SOAP notes from patient symptom summaries.
        Strictly adhere to the SOAP format (Subjective, Objective, Assessment, Plan).
        Use the provided symptom summary to populate the relevant sections.

        Symptom Summary:
        {summary_json_str}

        Provide the output in a JSON format with the following keys:
        {{
            "subjective": "string",
            "objective": "string",
            "assessment": "string",
            "plan": "string"
        }}
        Ensure the output is valid JSON.
        """

        try:
            json_output = await self.llm_client.generate_completion(
                prompt,
                model=self.llm_model_name,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            soap_data = json.loads(json_output)
            soap_note = SOAPNote(**soap_data)
            logger.info("LLMService: SOAP Note generated successfully.")
            return soap_note
        except json.JSONDecodeError as e:
            logger.error(
                f"LLMService: LLM output was not valid JSON for SOAP note: {json_output[:500]}... Error: {e}",
                exc_info=True,
            )
            raise RuntimeError("LLM failed to provide valid JSON for SOAP note.") from e
        except Exception as e:
            logger.error(
                f"LLMService: Error generating SOAP Note with LLM: {e}", exc_info=True
            )
            error_message = f"Failed to generate SOAP Note: {e}"
            raise RuntimeError(error_message) from e

    async def list_models(self) -> list[str]:
        """
        Lists available LLM models from the configured LLM client.
        """
        logger.info("LLMService: Listing available LLM models.")
        try:
            models = await self.llm_client.list_models()
            logger.info(f"LLMService: Found {len(models)} LLM models.")
            return models
        except Exception as e:
            logger.error(f"LLMService: Failed to list LLM models: {e}", exc_info=True)
            error_message = f"Failed to retrieve LLM models: {e}"
            raise RuntimeError(error_message) from e

    async def pull_model(self, model_name: str) -> bool:
        """
        Pulls a specified LLM model using the configured LLM client.
        """
        logger.info(f"LLMService: Attempting to pull LLM model: {model_name}")
        try:
            success = await self.llm_client.pull_model(model_name)
            if success:
                logger.info(f"LLMService: Successfully pulled model: {model_name}")
            else:
                logger.warning(f"LLMService: Failed to pull model: {model_name}")
            return success
        except Exception as e:
            logger.error(
                f"LLMService: Error pulling model '{model_name}': {e}", exc_info=True
            )
            error_message = f"Failed to pull LLM model: {e}"
            raise RuntimeError(error_message) from e

    async def delete_model(self, model_name: str) -> bool:
        """
        Deletes a specified LLM model using the configured LLM client.
        """
        logger.info(f"LLMService: Attempting to delete LLM model: {model_name}")
        try:
            success = await self.llm_client.delete_model(model_name)
            if success:
                logger.info(f"LLMService: Successfully deleted model: {model_name}")
            else:
                logger.warning(f"LLMService: Failed to delete model: {model_name}")
            return success
        except Exception as e:
            logger.error(
                f"LLMService: Error deleting model '{model_name}': {e}", exc_info=True
            )
            error_message = f"Failed to delete LLM model: {e}"
            raise RuntimeError(error_message) from e
