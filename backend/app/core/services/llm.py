import json
import re

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

    def _extract_json(self, text: str) -> str:
        """
        Extracts the first JSON object string from a given text,
        handling cases where the LLM might include markdown code blocks or
        explanatory text.
        """
        # Regex to find content within a JSON markdown block
        match_code_block = re.search(r"```json\n(.*?)```", text, re.DOTALL)
        if match_code_block:
            return match_code_block.group(1).strip()

        # Fallback: Try to find the first JSON object from the start
        # This is less robust but might catch cases where LLM omits markdown
        match_json_start = re.search(r"\{.*\}", text, re.DOTALL)
        if match_json_start:
            return match_json_start.group(0).strip()

        logger.warning(
            f"Could not find valid JSON string in LLM output: {text[:500]}..."
        )
        return ""  # Return empty string if no JSON found

    async def process_symptoms_and_summarize(self, transcript: str) -> SymptomSummary:
        """
        Analyzes the transcribed text to extract symptoms and generate a summary using LLM.
        ... (method remains the same) ...
        """
        logger.info("LLMService: Starting symptom processing and summarization.")

        prompt = f"""You are an AI assistant designed to extract key medical symptoms and provide a concise summary from a patient-provider conversation.
        Identify the main symptoms and their associated details.
        Summarize the overall narrative of the conversation in a clear and objective manner.
        Additionally, identify the patient's **chief complaint** (the primary reason for their visit in a concise phrase) and provide a **history of the present illness** (a chronological narrative detailing the symptoms, their onset, progression, and any associated factors).

        Conversation Transcript:
        "{transcript}"

        Please provide the output in a JSON format with the following keys:
        {{
            "symptoms": [
                {{"name": "string", "details": "string"}},
                // ... more symptoms if applicable
            ],
            "chief_complaint": "string", // Example: "Headache and nausea"
            "history_of_present_illness": "string", // Example: "Patient reports sudden onset of severe headache 2 days ago, accompanied by nausea and light sensitivity. Pain is throbbing, 8/10, not relieved by OTC medication."
            "summary_narrative": "string" // Overall summary of the conversation.
        }}
        Ensure the output is valid JSON and all requested fields are present, even if empty or concise.
        """

        try:
            raw_llm_output = await self.llm_client.generate_completion(
                prompt,
                model=self.llm_model_name,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            logger.debug(
                f"LLMService: Raw LLM output for symptom summary: {raw_llm_output[:1000]}..."
            )

            json_output = self._extract_json(raw_llm_output)
            logger.debug(
                f"LLMService: Extracted JSON from LLM output: {json_output[:500]}..."
            )

            if not json_output:
                logger.error(
                    f"LLM output did not contain a recognizable JSON structure. Raw output: {raw_llm_output[:1000]}..."
                )
                raise ValueError(
                    "LLM output did not contain a recognizable JSON structure."
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
        except ValueError as e:
            logger.error(
                f"LLMService: {e}. Raw LLM output: {raw_llm_output[:500]}...",
                exc_info=True,
            )
            error_message = f"LLM failed to provide valid JSON for symptom summary: {e}"
            raise RuntimeError(error_message) from e
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
            raw_llm_output = await self.llm_client.generate_completion(
                prompt,
                model=self.llm_model_name,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            logger.debug(
                f"LLMService: Raw LLM output for SOAP Note: {raw_llm_output[:1000]}..."
            )

            json_output = self._extract_json(raw_llm_output)
            logger.debug(
                f"LLMService: Extracted JSON from LLM output: {json_output[:500]}..."
            )

            if not json_output:
                logger.error(
                    f"LLM output did not contain a recognizable JSON structure. Raw output: {raw_llm_output[:1000]}..."
                )
                raise ValueError(
                    "LLM output did not contain a recognizable JSON structure."
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
        except ValueError as e:
            logger.error(
                f"LLMService: {e}. Raw LLM output: {raw_llm_output[:500]}...",
                exc_info=True,
            )
            error_message = f"LLM failed to provide valid JSON for SOAP note: {e}"
            raise RuntimeError(error_message) from e
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
