import asyncio
from typing import Any

import httpx

from app.core.clients.llm import LLMClient
from app.utils.config import settings
from app.utils.logging import logger


class OllamaClientError(Exception):
    """Custom exception for Ollama client errors."""


class OllamaClient(LLMClient):
    """
    Centralized Ollama client with connection management and model operations.
    """

    def __init__(self):
        self.host = settings.OLLAMA_HOST
        self.model_name = (
            settings.LLM_MODEL_NAME
        )  # Default model for generate_completion
        self.timeout = settings.OLLAMA_TIMEOUT
        self._client: httpx.AsyncClient | None = None
        self._connected: bool = False
        logger.info(f"OllamaClient initialized for host: {self.host}")

    async def _get_httpx_client(self) -> httpx.AsyncClient:
        """Lazily creates and returns the httpx.AsyncClient."""
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.host, timeout=self.timeout)
        return self._client

    async def test_connection(self) -> bool:
        """
        Test connection to Ollama server.
        """
        try:
            client = await self._get_httpx_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            self._connected = True
            logger.info("Successfully connected to Ollama at %s", self.host)
            return True
        except httpx.RequestError as e:
            logger.error("Failed to connect to Ollama at %s: %s", self.host, e)
            self._connected = False
            return False
        except Exception as e:
            logger.error(
                "An unexpected error occurred during Ollama connection test: %s", e
            )
            self._connected = False
            return False

    async def list_models(self) -> list[str]:
        """
        Get list of available models.
        """
        if not self._connected and not await self.test_connection():
            raise OllamaClientError("Not connected to Ollama server.")

        client = await self._get_httpx_client()
        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            models_data = response.json()
            available_models = [
                model["name"] for model in models_data.get("models", [])
            ]
            logger.debug("Available Ollama models: %s", available_models)
            return available_models
        except Exception as e:
            logger.error("Failed to list Ollama models: %s", e)
            error_message = f"Failed to list Ollama models: {e}"
            raise OllamaClientError(error_message) from e

    async def model_exists(self, model_name: str) -> bool:
        """
        Check if a specific model exists.
        """
        try:
            available_models = await self.list_models()
            return any(
                model_name in model or model.startswith(model_name)
                for model in available_models
            )
        except OllamaClientError:
            logger.error(
                "Error checking if model '%s' exists. Ensure Ollama server is running.",
                model_name,
            )
            return False

    async def pull_model(self, model_name: str, stream: bool = True) -> bool:
        """
        Pull a model from Ollama registry.
        """
        if not self._connected and not await self.test_connection():
            raise OllamaClientError("Not connected to Ollama server.")

        client = await self._get_httpx_client()
        logger.info("Starting pull for Ollama model '%s'...", model_name)

        payload = {"name": model_name, "stream": stream}
        try:
            if stream:
                async with client.stream("POST", "/api/pull", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            logger.info("Ollama pull progress: %s", line)
                logger.info(
                    "Ollama model '%s' pulled successfully (streaming).", model_name
                )
                await asyncio.sleep(1)  # Give Ollama a moment to register
                return await self.model_exists(model_name)
            logger.info("Pulling Ollama model '%s' without streaming...", model_name)
            return await self._pull_without_stream(model_name)

        except Exception as e:
            logger.error("Failed to pull Ollama model '%s': %s", model_name, e)
            return False

    async def _pull_without_stream(self, model_name: str) -> bool:
        """Internal method to pull model without streaming."""
        client = await self._get_httpx_client()
        try:
            response = await client.post(
                "/api/pull", json={"name": model_name, "stream": False}
            )
            response.raise_for_status()
            logger.info(
                "Ollama model '%s' pulled successfully (non-streaming).", model_name
            )
            await asyncio.sleep(1)  # Give Ollama a moment to register
            return await self.model_exists(model_name)
        except Exception as e:
            logger.error("Non-streaming Ollama pull failed for '%s': %s", model_name, e)
            return False

    async def generate_completion(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text completion from the Ollama model.
        """
        if not self._connected and not await self.test_connection():
            raise OllamaClientError("Not connected to Ollama server.")
        if not await self.model_exists(self.model_name):
            error_message = f"Default model '{self.model_name}' not available."
            raise OllamaClientError(error_message)

        client = await self._get_httpx_client()
        payload = {
            "model": kwargs.pop(
                "model", self.model_name
            ),  # Allow overriding default model
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.pop("temperature", settings.LLM_TEMPERATURE),
                "top_p": kwargs.pop("top_p", settings.LLM_TOP_P),
                **kwargs,  # Pass remaining kwargs to options if needed or directly to payload
            },
        }

        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            response_data = response.json()
            if "response" in response_data:
                logger.debug("Ollama completion generated successfully.")
                return response_data["response"]
            logger.warning("Ollama response missing 'response' key: %s", response_data)
            raise ValueError("Invalid response format from Ollama service.")
        except Exception as e:
            logger.error(f"Ollama generate completion failed: {e}", exc_info=True)
            error_message = f"Ollama generation failed: {e}"
            raise OllamaClientError(error_message) from e

    async def chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Chat with a model.
        """
        if not self._connected and not await self.test_connection():
            raise OllamaClientError("Not connected to Ollama server.")
        if not await self.model_exists(self.model_name):
            error_message = f"Default model '{self.model_name}' not available."
            raise OllamaClientError(error_message)

        client = await self._get_httpx_client()
        payload = {
            "model": kwargs.pop("model", self.model_name),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.pop("temperature", settings.LLM_TEMPERATURE),
                "top_p": kwargs.pop("top_p", settings.LLM_TOP_P),
                **kwargs,
            },
        }

        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            response_data = response.json()
            logger.debug("Ollama chat response received.")
            return response_data
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}", exc_info=True)
            error_message = f"Ollama chat failed: {e}"
            raise OllamaClientError(error_message) from e

    async def delete_model(self, model_name: str) -> bool:
        """
        Delete a model.
        """
        if not self._connected and not await self.test_connection():
            raise OllamaClientError("Not connected to Ollama server.")

        client = await self._get_httpx_client()
        try:
            response = await client.delete("/api/delete", json={"name": model_name})
            response.raise_for_status()
            logger.info("Ollama model '%s' deleted successfully.", model_name)
            return True
        except Exception as e:
            logger.error(
                f"Failed to delete Ollama model '%s': {e}", model_name, exc_info=True
            )
            return False

    async def __aenter__(self):
        await self._get_httpx_client()  # Ensure client is created for context entry
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            logger.info("OllamaClient httpx.AsyncClient closed.")
