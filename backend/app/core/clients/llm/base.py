from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    """
    Abstract Base Class defining the common interface for all LLM clients.
    All concrete LLM client implementations (e.g., OllamaClient, OpenAIClient)
    must adhere to this interface.
    """

    @abstractmethod
    async def generate_completion(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates a text completion based on the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            **kwargs: Additional generation parameters specific to the LLM.

        Returns:
            str: The generated text completion.
        """

    @abstractmethod
    async def chat(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        """
        Engages in a chat conversation with the LLM.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries
                                              (e.g., [{"role": "user", "content": "Hello!"}]).
            **kwargs: Additional chat parameters specific to the LLM.

        Returns:
            Dict[str, Any]: The chat response from the LLM.
        """

    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Tests the connection to the LLM service.

        Returns:
            bool: True if connection is successful, False otherwise.
        """

    @abstractmethod
    async def list_models(self) -> list[str]:
        """
        Lists available models from the LLM service.

        Returns:
            List[str]: A list of available model names.
        """

    @abstractmethod
    async def model_exists(self, model_name: str) -> bool:
        """
        Checks if a specific model exists on the LLM service.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model exists, False otherwise.
        """

    @abstractmethod
    async def pull_model(self, model_name: str, **kwargs: Any) -> bool:
        """
        Pulls a specified model from the LLM registry.

        Args:
            model_name (str): The name of the model to pull.
            **kwargs: Additional parameters for the pull operation.

        Returns:
            bool: True if the model is successfully pulled, False otherwise.
        """

    # Using async with for httpx clients
    @abstractmethod
    async def __aenter__(self):
        """Asynchronous context manager entry."""

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Asynchronous context manager exit."""
