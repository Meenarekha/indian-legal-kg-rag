from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Common interface for all LLM providers.

    Every provider must implement generate().
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a text response from the supplied prompt.
        """
        pass