from abc import abstractmethod


class BaseLLM:
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt using the given LLM backend."""