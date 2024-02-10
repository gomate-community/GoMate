from abc import ABC, abstractmethod


class base_reranker(ABC):
    """Define base reranker."""

    @abstractmethod
    def __init__(self, component_name=None):
        """Init required reranker according to component name."""
        ...

    def run(self, query, contexts):
        """Run the required reranker"""
        ...
