from abc import ABC, abstractmethod


class base_rewriter(ABC):
    """Define base rewriter."""

    @abstractmethod
    def __init__(self, component_name=None):
        """Init required rewriter according to component name."""
        ...

    def run(self, query, temperature=1e-10):
        """Run the required rewriter"""
        ...
