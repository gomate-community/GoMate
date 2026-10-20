"""Identity shared by search calls within one research run."""

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class SearchContext:
    """Keep anonymous search accounting scoped to a research run."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
