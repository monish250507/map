from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DriverProfile:
    driver_id: str
    aggressiveness: Optional[float] = None
    driver_profile: str = "normal"  # Default to "normal" if inference fails