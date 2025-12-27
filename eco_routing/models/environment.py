from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvironmentContext:
    weather_risk_index: Optional[float] = None
    traffic_factor: Optional[float] = None