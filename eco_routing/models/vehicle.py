from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VehicleProfile:
    vehicle_id: str
    fuel_efficiency_l_per_100km: Optional[float] = None
    vehicle_type: Optional[str] = None


