from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from eco_routing.config.settings import settings


DATASETS = {
    "main_agent_route_manager_india": "main_agent_route_manager_india.csv",
    "fuel_predictions_india": "fuel_predictions_india.csv",
    "co2_aggregates_india": "co2_aggregates_india.csv",
    "vehicle_agent_india_full": "vehicle_agent_india_full.csv",
    "driver_agent_india_full": "driver_agent_india_full.csv",
    "environment_agent_india_full": "environment_agent_india_full.csv",
    "simulations_india": "simulations_india.csv",
    "trips_india": "trips_india.csv",
    "insights_india": "insights_india.csv",
}


class DataRepository:
    """Loads and caches required CSV datasets."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or settings.data_dir
        self._cache: Dict[str, pd.DataFrame] = {}

    def _load(self, key: str) -> pd.DataFrame:
        if key in self._cache:
            return self._cache[key]
        filename = DATASETS[key]
        path = self.base_dir / filename
        if not path.exists():
            # Load empty frame with a marker column; logic must handle empties gracefully.
            df = pd.DataFrame()
        else:
            df = pd.read_csv(path)
        self._cache[key] = df
        return df

    @property
    def vehicles(self) -> pd.DataFrame:
        return self._load("vehicle_agent_india_full")

    @property
    def drivers(self) -> pd.DataFrame:
        return self._load("driver_agent_india_full")

    @property
    def environment(self) -> pd.DataFrame:
        return self._load("environment_agent_india_full")

    @property
    def fuel_predictions(self) -> pd.DataFrame:
        return self._load("fuel_predictions_india")

    @property
    def co2_aggregates(self) -> pd.DataFrame:
        return self._load("co2_aggregates_india")

    @property
    def route_manager(self) -> pd.DataFrame:
        return self._load("main_agent_route_manager_india")

    @property
    def simulations(self) -> pd.DataFrame:
        return self._load("simulations_india")

    @property
    def trips(self) -> pd.DataFrame:
        return self._load("trips_india")

    @property
    def insights(self) -> pd.DataFrame:
        return self._load("insights_india")


data_repository = DataRepository()


