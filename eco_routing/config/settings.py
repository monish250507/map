from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Radius (in meters) around the source within which the OSM road graph is built.
    # Increased to 20km so that typical city trips include both source and destination.
    osm_download_radius_m: int = 20000
    # When both source and destination are known, we build a bbox around them;
    # cap the span to avoid downloading excessively large graphs.
    max_bbox_km: int = 120
    bbox_margin_km: int = 20
    graph_cache_size: int = 4
    data_dir: Path = Path(__file__).resolve().parent.parent / "data_files"
    api_key_hash_algorithm: str = "sha256"

    model_config = {"env_prefix": "ECOROUTING_"}


settings = Settings()
