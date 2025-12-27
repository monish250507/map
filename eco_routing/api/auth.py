from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, Depends, Header, HTTPException, status

from eco_routing.config.settings import settings


router = APIRouter()

_API_KEYS_FILE = Path(__file__).resolve().parent.parent / "config" / "api_keys.txt"
_api_key_hashes: Dict[str, str] = {}


def _load_keys() -> None:
    if _API_KEYS_FILE.exists():
        with _API_KEYS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    _api_key_hashes[line] = line


def _save_key_hash(hash_value: str) -> None:
    _API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _API_KEYS_FILE.open("a", encoding="utf-8") as f:
        f.write(hash_value + "\n")
    _api_key_hashes[hash_value] = hash_value


_load_keys()


def _hash_key(raw_key: str) -> str:
    algo = settings.api_key_hash_algorithm.lower()
    h = hashlib.new(algo)
    h.update(raw_key.encode("utf-8"))
    return h.hexdigest()


def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    key_hash = _hash_key(x_api_key)
    if key_hash not in _api_key_hashes:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return x_api_key


@router.post("/generate-key")
def generate_key() -> Dict[str, str]:
    raw_key = os.urandom(32).hex()
    key_hash = _hash_key(raw_key)
    _save_key_hash(key_hash)
    return {"api_key": raw_key}


