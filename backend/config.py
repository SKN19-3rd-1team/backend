"""Centralized configuration for the backend."""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)


@dataclass
class Settings:
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    data_dir: str = os.getenv('DATA_DIR', 'backend/data')
    raw_json: str = os.getenv('RAW_JSON', 'backend/data/raw/konkuk_all_건국대.json')
    vector_store_path: str = os.getenv('VECTORSTORE_PATH', 'backend/data/processed/courses.parquet')
    model_name: str = 'gpt-3.5-turbo'


def get_settings() -> Settings:
    """Return default settings."""
    return Settings()
