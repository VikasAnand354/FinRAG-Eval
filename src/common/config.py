import os

import yaml
from dotenv import load_dotenv

from src.common.models import RunConfig


def load_run_config(path: str) -> RunConfig:
    load_dotenv()
    with open(path) as f:
        data = yaml.safe_load(f)
    return RunConfig(**data)


def get_azure_settings() -> dict[str, str]:
    required = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    return {
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_version": os.environ["AZURE_OPENAI_API_VERSION"],
        "deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
    }
