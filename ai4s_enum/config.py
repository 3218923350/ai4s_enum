import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EnumConfig:
    # GitHub
    github_token: Optional[str]
    github_rest_base: str = "https://api.github.com"
    github_graphql_url: str = "https://api.github.com/graphql"

    # SearchAPI (可选)
    search_key: Optional[str] = None
    search_base_url: str = "https://www.searchapi.io/"

    # Gemini / OpenAI-compatible（可选）
    gemini_api_key: Optional[str] = None
    gemini_api_base: str = "https://api.gpugeek.com/v1"
    gemini_model: str = "Vendor2/GPT-5.1"


def load_config_from_env() -> EnumConfig:
    return EnumConfig(
        github_token=os.getenv("GITHUB_TOKEN") or None,
        search_key=os.getenv("SEARCH_KEY") or None,
        search_base_url=os.getenv("SEARCH_BASE_URL") or "https://www.searchapi.io/",
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_api_base=os.getenv("GEMINI_API_BASE") or "https://api.gpugeek.com/v1",
        gemini_model=os.getenv("GEMINI_MODEL") or "Vendor2/GPT-5.1",
    )


