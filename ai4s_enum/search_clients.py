import re
import time
from typing import Dict, Iterable, List, Optional

from .config import EnumConfig
from .http_utils import safe_request
from .logger import get_logger

logger = get_logger()


def gh_headers(cfg: EnumConfig) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if cfg.github_token:
        headers["Authorization"] = f"Bearer {cfg.github_token}"
    return headers


def extract_github_urls(text: str) -> List[str]:
    if not text:
        return []
    pat = r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
    return [f"https://github.com/{a}/{b}" for a, b in re.findall(pat, text)]


def github_search(cfg: EnumConfig, query: str, *, pages: int = 2, per_page: int = 50) -> List[dict]:
    """GitHub REST Search repositories，返回 items（原始JSON对象列表）。"""
    repos: List[dict] = []
    for page in range(1, pages + 1):
        resp = safe_request(
            "GET",
            f"{cfg.github_rest_base}/search/repositories",
            headers=gh_headers(cfg),
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            },
            timeout=90,
            max_retries=40,
        )
        if resp is None:
            continue
        try:
            items = resp.json().get("items", [])
            repos.extend(items)
        except Exception as e:
            logger.warning(f"GitHub 搜索响应解析失败 | error={e}")
        time.sleep(5)
    return repos


def github_graphql_enrich(cfg: EnumConfig, full_names: Iterable[str]) -> Dict[str, dict]:
    """
    GraphQL enrich。若未配置 github_token，则返回空 dict。
    返回 {nameWithOwner -> meta}。
    """
    if not cfg.github_token:
        return {}

    full_names = list(full_names)
    results: Dict[str, dict] = {}
    for i in range(0, len(full_names), 20):
        batch = full_names[i : i + 20]
        parts = []
        for idx, fn in enumerate(batch):
            try:
                owner, name = fn.split("/", 1)
            except ValueError:
                continue
            parts.append(
                f"""
                  r{idx}: repository(owner: "{owner}", name: "{name}") {{
                    nameWithOwner
                    description
                    primaryLanguage {{ name }}
                    licenseInfo {{ spdxId name }}
                    stargazerCount
                    isFork
                    updatedAt
                    url
                  }}
                """
            )
        if not parts:
            continue
        query = "query {" + "\n".join(parts) + "}"
        resp = safe_request(
            "POST",
            cfg.github_graphql_url,
            headers={"Authorization": f"Bearer {cfg.github_token}"},
            json_data={"query": query},
            timeout=90,
            max_retries=40,
        )
        if resp is None:
            continue
        try:
            data = resp.json().get("data", {})
            for _, v in data.items():
                if v:
                    results[v["nameWithOwner"]] = v
        except Exception as e:
            logger.warning(f"GraphQL 响应解析失败 | error={e}")
        time.sleep(0.2)
    return results


def web_search(cfg: EnumConfig, query: str, *, num: int = 10) -> List[str]:
    """
    SearchAPI（google engine）抽取 GitHub URL。若未配置 SEARCH_KEY，则返回空列表。
    """
    if not cfg.search_key:
        return []
    if num <= 0:
        return []
    
    logger.debug(f"WebSearch | query={query}, num={num}")
    resp = safe_request(
        "GET",
        f"{cfg.search_base_url}api/v1/search",
        params={"q": query, "engine": "google", "num": num, "api_key": cfg.search_key},
        timeout=90,
        max_retries=40,
    )
    if resp is None:
        return []

    urls: List[str] = []
    try:
        for item in resp.json().get("organic_results", []):
            urls += extract_github_urls(item.get("link", ""))
            urls += extract_github_urls(item.get("snippet", ""))
    except Exception as e:
        logger.warning(f"WebSearch 响应解析失败 | error={e}")

    # 去重（保持顺序）
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def github_full_name_from_url(url: str) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"github\.com/([^/]+)/([^/#?]+)", url)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}"


