import json
import re
from typing import List, Optional

from .config import EnumConfig
from .http_utils import safe_request


def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    尝试从 LLM 输出中提取第一个 JSON object。
    """
    if not text:
        return None
    text = text.strip()
    # 直接是 JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 提取第一个 {...}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def gemini_analyze(cfg: EnumConfig, *, unit_id: str, unit_name: str, repo_meta: dict) -> dict:
    """
    使用 OpenAI-compatible 的 /chat/completions 调用 Gemini（可选）。
    若未配置 GEMINI_API_KEY，则返回 {"is_tool": False, "notes": "llm_disabled"}。
    """
    if not cfg.gemini_api_key:
        return {"is_tool": False, "notes": "llm_disabled"}

    # 同 safe_request：避免在 Cursor 沙箱里仅 import 就触发系统 SSL 证书权限问题
    import requests  # noqa: WPS433

    prompt = f"""
你是 AI4S 科研工具生态构建助手。

判断下面的候选是否是“可执行科研工具”（更偏向可安装/可运行的 CLI/库/工作流组件/平台，而不是纯论文/课程/笔记/数据集）。

叶子单元：
{unit_id} - {unit_name}

候选信息：
名称: {repo_meta.get("name")}
描述: {repo_meta.get("description")}
语言: {repo_meta.get("language")}
仓库: {repo_meta.get("url")}

请严格输出 JSON（不要额外文字）：
{{
  "is_tool": true/false,
  "tool_name": "...",
  "one_line_profile": "...",
  "subtask_category": "...",
  "primary_language": "...",
  "notes": "fork / wrapper / license unclear / empty"
}}
""".strip()

    r = requests.post(
        f"{cfg.gemini_api_base.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {cfg.gemini_api_key}", "Content-Type": "application/json"},
        json={"model": cfg.gemini_model, "temperature": 0.1, "messages": [{"role": "user", "content": prompt}]},
        timeout=90,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    obj = _extract_first_json_object(content)
    return obj or {"is_tool": False, "notes": "llm_parse_failed"}


