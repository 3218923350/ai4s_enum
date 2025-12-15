"""
使用 LLM 生成针对性的搜索查询词
"""
import json
from typing import Dict, List

from .config import EnumConfig
from .http_utils import safe_request
from .leaf_clusters import LeafCluster
from .logger import get_logger
from .units import Unit

logger = get_logger()


def llm_generate_queries(
    cfg: EnumConfig,
    *,
    cluster: LeafCluster,
    unit: Unit,
) -> Dict[str, List[str]]:
    """
    使用 LLM 理解单元语义，生成专业的搜索查询词。
    
    返回格式：
    {
      "github_queries": ["fastqc", "seqkit", "trimmomatic", ...],
      "websearch_queries": ["fastqc github", "quality control ngs github", ...],
      "known_tools": ["FastQC", "Trimmomatic", "Cutadapt", ...]
    }
    """
    if not cfg.gemini_api_key:
        return {
            "github_queries": [],
            "websearch_queries": [],
            "known_tools": [],
        }

    prompt = f"""你是 AI4S 科研工具生态专家。

**任务**：为以下科研单元生成 GitHub 和 Google 搜索的关键词。

**叶子簇信息**：
- ID: {cluster.leaf_cluster_id}
- 名称: {cluster.leaf_cluster_name}
- 领域: {cluster.domain}
- 典型对象: {cluster.typical_objects}
- 任务链: {cluster.task_chain}

**单元信息**：
- ID: {unit.unit_id}
- 名称: {unit.unit_name}
- 覆盖工具: {unit.coverage_tools}

**要求**：
1. **github_queries**（8-15个）：用于 GitHub Search 的关键词
   - 纯英文技术术语（如 "fastq", "quality control", "sequence alignment"）
   - 宽泛但与该单元强相关
   - 包含常见工具类型（如 "trimmer", "aligner", "caller"）
   
2. **websearch_queries**（3-6个）：用于 Google 搜索的精准查询
   - 必须包含 "github" 关键词定向
   - 组合具体技术栈（如 "fastqc quality control github"）
   - 针对已知工具或明确场景
   
3. **known_tools**（5-10个）：该领域已知的代表性工具名
   - 直接写工具名（如 "FastQC", "Trimmomatic", "Cutadapt"）
   - 优先知名度高、使用广泛的

**严格输出 JSON（不要额外文字）**：
{{
  "github_queries": [...],
  "websearch_queries": [...],
  "known_tools": [...]
}}
"""

    resp = safe_request(
        "POST",
        f"{cfg.gemini_api_base}/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg.gemini_api_key}",
            "Content-Type": "application/json",
        },
        json_data={
            "model": cfg.gemini_model,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
        max_retries=2,
    )

    if resp is None:
        logger.warning("LLM 查询词生成失败 | 回退到空查询列表")
        return {
            "github_queries": [],
            "websearch_queries": [],
            "known_tools": [],
        }

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        # 尝试提取 JSON
        content = content.strip()
        if content.startswith("```"):
            # 去除 markdown 代码块
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        data = json.loads(content)
        return {
            "github_queries": data.get("github_queries", []),
            "websearch_queries": data.get("websearch_queries", []),
            "known_tools": data.get("known_tools", []),
        }
    except Exception as e:
        logger.warning(f"LLM 查询词响应解析失败 | error={e}")
        return {
            "github_queries": [],
            "websearch_queries": [],
            "known_tools": [],
        }

