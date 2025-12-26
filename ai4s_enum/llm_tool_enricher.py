"""
使用 LLM 将候选 repo 转换为完整的工具定义
"""
import json
import time
from typing import Dict, List, Optional

from .config import EnumConfig
from .http_utils import safe_request
from .leaf_clusters import LeafCluster
from .logger import get_logger
from .units import Unit

logger = get_logger()


def llm_enrich_tools(
    cfg: EnumConfig,
    *,
    cluster: LeafCluster,
    unit: Unit,
    candidates: List[dict],
    batch_size: int = 10,
) -> List[dict]:
    """
    使用 LLM 将候选 repo 批量转换为完整的工具定义。
    
    输入 candidates 格式（来自 GraphQL enrich）：
    {
      "full_name": "...",
      "url": "...",
      "description": "...",
      "language": "...",
      "stars": 123,
      "license": "...",
      ...
    }
    
    输出 tools 格式：
    {
      "name": "...",
      "one_line_profile": "...",
      "detailed_description": "...",
      "domains": ["B1", "B1-01"],
      "subtask_category": [...],
      "application_level": "solver",
      "primary_language": "Python",
      "repo_url": "...",
      "help_website": [...],
      "license": "...",
      "tags": [...]
    }
    """
    if not cfg.gemini_api_key:
        logger.warning("LLM enrichment 未启用 | 原因=缺少 GEMINI_API_KEY")
        return []
    
    logger.info(f"开始 LLM 批量enrichment | 候选总数={len(candidates)}, batch_size={batch_size}")
    tools: List[dict] = []
    
    # 分批处理，避免单次 prompt 过长
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(candidates) + batch_size - 1) // batch_size
        logger.info(f"处理批次 {batch_num}/{total_batches} | 候选={i+1}-{min(i+batch_size, len(candidates))}/{len(candidates)}")
        
        batch_result = _enrich_batch(cfg, cluster=cluster, unit=unit, batch=batch)
        tools.extend(batch_result)
    
    return tools


def _enrich_batch(
    cfg: EnumConfig,
    *,
    cluster: LeafCluster,
    unit: Unit,
    batch: List[dict],
    max_parse_retries: int = 3,
) -> List[dict]:
    """处理单批候选，支持 JSON 解析失败时的重试"""
    
    # 构建候选列表的简要信息
    candidates_info = []
    for idx, cand in enumerate(batch):
        candidates_info.append({
            "idx": idx,
            "full_name": cand.get("full_name"),
            "url": cand.get("url"),
            "description": cand.get("description"),
            "language": cand.get("language"),
            "stars": cand.get("stars"),
            "license": cand.get("license"),
        })
    
    prompt = f"""你是 AI4S（AI for Science）科研工具生态专家。

**任务**：严格判断以下候选是否是「科研工具」，并生成完整定义。

**叶子簇上下文**：
- 簇ID: {cluster.leaf_cluster_id} - {cluster.leaf_cluster_name}
- 领域: {cluster.domain}
- 典型对象: {cluster.typical_objects}
- 单元ID: {unit.unit_id} - {unit.unit_name}
- 覆盖范围: {unit.coverage_tools}

**候选列表**：
```json
{json.dumps(candidates_info, ensure_ascii=False, indent=2)}
```

---

## ⚠️ 三条硬过滤规则（不满足任一条 → is_tool=false）

### 🧱 规则1：科学任务可映射性（最重要）
工具必须能直接用于以下至少一种科学任务：
- ✅ 科学数据生成（simulation/synthesis）
- ✅ 科学数据处理（alignment/filtering/QC/normalization）
- ✅ 科学数据分析（inference/estimation/statistics）
- ✅ 科学建模（physics/chemistry/biology/materials）
- ✅ 科学推断（structure prediction/dynamics/interaction）
- ✅ 科学可视化（专用于科学数据）

**典型反例（直接排除）**：
- ❌ 通用编程库（numpy/pandas 除外）
- ❌ Web框架（FastAPI/Flask等）
- ❌ 前端组件（React/Vue等）
- ❌ DevOps/CI/CD工具

### 🧱 规则2：排除"软件工程工具"
如果工具的主要价值是以下之一，直接排除：
- ❌ Web API 框架/REST 服务
- ❌ App UI/前端组件
- ❌ 多媒体编辑/播放器
- ❌ 安全扫描/SAST/测试框架
- ❌ DevOps/部署/监控

### 🧱 规则3：排除"非工具型仓库"
- ❌ Papers list / Awesome list / Curated resources
- ❌ 教程/笔记/课程作业
- ❌ 纯文档/README 型仓库
- ❌ 论文复现但无工具化接口
- ❌ 个人学习项目（star < 20 且无科学机构背书）

---

## ✅ 合格的科研工具示例（对标）
- FastQC, Trimmomatic, Cutadapt（生信QC）
- BWA, STAR, Bowtie（序列比对）
- GATK, Picard（变异检测）
- STalign, ClipKIT, PhyKIT（进化/系统发育）
- AlphaFold, RoseTTAFold（结构预测）

---

## 输出格式

**严格输出 JSON 对象（必须包含 "results" 字段，值为数组，不要额外文字）**：
```json
{{
  "results": [
    {{
      "idx": 0,
      "is_tool": true,
      "name": "FastQC",
      "one_line_profile": "Quality control tool for high throughput sequence data",
      "detailed_description": "...",
      "domains": ["{cluster.leaf_cluster_id}", "{unit.unit_id}"],
      "subtask_category": ["quality_control", "qc_report"],
      "application_level": "solver",
      "primary_language": "Java",
      "repo_url": "...",
      "help_website": [...],
      "license": "GPL-3.0",
      "tags": ["fastq", "quality-control", "ngs"]
    }},
    {{
      "idx": 1,
      "is_tool": false,
      "reason": "通用 Web 框架，不符合规则1（无科学任务映射）"
    }},
    {{
      "idx": 2,
      "is_tool": false,
      "reason": "Papers list，不符合规则3（非工具型仓库）"
    }}
  ]
}}
```

**字段说明**：
- application_level: 只能是 library/solver/workflow/platform/dataset/service
- subtask_category: 必须是科学任务相关（如 quality_control, alignment, variant_calling）
- 所有描述用英文

**关键**：
1. 必须输出有效的 JSON 对象，包含 "results" 字段
2. "results" 必须是数组，每个候选都必须出现（通过 idx 对应）
3. is_tool=false 必须给出明确 reason（引用三条规则）
4. 严格执行三条硬规则，**宁可漏掉边缘工具，也不要混入噪声**
"""

    # 直接使用与 llm_query_generator.py 相同的 HTTP 请求方式
    for parse_attempt in range(1, max_parse_retries + 1):
        # logger.info("prompt",prompt)
        resp = safe_request(
            "POST",
            f"{cfg.gemini_api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {cfg.gemini_api_key}",
                "Content-Type": "application/json",
            },
            json_data={
                "model": cfg.gemini_model,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=600,
            max_retries=40,
        )
        # logger.info("resp",resp.json())
        if resp is None:
            if parse_attempt == max_parse_retries:
                logger.warning(f"LLM enrichment HTTP 请求失败 | 批次大小={len(batch)}, 已重试 {max_parse_retries} 次")
                return []
            logger.warning(f"LLM enrichment HTTP 请求失败，重试中 | attempt={parse_attempt}/{max_parse_retries}")
            time.sleep(2 ** parse_attempt)
            continue

        try:
            content = resp.json()["choices"][0]["message"]["content"]
            # 尝试提取 JSON
            content = content.strip()
            if content.startswith("```"):
                # 去除 markdown 代码块
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
            
            # 解析 JSON
            parsed = json.loads(content)
            
            # 如果直接是数组，使用它
            if isinstance(parsed, list):
                results = parsed
            # 如果是对象，优先查找 "results" 字段
            elif isinstance(parsed, dict):
                if "results" in parsed and isinstance(parsed["results"], list):
                    results = parsed["results"]
                else:
                    # 查找第一个数组类型的值
                    results = None
                    for key, value in parsed.items():
                        if isinstance(value, list):
                            results = value
                            break
                    if results is None:
                        logger.warning(
                            f"LLM 返回 JSON 对象但无数组字段 | attempt={parse_attempt}/{max_parse_retries}, "
                            f"keys={list(parsed.keys())}, content_preview={content[:200]}"
                        )
                        if parse_attempt < max_parse_retries:
                            time.sleep(2 ** parse_attempt)
                            continue
                        return []
            else:
                logger.warning(
                    f"LLM 返回内容不是数组或对象 | attempt={parse_attempt}/{max_parse_retries}, "
                    f"type={type(parsed)}, content_preview={content[:200]}"
                )
                if parse_attempt < max_parse_retries:
                    time.sleep(2 ** parse_attempt)
                    continue
                return []
            
            # 提取 is_tool=true 的工具
            tools: List[dict] = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                if item.get("is_tool"):
                    tool = {k: v for k, v in item.items() if k != "idx" and k != "is_tool"}
                    tools.append(tool)
            
            logger.debug(f"批次解析成功 | 识别工具数={len(tools)}/{len(batch)}")
            return tools
            
        except Exception as e:
            logger.warning(f"LLM enrichment 响应解析失败 | attempt={parse_attempt}/{max_parse_retries}, error={e}")
            if parse_attempt < max_parse_retries:
                time.sleep(2 ** parse_attempt)
                continue
            return []
    
    return []

