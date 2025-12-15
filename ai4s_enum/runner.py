import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Set

from tqdm import tqdm

from .config import EnumConfig
from .logger import get_logger

logger = get_logger()
from .leaf_clusters import LeafCluster
from .llm_filter import gemini_analyze
from .llm_query_generator import llm_generate_queries
from .llm_tool_enricher import llm_enrich_tools
from .query_builder import (
    build_expansion_queries,
    build_github_queries,
    build_websearch_queries,
)
from .search_clients import (
    github_full_name_from_url,
    github_graphql_enrich,
    github_search,
    web_search,
)
from .units import Unit, parse_target_scale_upper_bound


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _heuristic_is_tool(candidate: dict) -> bool:
    """
    在没有 LLM 时做一个非常保守的启发式判断：
    - 不是 fork
    - 有描述
    - star 不太低（>= 5）
    """
    if candidate.get("is_fork") is True:
        return False
    if not (candidate.get("description") or "").strip():
        return False
    stars = candidate.get("stars")
    try:
        if stars is not None and int(stars) < 5:
            return False
    except Exception:
        pass
    return True


def _candidate_from_search_item(item: dict) -> dict:
    return {
        "full_name": item.get("full_name"),
        "url": item.get("html_url") or item.get("url"),
        "description": item.get("description"),
        "language": item.get("language"),
        "stars": item.get("stargazers_count"),
        "is_fork": item.get("fork"),
        "updated_at": item.get("updated_at"),
    }


def _candidate_from_graphql(meta: dict) -> dict:
    return {
        "full_name": meta.get("nameWithOwner"),
        "url": meta.get("url"),
        "description": meta.get("description"),
        "language": (meta.get("primaryLanguage") or {}).get("name"),
        "stars": meta.get("stargazerCount"),
        "is_fork": meta.get("isFork"),
        "updated_at": meta.get("updatedAt"),
        "license": (meta.get("licenseInfo") or {}).get("spdxId") or (meta.get("licenseInfo") or {}).get("name"),
    }


def collect_candidates_for_unit(
    cfg: EnumConfig,
    *,
    cluster: LeafCluster,
    unit: Unit,
    pages: int,
    per_page: int,
    web_num: int,
    max_rounds: int,
    seed_take: int,
    converge_delta: int,
    use_llm_queries: bool = False,
) -> dict:
    """
    复刻并泛化你原先的 run_unit：
    - 第一轮：base queries
    - 后续：用已有候选 repo 名反向扩展模板直到达到 target 或收敛
    """
    logger.info(f"开始收集候选 | 单元={unit.unit_id} ({unit.unit_name})")
    
    target = parse_target_scale_upper_bound(unit.target_scale) or 300
    target = max(50, min(target, 800))
    logger.info(f"目标候选数: {target}")

    candidates: Set[str] = set()
    searched_github_queries: Set[str] = set()
    searched_web_queries: Set[str] = set()
    all_queries: List[str] = []

    def run_github_queries(queries: Sequence[str]) -> None:
        """GitHub Search（宽泛但技术相关的英文关键词）"""
        for q in queries:
            q = q.strip()
            if not q or q in searched_github_queries:
                continue
            searched_github_queries.add(q)
            all_queries.append(f"[GH] {q}")
            
            logger.debug(f"GitHub Search | query={q}")
            # GitHub search
            for item in github_search(cfg, q, pages=pages, per_page=per_page):
                fn = item.get("full_name")
                if fn:
                    candidates.add(fn)

    def run_web_queries(queries: Sequence[str]) -> None:
        """WebSearch（精准定向，必须带 github 关键词）"""
        for q in queries:
            q = q.strip()
            if not q or q in searched_web_queries:
                continue
            searched_web_queries.add(q)
            all_queries.append(f"[WEB] {q}")
            
            logger.debug(f"WebSearch | query={q}")
            # WebSearch -> github urls
            for url in web_search(cfg, q, num=web_num):
                fn = github_full_name_from_url(url)
                if fn:
                    candidates.add(fn)

    # 第一轮：根据配置选择 LLM 生成或规则生成
    if use_llm_queries:
        logger.info(f"使用 LLM 生成查询词 | 单元={unit.unit_id}")
        llm_result = llm_generate_queries(cfg, cluster=cluster, unit=unit)
        logger.info(f"LLM 生成查询词结果: {llm_result}")
        # LLM 生成的查询
        github_queries = llm_result.get("github_queries", [])
        web_queries = llm_result.get("websearch_queries", [])
        known_tools = llm_result.get("known_tools", [])
        
        # 已知工具直接作为 GitHub 查询（最精准）
        for tool in known_tools:
            github_queries.insert(0, tool)
        
        logger.info(f"LLM 生成查询词 | GitHub={len(github_queries)}, WebSearch={len(web_queries)}, 已知工具={len(known_tools)}")
    else:
        # 规则生成的查询
        logger.info(f"使用规则生成查询词 | 单元={unit.unit_id}")
        github_queries = build_github_queries(cluster, unit)
        web_queries = build_websearch_queries(cluster, unit)
        logger.info(f"GitHub queries: {github_queries}")
        logger.info(f"WebSearch queries: {web_queries}")
        logger.info(f"规则生成查询词 | GitHub={len(github_queries)}, WebSearch={len(web_queries)}")
    
    logger.info(f"开始第 1 轮搜索 | GitHub queries={len(github_queries)}, WebSearch queries={len(web_queries)}")
    run_github_queries(github_queries)

    run_web_queries(web_queries)
    logger.info(f"第 1 轮完成 | 候选数={len(candidates)}")

    # 反向扩展：用已有工具名生成生态查询（只用 GitHub Search，更高效）
    round_idx = 2
    while len(candidates) < target and round_idx <= max_rounds:
        seed_tools = list(candidates)[:seed_take]
        expansion_queries = build_expansion_queries(seed_tools)
        prev = len(candidates)
        
        logger.info(f"开始第 {round_idx} 轮反向扩展 | 种子工具数={len(seed_tools)}, 扩展查询数={len(expansion_queries)}")
        run_github_queries(expansion_queries)  # 反向扩展只需 GitHub Search
        
        new_count = len(candidates) - prev
        logger.info(f"第 {round_idx} 轮完成 | 新增={new_count}, 总候选数={len(candidates)}")

        if new_count < converge_delta:
            logger.info(f"收敛提前终止 | 新增数({new_count}) < 阈值({converge_delta})")
            break
        round_idx += 1
    
    logger.info(f"候选收集完成 | 单元={unit.unit_id}, 最终候选数={len(candidates)}, 目标={target}")

    return {
        "target": target,
        "queries": all_queries,
        "candidate_full_names": sorted(candidates),
    }


def export_unit_json(
    cfg: EnumConfig,
    *,
    cluster: LeafCluster,
    unit: Unit,
    out_dir: str,
    pages: int,
    per_page: int,
    web_num: int,
    max_rounds: int,
    seed_take: int,
    converge_delta: int,
    use_llm: bool,
    max_llm: Optional[int],
    use_llm_queries: bool = False,
    dry_run: bool = False,
) -> str:
    logger.info(f"=" * 80)
    logger.info(f"开始处理单元 | 单元={unit.unit_id} ({unit.unit_name})")
    logger.info(f"=" * 80)
    
    os.makedirs(out_dir, exist_ok=True)

    if dry_run:
        logger.info("Dry-run 模式 | 不执行实际搜索，仅生成查询预览")
        # 不做任何联网/requests 调用，仅把"会跑哪些 query、输出到哪"写出来，方便在沙箱里验证结构
        github_queries = build_github_queries(cluster, unit)
        web_queries = build_websearch_queries(cluster, unit)
        collected = {
            "target": parse_target_scale_upper_bound(unit.target_scale) or 300,
            "queries": {
                "github": github_queries,
                "websearch": web_queries,
            },
            "candidate_full_names": [],
        }
        payload = {
            "generated_at": _now_iso(),
            "metadata": {
                "leaf_cluster": asdict(cluster),
                "unit": asdict(unit),
                "search": {
                    "target_candidates": collected["target"],
                    "queries": collected["queries"],
                    "total_candidates": 0,
                    "tool_candidates": 0,
                    "final_tools": 0,
                    "dry_run": True,
                },
            },
            "tools": [],
        }
        out_path = os.path.join(out_dir, f"{unit.unit_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Dry-run 输出完成 | 文件={out_path}")
        return out_path

    collected = collect_candidates_for_unit(
        cfg,
        cluster=cluster,
        unit=unit,
        pages=pages,
        per_page=per_page,
        web_num=web_num,
        max_rounds=max_rounds,
        seed_take=seed_take,
        converge_delta=converge_delta,
        use_llm_queries=use_llm_queries,
    )
    full_names = collected["candidate_full_names"]

    # GraphQL enrich 获取完整元数据
    logger.info(f"开始 GraphQL enrich | 候选数={len(full_names)}")
    gql = github_graphql_enrich(cfg, full_names)
    logger.info(f"GraphQL enrich 完成 | 成功={len(gql)}/{len(full_names)}")

    candidates: List[dict] = []
    for fn in full_names:
        if fn in gql:
            candidates.append(_candidate_from_graphql(gql[fn]))
        else:
            candidates.append({"full_name": fn})

    # 启发式过滤（去掉明显不是工具的：fork、无描述、star<5）
    logger.info(f"开始启发式过滤 | 候选数={len(candidates)}")
    for c in candidates:
        c["is_tool_candidate"] = _heuristic_is_tool(c)
    
    tool_candidates = [c for c in candidates if c.get("is_tool_candidate")]
    logger.info(f"启发式过滤完成 | 保留={len(tool_candidates)}/{len(candidates)}")
    
    tools: List[dict] = []
    if use_llm and cfg.gemini_api_key:
        # 使用 LLM 批量生成完整工具定义
        logger.info(f"开始 LLM 批量生成工具定义 | 候选数={len(tool_candidates)}，单批50")
        tools = llm_enrich_tools(
            cfg,
            cluster=cluster,
            unit=unit,
            candidates=tool_candidates,
            batch_size=50,
        )
        logger.info(f"LLM 生成完成 | 识别出有效工具={len(tools)}/{len(tool_candidates)}")
    else:
        # 不使用 LLM，只输出基础信息
        logger.info(f"未启用 LLM | 输出基础信息: {len(tool_candidates)} 个候选")
        tools = [
            {
                "name": (c.get("full_name") or "").split("/")[-1] if c.get("full_name") else "unknown",
                "one_line_profile": c.get("description") or "",
                "detailed_description": "",
                "domains": [cluster.leaf_cluster_id, unit.unit_id],
                "subtask_category": [],
                "application_level": "unknown",
                "primary_language": c.get("language") or "unknown",
                "repo_url": c.get("url") or (f"https://github.com/{c['full_name']}" if c.get("full_name") else ""),
                "help_website": [c.get("url")] if c.get("url") else [],
                "license": c.get("license") or "unknown",
                "tags": [],
            }
            for c in tool_candidates
        ]
    
    # 为工具添加 id（自增）
    for idx, tool in enumerate(tools, start=1):
        tool["id"] = idx
    
    logger.info(f"工具ID分配完成 | 工具数={len(tools)}")

    payload = {
        "generated_at": _now_iso(),
        "metadata": {
            "leaf_cluster": asdict(cluster),
            "unit": asdict(unit),
            "search": {
                "target_candidates": collected["target"],
                "queries": collected["queries"],
                "total_candidates": len(candidates),
                "tool_candidates": len(tool_candidates),
                "final_tools": len(tools),
            },
        },
        "tools": tools,
    }

    out_path = os.path.join(out_dir, f"{unit.unit_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(out_path) / 1024  # KB
    logger.info(f"单元处理完成 | 单元={unit.unit_id}, 工具数={len(tools)}, 文件={out_path} ({file_size:.1f} KB)")
    logger.info(f"=" * 80)
    return out_path


def export_leaf_cluster(
    cfg: EnumConfig,
    *,
    cluster: LeafCluster,
    units: Sequence[Unit],
    out_root: str,
    pages: int = 2,
    per_page: int = 50,
    web_num: int = 10,
    max_rounds: int = 6,
    seed_take: int = 20,
    converge_delta: int = 5,
    use_llm: bool = False,
    max_llm: Optional[int] = None,
    use_llm_queries: bool = False,
    dry_run: bool = False,
) -> List[str]:
    logger.info("")
    logger.info("#" * 100)
    logger.info(f"# 开始处理叶子簇 | 簇ID={cluster.leaf_cluster_id} ({cluster.leaf_cluster_name})")
    logger.info(f"# 单元数={len(units)}, 输出目录={out_root}/{cluster.leaf_cluster_id}")
    logger.info("#" * 100)
    logger.info("")
    
    out_dir = os.path.join(out_root, cluster.leaf_cluster_id)
    paths: List[str] = []
    for idx, u in enumerate(units, 1):
        logger.info(f"[{idx}/{len(units)}] 准备处理单元 | 单元={u.unit_id} ({u.unit_name})")
        paths.append(
            export_unit_json(
                cfg,
                cluster=cluster,
                unit=u,
                out_dir=out_dir,
                pages=pages,
                per_page=per_page,
                web_num=web_num,
                max_rounds=max_rounds,
                seed_take=seed_take,
                converge_delta=converge_delta,
                use_llm=use_llm,
                max_llm=max_llm,
                use_llm_queries=use_llm_queries,
                dry_run=dry_run,
            )
        )
    
    logger.info("")
    logger.info("#" * 100)
    logger.info(f"# 叶子簇处理完成 | 簇ID={cluster.leaf_cluster_id}")
    logger.info(f"# 成功生成={len(paths)} 个单元文件，输出目录={out_dir}")
    logger.info("#" * 100)
    logger.info("")
    
    return paths


