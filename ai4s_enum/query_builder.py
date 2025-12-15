import re
from typing import Iterable, List, Sequence

from .leaf_clusters import LeafCluster
from .units import Unit


def _split_tokens(s: str) -> List[str]:
    """
    从中文/英文混合描述中抽取可用于搜索的 token。
    - 保留英文/数字/连字符片段
    - 也保留少量中文短语（用原始片段）
    """
    if not s:
        return []
    s = s.strip().strip('"')
    # 常见分隔符归一
    s = re.sub(r"[，,;；/|]+", " ", s)
    s = re.sub(r"[（）()]+", " ", s)
    parts = re.split(r"\s+|、", s)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 过滤太短的噪音
        if len(p) <= 1:
            continue
        out.append(p)
    # 去重保持顺序
    seen = set()
    dedup: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def domain_boost_terms(domain: str) -> List[str]:
    d = (domain or "").lower()
    if "bioinfo" in d or "genomics" in d or "evo" in d or "kg" in d:
        return ["bioinformatics", "genomics"]
    if "structbio" in d:
        return ["protein", "structure", "bioinformatics"]
    if "drug" in d:
        return ["drug discovery", "cheminformatics"]
    if "compchem" in d:
        return ["computational chemistry", "quantum chemistry"]
    if "chemistry" in d:
        return ["cheminformatics"]
    if "materials" in d:
        return ["materials science", "dft", "molecular dynamics"]
    if "earth" in d or "climate" in d or "environment" in d or "geo" in d:
        return ["geospatial", "remote sensing", "climate"]
    if "astronomy" in d:
        return ["astronomy", "fits"]
    if "neuro" in d:
        return ["neuroscience", "fmri", "eeg"]
    if "med" in d or "health" in d:
        return ["medical imaging", "dicom"]
    if "infra" in d or "hpc" in d:
        return ["hpc", "docker", "conda"]
    if "workflow" in d:
        return ["workflow", "pipeline"]
    return []


def _extract_english_keywords(text: str) -> List[str]:
    """只提取纯英文技术关键词（不允许中文字符）"""
    tokens = _split_tokens(text)
    out: List[str] = []
    for t in tokens:
        # 不允许包含中文字符
        if re.search(r"[\u4e00-\u9fff]", t):
            continue
        # 必须包含英文字母
        if not re.search(r"[a-zA-Z]", t):
            continue
        # 过滤太短的词
        if len(t) < 3:
            continue
        # 转小写并去除多余空格
        t = t.lower().strip()
        if t:
            out.append(t)
    return out


def build_github_queries(cluster: LeafCluster, unit: Unit) -> List[str]:
    """
    GitHub Search queries（相对宽泛但技术相关）：
    - 用英文技术关键词 + 领域词
    - 避免纯中文、避免过于碎片的词
    """
    boost = domain_boost_terms(cluster.domain)
    
    # 只提取英文关键词
    unit_keywords = _extract_english_keywords(unit.coverage_tools)
    cluster_keywords = _extract_english_keywords(cluster.typical_objects)
    
    queries: List[str] = []
    
    # 1. 单元覆盖工具的英文关键词（最核心）
    for kw in unit_keywords[:8]:
        queries.append(kw)
        # 组合领域词
        for b in boost[:1]:
            queries.append(f"{kw} {b}")
    
    # 2. 叶子簇典型对象（如 FASTQ/FASTA, BAM/CRAM）
    for kw in cluster_keywords[:5]:
        queries.append(kw)
        for b in boost[:1]:
            queries.append(f"{kw} {b}")
    
    # 去重
    seen = set()
    out: List[str] = []
    for q in queries:
        q = q.strip().lower()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def build_websearch_queries(cluster: LeafCluster, unit: Unit) -> List[str]:
    """
    WebSearch queries（精准、限定）：
    - 必须包含 'github' 关键词强制定向
    - 用具体技术栈组合
    - 数量少但精准
    """
    boost = domain_boost_terms(cluster.domain)
    unit_keywords = _extract_english_keywords(unit.coverage_tools)
    
    queries: List[str] = []
    
    # 核心关键词 + github 定向
    for kw in unit_keywords[:5]:
        queries.append(f"{kw} github")
        # 加上领域词增强精准度
        if boost:
            queries.append(f"{kw} {boost[0]} github")
    
    # 去重
    seen = set()
    out: List[str] = []
    for q in queries:
        q = q.strip().lower()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def build_expansion_queries(seed_repo_full_names: Sequence[str]) -> List[str]:
    """
    复用你现在 run 里的“生态扩量模板”，用已有候选 repo 名反向扩展。
    """
    ecosystem_templates = [
        "{x} snakemake",
        "{x} nextflow",
        "{x} pipeline",
        "{x} workflow",
        "{x} wrapper",
        "{x} bioconda",
        "{x} galaxy",
    ]
    queries: List[str] = []
    for fn in seed_repo_full_names:
        tool = (fn.split("/")[-1] or "").strip()
        if not tool:
            continue
        for tpl in ecosystem_templates:
            queries.append(tpl.format(x=tool))
    # 去重
    seen = set()
    out: List[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


