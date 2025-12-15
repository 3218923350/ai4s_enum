import csv
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class LeafCluster:
    leaf_cluster_id: str
    leaf_cluster_name: str
    domain: str
    typical_objects: str
    task_chain: str
    tool_form: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_leaf_clusters(csv_path: str) -> Dict[str, LeafCluster]:
    """读取 leaf_clusters.csv，返回 {叶子簇ID -> LeafCluster}。"""
    clusters: Dict[str, LeafCluster] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = (row.get("叶子簇ID") or "").strip()
            if not cid:
                continue
            clusters[cid] = LeafCluster(
                leaf_cluster_id=cid,
                leaf_cluster_name=(row.get("叶子簇名称（聚合主题域）") or "").strip(),
                domain=(row.get("覆盖领域") or "").strip(),
                typical_objects=(row.get("典型对象/数据形态") or "").strip(),
                task_chain=(row.get("覆盖的任务链（聚合）") or "").strip(),
                tool_form=(row.get("工具形态侧重") or "").strip(),
            )
    return clusters


def list_leaf_cluster_ids(csv_path: str) -> List[str]:
    return sorted(load_leaf_clusters(csv_path).keys())


