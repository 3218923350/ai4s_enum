import csv
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Unit:
    unit_id: str
    unit_name: str
    target_scale: str
    coverage_tools: str

    def to_dict(self) -> dict:
        return asdict(self)


_PREFIX_RE = re.compile(r"^([A-Za-z]+)")


def units_filename_for_leaf_cluster_id(leaf_cluster_id: str) -> str:
    """
    规则：取叶子簇ID的字母前缀（例如 B1->B, AI5->AI），映射到 <prefix>_units.csv
    """
    m = _PREFIX_RE.match(leaf_cluster_id.strip())
    if not m:
        raise ValueError(f"无法从叶子簇ID解析字母前缀: {leaf_cluster_id}")
    prefix = m.group(1)
    return f"{prefix}_units.csv"


def load_units(units_csv_path: str) -> List[Unit]:
    units: List[Unit] = []
    with open(units_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = (row.get("单元ID") or "").strip()
            if not uid:
                continue
            units.append(
                Unit(
                    unit_id=uid,
                    unit_name=(row.get("单元名称") or "").strip(),
                    target_scale=(row.get("目标规模") or "").strip(),
                    coverage_tools=(row.get("主要覆盖工具") or "").strip().strip('"'),
                )
            )
    return units


def filter_units_for_leaf_cluster(units: List[Unit], leaf_cluster_id: str) -> List[Unit]:
    prefix = leaf_cluster_id.strip() + "-"
    return [u for u in units if u.unit_id.startswith(prefix)]


def resolve_units_for_leaf_cluster(
    leaf_cluster_id: str,
    *,
    units_dir: str,
) -> Tuple[str, List[Unit]]:
    """
    给定 leaf_cluster_id（如 B1），在 units_dir 中找到对应的 *_units.csv，
    并返回 (units_csv_path, 该簇下的 units)。
    """
    fname = units_filename_for_leaf_cluster_id(leaf_cluster_id)
    units_csv_path = os.path.join(units_dir, fname)
    if not os.path.exists(units_csv_path):
        raise FileNotFoundError(f"未找到 units 文件: {units_csv_path}")
    all_units = load_units(units_csv_path)
    return units_csv_path, filter_units_for_leaf_cluster(all_units, leaf_cluster_id)


def parse_target_scale_upper_bound(target_scale: str) -> Optional[int]:
    """
    解析类似 '200–450' / '150-350' / '100–250'，返回上界数字；解析失败返回 None。
    """
    if not target_scale:
        return None
    s = target_scale.replace("—", "-").replace("–", "-").strip()
    nums = re.findall(r"\d+", s)
    if not nums:
        return None
    if len(nums) == 1:
        return int(nums[0])
    return int(nums[-1])


