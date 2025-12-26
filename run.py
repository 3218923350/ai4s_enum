import argparse
import os

from ai4s_enum.config import load_config_from_env
from ai4s_enum.leaf_clusters import load_leaf_clusters
from ai4s_enum.logger import setup_logger
from ai4s_enum.runner import export_leaf_cluster
from ai4s_enum.units import resolve_units_for_leaf_cluster


def _parse_leaf_arg(s: str):
    if not s:
        return []
    parts = []
    for x in s.split(","):
        x = x.strip()
        if x:
            parts.append(x)
    return parts


def main():
    # 基于脚本所在目录定位默认路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    ap = argparse.ArgumentParser(
        description="AI4S 工具生态枚举：从 leaf_clusters.csv + units/*.csv 自动生成搜索任务并输出 out/<簇ID>/<单元ID>.json"
    )
    ap.add_argument("--leaf-clusters-csv", default=os.path.join(script_dir, "leaf_clusters.csv"))
    ap.add_argument("--units-dir", default=os.path.join(script_dir, "units2"))
    ap.add_argument("--out-root", default=os.path.join(script_dir, "out"))
    ap.add_argument("--leaf", default="", help="指定叶子簇ID（逗号分隔），例如 B1,B2,AI5；为空则需 --all")
    ap.add_argument("--all", action="store_true", help="对 leaf_clusters.csv 中所有叶子簇运行")

    # 搜索参数（调整为更合理的默认值）
    ap.add_argument("--pages", type=int, default=2, help="GitHub Search 每个查询翻页数（默认2）")
    ap.add_argument("--per-page", type=int, default=30, help="每页结果数（默认30）")
    ap.add_argument("--web-num", type=int, default=5, help="WebSearch 每个查询取结果数（默认5，设0禁用）")
    ap.add_argument("--max-rounds", type=int, default=3, help="反向扩展最大轮数（默认3）")
    ap.add_argument("--seed-take", type=int, default=20, help="每轮反向扩展取种子数（默认20）")
    ap.add_argument("--converge-delta", type=int, default=5, help="收敛阈值（默认5）")

    # LLM 参数（默认启用）
    ap.add_argument("--use-llm", action="store_true", default=True, help="启用 LLM 生成完整工具定义（默认启用）")
    ap.add_argument("--no-llm", dest="use_llm", action="store_false", help="禁用 LLM 工具定义生成")
    ap.add_argument("--max-llm", type=int, default=500, help="每个单元最多 LLM 处理的候选数（默认100，成本控制）")
    ap.add_argument("--llm-queries", action="store_true", default=True, help="使用 LLM 生成查询词（默认启用）")
    ap.add_argument("--no-llm-queries", dest="llm_queries", action="store_false", help="禁用 LLM 查询词生成，使用规则生成")
    
    # 其他参数
    ap.add_argument("--dry-run", action="store_true", help="不联网，仅生成查询预览")
    ap.add_argument("--log-file", default="logs/enum.log", help="日志文件路径（可选）")
    ap.add_argument("--log-level", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别（默认INFO）")

    args = ap.parse_args()

    # 设置日志
    import logging
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger(log_file=args.log_file, level=log_level)
    
    logger.info("=" * 100)
    logger.info("AI4S 工具生态枚举系统启动")
    logger.info("=" * 100)
    logger.info(f"配置 | leaf_clusters={args.leaf_clusters_csv}")
    logger.info(f"配置 | units_dir={args.units_dir}")
    logger.info(f"配置 | out_root={args.out_root}")
    logger.info(f"配置 | use_llm={args.use_llm}, llm_queries={args.llm_queries}, dry_run={args.dry_run}")
    logger.info("")

    cfg = load_config_from_env()
    clusters = load_leaf_clusters(args.leaf_clusters_csv)
    logger.info(f"加载叶子簇 | 总数={len(clusters)}")

    leaf_ids = _parse_leaf_arg(args.leaf)
    if args.all:
        leaf_ids = list(clusters.keys())
    if not leaf_ids:
        logger.error("未指定叶子簇 | 请使用 --leaf B1 或 --all")
        raise SystemExit("请使用 --leaf B1 或 --all")
    
    logger.info(f"准备处理叶子簇 | 数量={len(leaf_ids)}, 列表={leaf_ids}")
    logger.info("")

    os.makedirs(args.out_root, exist_ok=True)

    success_count = 0
    for idx, leaf_id in enumerate(leaf_ids, 1):
        logger.info(f"[{idx}/{len(leaf_ids)}] 开始处理叶子簇 | 簇ID={leaf_id}")
        
        if leaf_id not in clusters:
            logger.warning(f"叶子簇不存在 | 簇ID={leaf_id}, 跳过")
            continue
        cluster = clusters[leaf_id]
        _, units = resolve_units_for_leaf_cluster(leaf_id, units_dir=args.units_dir)
        if not units:
            logger.warning(f"未找到单元 | 簇ID={leaf_id}, 跳过")
            continue

        paths = export_leaf_cluster(
            cfg,
            cluster=cluster,
            units=units,
            out_root=args.out_root,
            pages=args.pages,
            per_page=args.per_page,
            web_num=args.web_num,
            max_rounds=args.max_rounds,
            seed_take=args.seed_take,
            converge_delta=args.converge_delta,
            use_llm=args.use_llm,
            max_llm=args.max_llm,
            use_llm_queries=args.llm_queries,
            dry_run=args.dry_run,
        )
        success_count += 1
        logger.info(f"[{idx}/{len(leaf_ids)}] 叶子簇完成 | 簇ID={leaf_id}, 文件数={len(paths)}, 输出={os.path.join(args.out_root, leaf_id)}")
        logger.info("")
    
    logger.info("=" * 100)
    logger.info(f"全部任务完成 | 成功={success_count}/{len(leaf_ids)}")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
