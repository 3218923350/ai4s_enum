"""
统一的日志配置
"""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "ai4s_enum", log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    配置并返回 logger
    
    Args:
        name: logger 名称
        log_file: 日志文件路径（可选）
        level: 日志级别
    
    Returns:
        配置好的 logger
    """
    logger = logging.getLogger(name)
    
    # 避免重复配置
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 格式化器
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler（可选）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 全局默认 logger
_default_logger = None


def get_logger() -> logging.Logger:
    """获取默认 logger"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger

