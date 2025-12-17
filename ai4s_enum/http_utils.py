import random
import time
from typing import Iterable, Optional

from .logger import get_logger

logger = get_logger()


def safe_request(
    method: str,
    url: str,
    *,
    headers=None,
    params=None,
    json_data=None,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_base: float = 1.5,
    retry_statuses: Iterable[int] = (429, 500, 502, 503, 504),
) -> Optional[object]:
    """通用带重试的 HTTP 请求。成功(<400)返回 Response，否则返回 None。"""
    # 为了兼容 Cursor 沙箱：仅在真正发请求时再 import requests（否则可能触发系统 SSL 证书读取权限问题）
    import requests  # noqa: WPS433

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_data,
                timeout=timeout,
            )

            if resp.status_code < 400:
                return resp

            if resp.status_code in set(retry_statuses):
                raise requests.HTTPError(f"Retryable HTTP {resp.status_code}")

            resp.raise_for_status()

        except Exception as e:
            if attempt == max_retries:
                logger.warning(f"请求失败 ({max_retries} 次重试后) | url={url[:80]}, error={e}")
                return None
            sleep_time = (backoff_base**attempt) + random.uniform(0, 20)
            logger.warning(f"请求错误，即将重试 | attempt={attempt}/{max_retries}, retry_in={sleep_time:.1f}s, error={e}")
            time.sleep(sleep_time)

    return None


