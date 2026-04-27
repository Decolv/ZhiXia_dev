"""网络连通性检测工具"""

import logging
import socket
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# 默认检测配置
_DEFAULT_CHECK_HOST = ("223.5.5.5", 53)  # 阿里云DNS，国内访问快
_DEFAULT_CHECK_TIMEOUT = 3  # 秒
_DEFAULT_CACHE_TTL = 30  # 缓存有效期（秒）


class NetworkStatusCache:
    """网络状态缓存，避免频繁检测"""

    def __init__(self, ttl_seconds: float = _DEFAULT_CACHE_TTL):
        self._ttl = ttl_seconds
        self._last_check_time: float = 0
        self._last_status: bool = False

    def get_status(self, force_check: bool = False) -> bool:
        """获取网络状态，如果缓存过期则重新检测"""
        now = time.time()
        if force_check or (now - self._last_check_time) > self._ttl:
            self._last_status = check_internet_connectivity()
            self._last_check_time = now
            logger.debug("网络状态检测: %s", "在线" if self._last_status else "离线")
        return self._last_status

    def invalidate(self) -> None:
        """使缓存失效，下次调用将重新检测"""
        self._last_check_time = 0


# 全局缓存实例
_network_cache: Optional[NetworkStatusCache] = None


def get_network_cache(ttl_seconds: float = _DEFAULT_CACHE_TTL) -> NetworkStatusCache:
    """获取全局网络状态缓存实例"""
    global _network_cache
    if _network_cache is None:
        _network_cache = NetworkStatusCache(ttl_seconds)
    return _network_cache


def check_internet_connectivity(
    host: Tuple[str, int] = _DEFAULT_CHECK_HOST,
    timeout: float = _DEFAULT_CHECK_TIMEOUT,
) -> bool:
    """
    检测互联网连通性

    Args:
        host: 检测目标 (host, port)，默认使用阿里云DNS
        timeout: 连接超时时间（秒）

    Returns:
        True: 网络可用
        False: 网络不可用
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex(host)
        sock.close()
        return result == 0
    except Exception as exc:
        logger.debug("网络检测异常: %s", exc)
        return False


def is_online(use_cache: bool = True, force_refresh: bool = False) -> bool:
    """
    检查是否在线（带缓存）

    Args:
        use_cache: 是否使用缓存
        force_refresh: 是否强制刷新缓存

    Returns:
        True: 在线
        False: 离线
    """
    if not use_cache:
        return check_internet_connectivity()

    cache = get_network_cache()
    return cache.get_status(force_check=force_refresh)


def invalidate_network_cache() -> None:
    """使网络状态缓存失效"""
    cache = get_network_cache()
    cache.invalidate()
