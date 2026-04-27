"""Pygame 生命周期全局管理器

多个渲染器共享同一个 pygame 实例，使用引用计数管理。
"""

import logging
import threading

logger = logging.getLogger(__name__)

class PygameManager:
    """Pygame 生命周期全局管理器。
    
    使用引用计数管理 pygame.init() 和 pygame.quit()，
    确保多个渲染器共享时不会冲突。
    """
    
    _initialized = False
    _ref_count = 0
    _lock = threading.Lock()
    
    @classmethod
    def init(cls) -> bool:
        """初始化 pygame。返回是否成功。"""
        with cls._lock:
            if cls._ref_count == 0:
                try:
                    import pygame
                    pygame.init()
                    cls._initialized = True
                    logger.debug("Pygame 已初始化")
                except ImportError:
                    logger.error("pygame 未安装")
                    return False
                except Exception as exc:
                    logger.error("Pygame 初始化失败: %s", exc)
                    return False
            cls._ref_count += 1
            return cls._initialized
    
    @classmethod
    def quit(cls) -> None:
        """减少引用计数，当为0时调用 pygame.quit()。"""
        with cls._lock:
            if cls._ref_count > 0:
                cls._ref_count -= 1
                if cls._ref_count == 0 and cls._initialized:
                    try:
                        import pygame
                        pygame.quit()
                        cls._initialized = False
                        logger.debug("Pygame 已退出")
                    except Exception:
                        pass
