"""内存工具"""

import gc


def force_gc() -> None:
    gc.collect()
    gc.collect()


def check_memory() -> float | None:
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except FileNotFoundError:
        pass
    return None
