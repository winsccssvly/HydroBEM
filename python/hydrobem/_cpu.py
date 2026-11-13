"""The check that the processor runs AVX2, which the extension module is built for (with FMA,
``.cargo/config.toml``): imported before the module, which would otherwise stop the process at its
first instruction. The system is asked (Windows: IsProcessorFeaturePresent, which also reports
whether the system saves the AVX registers; Linux: /proc/cpuinfo); no code is run to probe the
processor. FMA is not asked separately: Windows has no feature flag for it, and every processor that
reports AVX2 also has FMA (on Linux both flags are read)."""

import os
import sys


def cpu_supported():
    """True when the processor runs AVX2 (and FMA); None when the system cannot tell."""
    if sys.platform == "win32":
        import ctypes

        try:
            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))  # PF_AVX2_INSTRUCTIONS_AVAILABLE
        except (AttributeError, OSError):
            return None
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="replace") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = line.split(":", 1)[-1].split()
                    return "avx2" in flags and "fma" in flags
    except OSError:
        return None
    return None


has_avx2 = cpu_supported  # the earlier name

if cpu_supported() is False and not os.environ.get("HYDROBEM_SKIP_CPU_CHECK"):
    raise RuntimeError(
        "this processor does not report AVX2 and FMA, which the hydrobem engine is built for; "
        "set HYDROBEM_SKIP_CPU_CHECK=1 to load it anyway (if the system reports the processor wrongly)"
    )
