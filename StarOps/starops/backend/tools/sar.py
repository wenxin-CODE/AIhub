"""SAR 侧领域工具。

第一阶段实现 L1 经典基线与指标计算。
L2 深度模型（GeomaticsAndRS/sar）封装见 dl_despeckle.py，第二阶段接入。

依赖分层（重要）：
- **算法层只用 numpy**。局部方差、ENL、Lee / 自适应 / 中值滤波、
  FFT 条纹检测、PSNR / SSIM 全部在 tools/imaging.py 中纯 numpy 实现。
- **I/O 层优先用自带的 PNG 编解码**（zlib + struct）。
- **OpenCV 是可选增强**：装了才支持 TIFF / GeoTIFF 等更多格式，
  没装也不影响 PNG 演示链路与全部指标计算。

这样 SAR 侧不会被重依赖卡住，且 L1 与 L2 的指标口径天然统一。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from schemas.models import ImageMetrics, ToolResult
from tools import imaging

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

# 算法层依赖：只有 numpy
_NUMPY_MISSING = np is None
# 图像编解码依赖：优先自带 PNG，cv2 缺失时只支持 PNG
_CV2_MISSING = cv2 is None

_NUMPY_HINT = (
    "图像处理依赖缺失（numpy）。执行 pip install numpy 后即可启用 SAR 侧功能。"
)
_CV2_HINT = (
    "当前仅支持 PNG 格式读写；安装 opencv-python-headless 后可读取 TIFF / GeoTIFF。"
)


def _elapsed(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


# ---------------------------------------------------------------- 图像读写

def _is_png(path: str | Path) -> bool:
    return str(path).lower().endswith(".png")


def _load_gray(path: str) -> Any:
    """读取影像为单通道 float32 灰度图，取值归一化到 [0,1]。"""
    if _NUMPY_MISSING:
        raise RuntimeError(_NUMPY_HINT)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"无法读取影像：{path}")

    # PNG 走自带实现，无需 OpenCV
    if _is_png(p) and _CV2_MISSING:
        img, _ = imaging.read_png(p)
        return img.astype(np.float32)

    if cv2 is None:
        # 只有 PNG 能读
        if _is_png(p):
            img, _ = imaging.read_png(p)
            return img.astype(np.float32)
        raise RuntimeError(_CV2_HINT)

    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取影像：{path}")
    if img.ndim == 3:
        img = img[:, :, 0] if img.shape[2] >= 3 else img[:, :, 0]
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _save_gray(path: str, img: Any) -> str:
    """写回灰度影像。PNG 用自带实现，其他格式需 OpenCV。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)

    if _is_png(out) and _CV2_MISSING:
        imaging.write_png(out, (arr * 255).astype(np.uint8))
        return str(out)

    if cv2 is None:
        raise RuntimeError(_CV2_HINT)

    cv2.imwrite(str(out), (arr * 255).astype(np.uint8))
    return str(out)


# ---------------------------------------------------------------- 元数据

def read_image_metadata(path: str) -> ToolResult:
    """读取影像元数据。"""
    start = time.perf_counter()
    p = Path(path)
    if not p.exists():
        return ToolResult.fail(f"文件不存在：{path}")
    if _NUMPY_MISSING:
        return ToolResult.fail(_NUMPY_HINT)

    try:
        if _is_png(p) and _CV2_MISSING:
            _, meta = imaging.read_png(p)
            meta["format"] = "PNG"
        elif cv2 is None:
            return ToolResult.fail(_CV2_HINT)
        else:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                return ToolResult.fail(f"无法解析影像：{path}")
            meta = {
                "width": int(img.shape[1]),
                "height": int(img.shape[0]),
                "channels": int(img.shape[2]) if img.ndim == 3 else 1,
                "dtype": str(img.dtype),
                "bit_depth": int(img.dtype.itemsize * 8),
            }
            meta["format"] = p.suffix.lstrip(".").upper()
    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(str(exc))

    meta["filename"] = p.name
    meta["size_bytes"] = p.stat().st_size
    return ToolResult.ok(data=meta, elapsed_ms=_elapsed(start))


# ---------------------------------------------------------------- 质量指标

def calculate_image_metrics(path: str) -> ToolResult:
    """计算 SAR 质量指标。

    局部方差与 CV 反映斑点强度；ENL（等效视数）为参考量；
    条纹强度走 FFT 谱峰检测（经典方法，不用深度学习）。
    """
    start = time.perf_counter()
    try:
        img = _load_gray(path)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(str(exc))

    mean = float(img.mean())
    std = float(img.std())
    cv = std / mean if mean > 1e-9 else 0.0

    local_variance = imaging.local_variance(img, ksize=7)
    enl = (mean * mean) / local_variance if local_variance > 1e-12 else None

    striping_strength, striping_direction = imaging.detect_striping_fft(img)

    metrics = ImageMetrics(
        gray_mean=round(mean, 6),
        gray_std=round(std, 6),
        local_variance=round(local_variance, 8),
        cv=round(cv, 6),
        enl=round(enl, 4) if enl is not None else None,
        noise_estimate=round(cv, 6),
        striping_strength=striping_strength,
        striping_direction=striping_direction,
    )
    return ToolResult.ok(data=metrics.model_dump(), elapsed_ms=_elapsed(start))


def detect_striping(path: str) -> ToolResult:
    """条纹检测工具。"""
    start = time.perf_counter()
    try:
        img = _load_gray(path)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(str(exc))
    strength, direction = imaging.detect_striping_fft(img)
    # 强度是 0~1 的峰值突出度，0.25 以上视为显著条纹
    has_striping = strength is not None and strength > 0.25
    return ToolResult.ok(
        data={
            "has_striping": has_striping,
            "strength": strength,
            "direction_deg": direction,
            "method": "FFT 谱峰检测（经典频域方法）",
        },
        elapsed_ms=_elapsed(start),
    )


# ---------------------------------------------------------------- L1 经典滤波

FILTER_METHODS = {
    "median": "中值滤波",
    "lee": "Lee 滤波",
    "adaptive": "自适应滤波",
}


def apply_speckle_filter(
    path: str,
    method: str = "lee",
    kernel_size: int = 5,
    output_path: str | None = None,
) -> ToolResult:
    """L1 经典滤波基线（纯 numpy 实现）。"""
    start = time.perf_counter()
    if _NUMPY_MISSING:
        return ToolResult.fail(_NUMPY_HINT)
    if method not in FILTER_METHODS:
        return ToolResult.fail(f"未知滤波方法：{method}，可选 {list(FILTER_METHODS)}")

    try:
        img = _load_gray(path)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(str(exc))

    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    out = imaging.FILTER_FUNCS[method](img, k)

    out_path = output_path or str(Path(path).with_name(f"{Path(path).stem}_{method}.png"))
    try:
        _save_gray(out_path, out)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(str(exc))

    return ToolResult.ok(
        data={
            "output_path": out_path,
            "method": method,
            "method_label": FILTER_METHODS[method],
            "filter_level": "L1_classical",
            "kernel_size": k,
        },
        elapsed_ms=_elapsed(start),
    )


# ---------------------------------------------------------------- 对比

def compare_image_quality(
    original_path: str,
    processed_path: str,
    reference_path: str | None = None,
) -> ToolResult:
    """处理前后指标对比。同一套指标口径，供多管线横评。"""
    start = time.perf_counter()
    try:
        orig = _load_gray(original_path)
        proc = _load_gray(processed_path)
    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(str(exc))

    def _stats(a: Any) -> dict[str, float]:
        m = float(a.mean())
        s = float(a.std())
        lv = imaging.local_variance(a, ksize=7)
        return {
            "gray_mean": round(m, 6),
            "gray_std": round(s, 6),
            "cv": round(s / m, 6) if m > 1e-9 else 0.0,
            "local_variance": round(lv, 8),
            "enl": round((m * m) / lv, 4) if lv > 1e-12 else 0.0,
        }

    before = _stats(orig)
    after = _stats(proc)
    delta = {k: round(after[k] - before[k], 6) for k in before}

    payload: dict[str, Any] = {
        "before": before,
        "after": after,
        "delta": delta,
        "improvement": {
            "cv_reduced_pct": round(
                (before["cv"] - after["cv"]) / before["cv"] * 100, 2
            ) if before["cv"] > 0 else 0.0,
            "enl_improved": after["enl"] > before["enl"],
        },
    }

    if reference_path and Path(reference_path).exists():
        ref = _load_gray(reference_path)
        payload["reference_metrics"] = {
            "psnr": round(imaging.psnr(proc, ref), 3),
            "ssim": round(imaging.ssim(proc, ref), 4),
        }

    return ToolResult.ok(data=payload, elapsed_ms=_elapsed(start))
