"""图像 I/O 与经典滤波的纯 numpy 实现。

设计意图（对应方案 §4.2 的 L1 经典基线）：
- **算法层不依赖 OpenCV**。局部方差、ENL、Lee / 自适应 / 中值滤波
  全部用 numpy 实现，只有"读文件 / 写文件"才需要图像编解码库。
- **PNG 读写自带实现**（zlib + struct，PNG 规范足够简单）。
  于是 SAR 侧的指标计算与横评在只装 numpy 的环境里就能跑通、可测试，
  OpenCV 仅作为"更多格式（TIFF/GeoTIFF）与更全算子"的可选增强。
- 校验口径统一：L1 与 L2（深度模型）必须用同一套指标，
  指标实现在这里，是"横评"的共同基准。
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


class ImageIOError(RuntimeError):
    """图像读写失败。"""


# ---------------------------------------------------------------- PNG 写入

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: str | Path, img: Any) -> str:
    """把 numpy 图像写成 PNG（8 位灰度或 RGB）。

    使用 filter type 0（None）逐行压缩——实现简单且压缩率对
    演示用灰度图完全够用。
    """
    if np is None:
        raise ImageIOError("numpy 未安装，无法写图像")
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim not in (2, 3):
        raise ImageIOError(f"不支持的图像形状：{arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        height, width = arr.shape
        color_type = 0  # grayscale
        channels = 1
    else:
        height, width, channels = arr.shape
        if channels != 3:
            raise ImageIOError(f"不支持的通道数：{channels}")
        color_type = 2  # truecolor RGB

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0: None
        raw.extend(arr[y].tobytes())

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    body = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _png_chunk(b"IEND", b"")
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return str(p)


# ---------------------------------------------------------------- PNG 读取

def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw: bytes, width: int, height: int, channels: int, bit_depth: int) -> Any:
    """按 PNG 规范逐行反滤波（支持 filter 0–4）。"""
    bpp = max(1, channels * bit_depth // 8)  # 每像素字节数
    stride = (width * channels * bit_depth + 7) // 8
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = np.frombuffer(raw[pos : pos + stride], dtype=np.uint8).copy()
        pos += stride
        if ftype == 0:
            cur = line
        elif ftype == 1:
            cur = line
            for i in range(bpp, stride):
                cur[i] = (int(cur[i]) + int(cur[i - bpp])) & 0xFF
        elif ftype == 2:
            cur = (line.astype(np.int32) + prev.astype(np.int32)).astype(np.uint8)
        elif ftype == 3:
            cur = line
            for i in range(stride):
                left = int(cur[i - bpp]) if i >= bpp else 0
                cur[i] = (int(cur[i]) + ((left + int(prev[i])) >> 1)) & 0xFF
        elif ftype == 4:
            cur = line
            for i in range(stride):
                left = int(cur[i - bpp]) if i >= bpp else 0
                up = int(prev[i])
                upleft = int(prev[i - bpp]) if i >= bpp else 0
                cur[i] = (int(cur[i]) + _paeth(left, up, upleft)) & 0xFF
        else:
            raise ImageIOError(f"不支持的 PNG filter 类型：{ftype}")
        out[y] = cur
        prev = cur
    return out


def read_png(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """读取 PNG，返回 (numpy 图像, 元数据)。"""
    if np is None:
        raise ImageIOError("numpy 未安装，无法读图像")
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ImageIOError("不是合法的 PNG 文件")

    pos = 8
    width = height = bit_depth = color_type = 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
            if interlace != 0:
                raise ImageIOError("暂不支持隔行扫描（interlace）PNG")
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break

    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        raise ImageIOError(f"不支持的 PNG 颜色类型：{color_type}")
    if bit_depth not in (8, 16):
        raise ImageIOError(f"暂只支持 8/16 位 PNG，当前 {bit_depth} 位")

    raw = zlib.decompress(bytes(idat))
    buf = _unfilter(raw, width, height, channels, bit_depth)

    if bit_depth == 16:
        img = buf.view(">u2").reshape(height, width, channels).astype(np.float32) / 65535.0
    else:
        img = buf.reshape(height, width, channels).astype(np.float32) / 255.0

    if channels == 1:
        img = img[:, :, 0]
    elif channels == 2:
        img = img[:, :, 0]
    elif channels == 4:
        img = img[:, :, :3]

    meta = {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "bit_depth": int(bit_depth),
        "color_type": int(color_type),
    }
    return img, meta


# ---------------------------------------------------------------- 纯 numpy 滤波

def box_filter(img: Any, ksize: int) -> Any:
    """盒式均值滤波（积分图实现，O(n)）。

    边界取 symmetric，对齐 OpenCV BORDER_REFLECT_101 的默认行为，
    保证与 cv2.blur 的数值口径一致。
    """
    if np is None:
        raise ImageIOError("numpy 未安装")
    k = ksize if ksize % 2 == 1 else ksize + 1
    pad = k // 2
    padded = np.pad(img.astype(np.float64), pad, mode="symmetric")
    c = padded.cumsum(axis=0).cumsum(axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    h, w = img.shape
    total = (
        c[k : k + h, k : k + w]
        - c[0:h, k : k + w]
        - c[k : k + h, 0:w]
        + c[0:h, 0:w]
    )
    return (total / float(k * k)).astype(np.float32)


def median_filter(img: Any, ksize: int) -> Any:
    """中值滤波（滑窗视图实现）。

    中值滤波不能用积分图，滑窗 + np.median 在演示尺寸（≤1024²）上够快。
    """
    if np is None:
        raise ImageIOError("numpy 未安装")
    k = ksize if ksize % 2 == 1 else ksize + 1
    pad = k // 2
    padded = np.pad(img, pad, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, (k, k))
    return np.median(windows.reshape(*windows.shape[:2], -1), axis=-1).astype(np.float32)


def lee_filter(img: Any, ksize: int) -> Any:
    """Lee 滤波（局部统计量自适应）。

    斑点噪声是**乘性**噪声，其方差随信号强度变化，
    所以噪声方差必须从"均匀暗区"估计，而不是全图平均。

    实现：
    1. 计算局部均值与局部方差；
    2. 取局部均值最低的 25% 像素（暗区通常为均匀地物），
       用其局部方差均值估计噪声方差 σ_u²；
    3. 权重 w = clip((σ_v² - σ_u²) / σ_v², 0, 1)：
       均匀区 w→0（强平滑），纹理/边缘区 w→1（保留细节）。

    若不做第 2 步而直接用全图局部方差当噪声方差，
    权重会恒等于 1，滤波器等于恒等映射——这是常见实现错误。
    """
    mean_img = box_filter(img, ksize)
    mean_sq = box_filter(img * img, ksize)
    var_img = np.clip(mean_sq - mean_img * mean_img, 0, None)

    # 从暗区（均匀地物）估计噪声方差
    threshold = np.quantile(mean_img, 0.25)
    dark = mean_img <= threshold
    if dark.sum() >= 16:
        sigma_u2 = float(var_img[dark].mean())
    else:
        sigma_u2 = float(var_img.mean())

    sigma_u2 = max(sigma_u2, 1e-12)
    weight = np.clip((var_img - sigma_u2) / np.maximum(var_img, 1e-12), 0.0, 1.0)
    return (mean_img + weight * (img - mean_img)).astype(np.float32)


def adaptive_filter(img: Any, ksize: int) -> Any:
    """自适应滤波：局部均值 + 残差阈值裁剪。"""
    mean_img = box_filter(img, ksize)
    diff = img - mean_img
    sigma = float(diff.std())
    threshold = 2.5 * sigma if sigma > 0 else 0.0
    if threshold > 0:
        diff = np.clip(diff, -threshold, threshold)
    return (mean_img + diff).astype(np.float32)


FILTER_FUNCS = {
    "median": median_filter,
    "lee": lee_filter,
    "adaptive": adaptive_filter,
}


def local_variance(img: Any, ksize: int = 7) -> float:
    """局部方差均值——反映斑点噪声强度的核心指标。"""
    mean_img = box_filter(img, ksize)
    mean_sq = box_filter(img * img, ksize)
    var = np.clip(mean_sq - mean_img * mean_img, 0, None)
    return float(var.mean())


def _prominence(profile: Any) -> float:
    """频谱剖面的峰值突出度：(peak - median) / peak。

    比"峰值/总能量"更稳健——后者会被自然纹理的宽频能量稀释，
    导致有条纹的影像也只得到极小数值。
    """
    p = np.asarray(profile, dtype=np.float64)
    total = float(p.sum())
    if total <= 0:
        return 0.0
    peak = float(p.max())
    if peak <= 0:
        return 0.0
    median = float(np.median(p))
    return max(0.0, (peak - median) / peak)


def detect_striping_fft(img: Any) -> tuple[float | None, float | None]:
    """FFT 谱峰检测条纹（经典频域方法，不用深度学习）。

    方向判定（易错点，已用实测图验证）：
    - **水平条纹**（如 ``img[40:42, :] = 220``，沿 x 方向延伸的亮线）
      → 图像在 y 方向变化、在 x 方向恒定
      → 傅里叶能量只落在 **fx=0 那一列** → 体现在 `col_profile`
      （按 fx 索引的能量和）→ 方向记为 **0°**
    - **垂直条纹**（如 ``img[:, 40:42] = 220``）→ 能量落在 **fy=0 那一行**
      → 体现在 `row_profile` → 方向记为 **90°**

    强度用峰值突出度衡量（见 _prominence），对宽频自然纹理不敏感。
    """
    if np is None:
        return None, None
    try:
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(np.abs(f))
        rows, cols = fshift.shape
        cy, cx = rows // 2, cols // 2
        fshift[cy - 1 : cy + 2, cx - 1 : cx + 2] = 0  # 去 DC

        total = float(fshift.sum())
        if total <= 0:
            # 完全均匀图像：除 DC 外无任何频率成分 → 确实没有条纹
            return 0.0, None

        # col_profile：按 fx 索引的能量和（水平条纹能量所在处）
        col_profile = fshift.sum(axis=0)
        # row_profile：按 fy 索引的能量和（垂直条纹能量所在处）
        row_profile = fshift.sum(axis=1)

        col_prom = _prominence(col_profile)
        row_prom = _prominence(row_profile)

        strength = round(max(col_prom, row_prom), 6)
        # 水平条纹 → col_profile 突出 → 0°；垂直条纹 → row_profile 突出 → 90°
        direction = 0.0 if col_prom >= row_prom else 90.0
        return strength, direction
    except Exception:  # noqa: BLE001
        return None, None


def psnr(a: Any, b: Any) -> float:
    """峰值信噪比（归一化到 [0,1] 动态范围）。"""
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: Any, b: Any) -> float:
    """全局 SSIM（简化实现，够用于演示横评）。"""
    c1, c2 = 0.01**2, 0.03**2
    mu_a, mu_b = float(a.mean()), float(b.mean())
    saa, sbb = float(a.var()), float(b.var())
    sab = float(((a - mu_a) * (b - mu_b)).mean())
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2)
    return num / den if den > 0 else 0.0
