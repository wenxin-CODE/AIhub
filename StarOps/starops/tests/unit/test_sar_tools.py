"""SAR 侧工具测试。

依赖 numpy（算法层）。若环境中缺失则跳过——安装后自动启用。
图像文件用自带的纯 numpy PNG 编码器生成，不依赖 OpenCV。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP_DB = Path(tempfile.mkdtemp(prefix="starops_sar_")) / "test.db"
os.environ["STAROPS_DB_PATH"] = str(_TMP_DB)

tools_sar = pytest.importorskip("tools.sar", reason="tools.sar 不可用")
imaging = pytest.importorskip("tools.imaging", reason="tools.imaging 不可用")

np = pytest.importorskip("numpy", reason="numpy 未安装")


@pytest.fixture
def sample_image(tmp_path):
    """造一张带斑点噪声的测试影像（乘性斑点噪声）。"""
    rng = np.random.default_rng(42)
    h, w = 128, 128
    # 平滑地物（低频结构）
    yy, xx = np.mgrid[0:h, 0:w]
    base = 100 + 60 * np.sin(xx / 12.0) + 40 * np.cos(yy / 15.0)
    # 乘性斑点噪声
    speckle = rng.gamma(shape=2.0, scale=0.5, size=(h, w))
    img = np.clip(base * speckle, 0, 255).astype(np.uint8)
    path = tmp_path / "sample.png"
    imaging.write_png(path, img)
    return str(path), img


# ---------------------------------------------------------------- 元数据

def test_read_image_metadata(sample_image):
    path, _ = sample_image
    res = tools_sar.read_image_metadata(path)
    assert res.status == "success"
    assert res.data["width"] == 128
    assert res.data["height"] == 128
    assert res.data["bit_depth"] in (8, 16)


def test_read_metadata_missing_file():
    res = tools_sar.read_image_metadata("/nonexistent/x.png")
    assert res.status == "failed"
    assert "不存在" in (res.reason or "")


# ---------------------------------------------------------------- PNG 编解码

def test_png_roundtrip(tmp_path):
    """自带 PNG 编码器 → 解码器 应完全还原像素。"""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(37, 53), dtype=np.uint8)
    p = tmp_path / "rt.png"
    imaging.write_png(p, arr)
    img, meta = imaging.read_png(p)
    assert meta["width"] == 53 and meta["height"] == 37
    assert np.array_equal((img * 255).round().astype(np.uint8), arr)


def test_box_filter_matches_mean_of_window():
    """积分图盒式滤波应与逐窗口均值一致（验证实现正确性）。"""
    rng = np.random.default_rng(11)
    img = rng.random((16, 16)).astype(np.float32)
    out = imaging.box_filter(img, 3)
    # 中心像素：取 3x3 邻域均值（边界用 symmetric 填充，中心点不受影响）
    expect = img[0:3, 0:3].mean()
    assert abs(float(out[1, 1]) - expect) < 1e-5


# ---------------------------------------------------------------- 质量指标

def test_calculate_metrics(sample_image):
    path, _ = sample_image
    res = tools_sar.calculate_image_metrics(path)
    assert res.status == "success"
    m = res.data
    assert m["gray_mean"] is not None
    assert m["local_variance"] is not None
    assert m["cv"] is not None
    # ENL 应为正
    assert m["enl"] is None or m["enl"] > 0


def test_metrics_detect_speckle_high_cv():
    """纯斑点噪声图像的 CV 应显著大于平滑图像。"""
    rng = np.random.default_rng(3)
    smooth = np.tile(np.linspace(0.2, 0.8, 64), (64, 1)).astype(np.float32)
    speckled = (smooth * rng.gamma(2.0, 0.5, size=(64, 64))).astype(np.float32)

    lv_s = imaging.local_variance(smooth)
    lv_p = imaging.local_variance(speckled)
    assert lv_p > lv_s


# ---------------------------------------------------------------- 条纹检测

def test_detect_striping_runs(sample_image):
    path, _ = sample_image
    res = tools_sar.detect_striping(path)
    assert res.status == "success"
    assert "strength" in res.data
    assert res.data["method"].startswith("FFT")


def test_detect_striping_finds_horizontal_lines(tmp_path):
    """人为加入水平条纹，检测器应给出更高的强度与 0° 方向。"""
    h, w = 128, 128
    clean = np.full((h, w), 100.0)
    striped = clean.copy()
    striped[40:42, :] = 220.0  # 水平亮条

    p = tmp_path / "stripe.png"
    imaging.write_png(p, striped.astype(np.uint8))

    res = tools_sar.detect_striping(str(p))
    assert res.status == "success"
    assert res.data["strength"] > 0
    # 水平条纹 → 图像在 y 方向变化 → 能量落在 fx=0 列 → 方向 0°
    assert res.data["direction_deg"] == 0.0
    assert res.data["has_striping"] is True


def test_detect_striping_finds_vertical_lines(tmp_path):
    """垂直条纹应给出 90° 方向。"""
    h, w = 128, 128
    striped = np.full((h, w), 100.0)
    striped[:, 40:42] = 220.0  # 垂直亮条

    p = tmp_path / "stripe_v.png"
    imaging.write_png(p, striped.astype(np.uint8))

    res = tools_sar.detect_striping(str(p))
    assert res.status == "success"
    # 垂直条纹 → 图像在 x 方向变化 → 能量落在 fy=0 行 → 方向 90°
    assert res.data["direction_deg"] == 90.0
    assert res.data["has_striping"] is True


def test_clean_uniform_image_reports_no_striping(tmp_path):
    """完全均匀图像应报 0 强度、方向未知、不报警。"""
    h, w = 128, 128
    p = tmp_path / "clean.png"
    imaging.write_png(p, np.full((h, w), 100, dtype=np.uint8))

    res = tools_sar.detect_striping(str(p))
    assert res.status == "success"
    assert res.data["strength"] == 0.0
    assert res.data["has_striping"] is False


def test_random_noise_not_flagged_as_striping(tmp_path):
    """纯随机噪声不应被误判为条纹（强度须低于阈值）。"""
    rng = np.random.default_rng(1)
    p = tmp_path / "noise.png"
    imaging.write_png(p, rng.integers(0, 256, size=(128, 128), dtype=np.uint8))

    res = tools_sar.detect_striping(str(p))
    assert res.status == "success"
    assert res.data["has_striping"] is False


def test_striping_detector_separates_striped_from_clean(tmp_path):
    """条纹图与干净图的检测强度应有明显区分。"""
    h, w = 128, 128
    clean = np.full((h, w), 100.0)
    striped = clean.copy()
    striped[40:42, :] = 220.0

    pc = tmp_path / "clean.png"
    ps = tmp_path / "striped.png"
    imaging.write_png(pc, clean.astype(np.uint8))
    imaging.write_png(ps, striped.astype(np.uint8))

    s_clean = tools_sar.detect_striping(str(pc)).data["strength"]
    s_striped = tools_sar.detect_striping(str(ps)).data["strength"]
    assert s_striped > s_clean


# ---------------------------------------------------------------- L1 滤波

@pytest.mark.parametrize("method", ["median", "lee", "adaptive"])
def test_apply_filter_all_methods(sample_image, method):
    path, _ = sample_image
    res = tools_sar.apply_speckle_filter(path, method=method)
    assert res.status == "success"
    assert res.data["filter_level"] == "L1_classical"
    assert Path(res.data["output_path"]).exists()


def test_filter_rejects_unknown_method(sample_image):
    path, _ = sample_image
    res = tools_sar.apply_speckle_filter(path, method="nonexistent")
    assert res.status == "failed"
    assert "未知滤波方法" in (res.reason or "")


def test_lee_filter_reduces_speckle(sample_image):
    """Lee 滤波应降低局部方差（斑点被抑制）。"""
    path, _ = sample_image
    before = tools_sar.calculate_image_metrics(path).data["local_variance"]
    filtered = tools_sar.apply_speckle_filter(path, method="lee")
    after = tools_sar.calculate_image_metrics(filtered.data["output_path"]).data["local_variance"]
    assert after < before


def test_lee_filter_actually_reduces_cv(sample_image):
    """Lee 滤波必须真正降低 CV，而不只是"不劣化"。

    回归测试：早期实现用全图局部方差估计噪声方差，
    导致权重恒为 1、滤波器退化为恒等映射，CV 几乎不变。
    """
    path, _ = sample_image
    before = tools_sar.calculate_image_metrics(path).data["cv"]
    filtered = tools_sar.apply_speckle_filter(path, method="lee")
    after = tools_sar.calculate_image_metrics(filtered.data["output_path"]).data["cv"]
    # 至少要降 5% 才算有效抑制
    assert after < before * 0.95


def test_median_filter_reduces_speckle(sample_image):
    """中值滤波也应降低局部方差。"""
    path, _ = sample_image
    before = tools_sar.calculate_image_metrics(path).data["local_variance"]
    filtered = tools_sar.apply_speckle_filter(path, method="median", kernel_size=5)
    after = tools_sar.calculate_image_metrics(filtered.data["output_path"]).data["local_variance"]
    assert after < before


def test_filter_preserves_mean_intensity(sample_image):
    """滤波应保持平均辐射强度（否则地物反演会失真）。"""
    path, _ = sample_image
    before = tools_sar.calculate_image_metrics(path).data["gray_mean"]
    filtered = tools_sar.apply_speckle_filter(path, method="lee")
    after = tools_sar.calculate_image_metrics(filtered.data["output_path"]).data["gray_mean"]
    assert abs(after - before) / before < 0.05


# ---------------------------------------------------------------- 对比

def test_compare_quality_same_metrics(sample_image):
    path, _ = sample_image
    filtered = tools_sar.apply_speckle_filter(path, method="lee")
    res = tools_sar.compare_image_quality(path, filtered.data["output_path"])
    assert res.status == "success"
    assert "before" in res.data and "after" in res.data and "delta" in res.data
    # 处理前后使用同一套指标口径
    assert set(res.data["before"].keys()) == set(res.data["after"].keys())


def test_compare_quality_with_reference(sample_image):
    path, _ = sample_image
    filtered = tools_sar.apply_speckle_filter(path, method="lee")
    res = tools_sar.compare_image_quality(
        path, filtered.data["output_path"], reference_path=path
    )
    assert res.status == "success"
    assert "psnr" in res.data["reference_metrics"]
    assert "ssim" in res.data["reference_metrics"]


# ---------------------------------------------------------------- 路径一致性

def test_png_codec_matches_opencv_decode(sample_image, tmp_path):
    """自带 PNG 解码器必须与 OpenCV 解码结果逐像素一致。

    这条护栏防止两条代码路径（有无 OpenCV）悄悄分叉——
    分叉意味着"同一张图在不同机器上指标不同"，横评结论不可信。
    """
    cv2 = pytest.importorskip("cv2", reason="opencv 未安装，无法对比")
    path, original = sample_image

    # 用 OpenCV 读同一张 PNG
    via_cv2 = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
    # 用自带解码器读
    via_own, _ = imaging.read_png(path)

    assert via_own.shape == via_cv2.shape
    assert np.abs(via_own - via_cv2).max() < 1e-6


def test_metrics_identical_across_formats(sample_image, tmp_path):
    """同一张图存成 PNG 与 TIFF，指标必须完全一致。"""
    cv2 = pytest.importorskip("cv2", reason="opencv 未安装，无法写 TIFF")
    path, _ = sample_image

    tif = tmp_path / "same.tif"
    cv2.imwrite(str(tif), cv2.imread(path, cv2.IMREAD_UNCHANGED))

    m_png = tools_sar.calculate_image_metrics(path).data
    m_tif = tools_sar.calculate_image_metrics(str(tif)).data

    for k in ("gray_mean", "cv", "enl", "local_variance"):
        assert abs(m_png[k] - m_tif[k]) < 1e-9


def test_filter_output_identical_across_formats(sample_image, tmp_path):
    """PNG 与 TIFF 输入下，同一滤波算子的输出指标必须一致。"""
    cv2 = pytest.importorskip("cv2", reason="opencv 未安装，无法写 TIFF")
    path, _ = sample_image

    tif = tmp_path / "same.tif"
    cv2.imwrite(str(tif), cv2.imread(path, cv2.IMREAD_UNCHANGED))

    r_png = tools_sar.apply_speckle_filter(path, method="lee")
    r_tif = tools_sar.apply_speckle_filter(str(tif), method="lee")

    c_png = tools_sar.compare_image_quality(path, r_png.data["output_path"]).data
    c_tif = tools_sar.compare_image_quality(str(tif), r_tif.data["output_path"]).data

    assert abs(c_png["after"]["cv"] - c_tif["after"]["cv"]) < 1e-9
    assert abs(c_png["after"]["enl"] - c_tif["after"]["enl"]) < 1e-9
