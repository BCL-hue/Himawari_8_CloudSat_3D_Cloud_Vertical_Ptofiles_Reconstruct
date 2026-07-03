"""
MLP模型Case推理与三维可视化 - AHI + 几何信息 + BT扩展指标 + ERA5廓线(r,t)
使用已训练的MLP模型，对Case_data中的真实场进行整幅区域推理，
并生成二维/三维结果可视化。

输入:
- Case_data/AHI/*.nc
- Case_data/ERA5/{r,w,q,t,SKT}/*.nc
- Case_data/DEM/ETOPO2v2c_f4.nc

输出:
- 单时次小区域最大反射率平面图
- 单时次小区域云顶高度平面图
- 单时次小区域多高度层切片图
- 单时次小区域三维分布图 (透明表面渲染)

作者: Claude + Liu
日期: 2026-07-03
适配: Liu_20260702_train_MLP.py 训练的MLP模型
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from skimage import exposure
import cmaps

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.mpl.ticker as cticker

    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False


# ============================================
# 配置区域
# ============================================

CASE_ROOT = r"/mnt/g/3D_Cloud_Reflectivety_Profile_Projection/3D_Train_data/Case_data"
SAVE_DIR = r"/mnt/g/3D_Cloud_Reflectivety_Profile_Projection/Point_projection/CloudSat_GEOPROF/L_20260702_code/MLP/results/case"

BATCH_SIZE = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型配置 (与训练脚本 Liu_20260702_train_MLP.py 一致)
IN_DIM = 16 + 7 + 12 + 2 * 27  # AHI(16) + geo(7) + BT扩展(12) + ERA5(r,t各27层) = 89
OUT_DIM = 85
DROPOUT = 0.2

CHECKPOINT_PATH = Path(
    r"/mnt/g/3D_Cloud_Reflectivety_Profile_Projection/Point_projection"
    r"/CloudSat_GEOPROF/L_20260702_code/MLP/checkpoints/best_model.pt"
)

CASE_TIME_INDEX = 14

REGION_PRESETS = {
    "east_china_30_34_135_138": (30.0, 34.0, 135.0, 138.0),
    "japan_28p5_32_135_138": (28.5, 32.0, 135.0, 138.0),
    "south_china_sea_14p5_15p5_116_118": (14.5, 15.5, 116.0, 118.0),
    "philippines_8_9_120_121p5": (8.0, 9.0, 120.0, 121.5),
    "south_china_24_25_111_113": (24.0, 25.0, 111.0, 113.0),
    "custom_region": (30.5, 31.5, 137.5, 138.5),
    "Typhoon": (17, 19, 128.5, 130.5),
    "land_case": (29.5, 31.5, 111.5, 113.5),
    "Typhoon_0916": (29.8, 33.8, 117.7, 121.7),
    "Typhoon_0915": (24.0, 28.0, 114.9, 118.9),
    "Typhoon_0914": (20.5, 24.5, 117.9, 121.9),
    "Typhoon_0911": (14.7, 18.7, 132.4, 136.4),
    "Typhoon_0912": (16.3, 20.3, 127.4, 131.4),
    "Typhoon_0913": (18.2, 22.3, 122.4, 126.4),
}

ACTIVE_REGION_PRESET = "Typhoon"
CUSTOM_REGION_BOUNDS = None
CONTEXT_BOUNDS = (5.0, 40.0, 100.0, 140.0)

if CUSTOM_REGION_BOUNDS is None:
    ACTIVE_REGION_NAME = ACTIVE_REGION_PRESET
    (
        REGION_LAT_MIN,
        REGION_LAT_MAX,
        REGION_LON_MIN,
        REGION_LON_MAX,
    ) = REGION_PRESETS[ACTIVE_REGION_PRESET]
else:
    ACTIVE_REGION_NAME = "custom_region"
    (
        REGION_LAT_MIN,
        REGION_LAT_MAX,
        REGION_LON_MIN,
        REGION_LON_MAX,
    ) = CUSTOM_REGION_BOUNDS

(
    CONTEXT_LAT_MIN,
    CONTEXT_LAT_MAX,
    CONTEXT_LON_MIN,
    CONTEXT_LON_MAX,
) = CONTEXT_BOUNDS

SPATIAL_STRIDE = 1

ELEVATION_ANGLE = 45

# 3D渲染参数
SCATTER_THRESHOLD_DBZ = -30.0
SCATTER_MAX_POINTS = 800000


# ============================================
# 训练统计量 (与训练脚本一致)
# ============================================

CHANNEL_MEANS = np.array(
    [
        0.37, 0.35, 0.34, 0.38, 0.20, 0.17,
        280.54, 231.33, 238.30, 244.07, 259.59, 247.45,
        260.61, 259.00, 256.71, 249.12,
    ],
    dtype=np.float32,
)

CHANNEL_STDS = np.array(
    [
        0.23, 0.23, 0.23, 0.25, 0.12, 0.10,
        17.07, 9.31, 11.94, 14.20, 22.25, 14.30,
        22.95, 23.18, 22.34, 18.16,
    ],
    dtype=np.float32,
)

BT_DIFFS_EXTENDED_MEANS = np.array(
    [
        10.0274, 21.5406, -12.7420, -6.9704, -5.7717,
        2.2957, 3.9038, -256.1171, -11.4865,
        0.2382, 0.0270, 36.8777,
    ],
    dtype=np.float32,
)

BT_DIFFS_EXTENDED_STDS = np.array(
    [
        6.7557, 11.6445, 6.0438, 3.1137, 3.1410,
        1.7798, 2.7126, 23.5625, 6.0939,
        0.2806, 0.0602, 24.0085,
    ],
    dtype=np.float32,
)

ERA5_RH_LOW_MEANS = np.array(
    [73.5363, 76.9706, 78.5329, 77.5105, 76.3599, 75.4709,
     74.5204, 73.5544, 72.4292, 71.1167, 69.6088, 66.1056,
     63.3191, 62.3721],
    dtype=np.float32,
)
ERA5_RH_LOW_STDS = np.array(
    [16.8317, 18.5724, 19.2897, 19.0874, 19.2325, 19.8277,
     20.6616, 21.6271, 22.4810, 23.3625, 24.1576, 25.1004,
     26.0628, 28.3531],
    dtype=np.float32,
)

ERA5_T_LOW_MEANS = np.array(
    [296.62, 294.93, 293.40, 292.10, 290.83, 289.56,
     288.31, 287.05, 285.79, 284.50, 283.18, 280.35,
     277.15, 273.54],
    dtype=np.float32,
)
ERA5_T_LOW_STDS = np.array(
    [7.13, 7.07, 7.07, 7.10, 7.08, 7.01,
     6.90, 6.78, 6.65, 6.51, 6.38, 6.16,
     5.96, 5.73],
    dtype=np.float32,
)

ERA5_RH_HIGH_MEANS = np.array(
    [60.66, 57.94, 56.65, 57.42, 59.29, 61.07,
     63.14, 63.79, 63.20, 60.86, 57.69, 56.23, 57.08],
    dtype=np.float32,
)
ERA5_RH_HIGH_STDS = np.array(
    [30.45, 30.36, 30.54, 31.30, 32.08, 32.54,
     33.60, 34.76, 36.12, 37.45, 38.38, 38.82, 39.87],
    dtype=np.float32,
)

ERA5_T_HIGH_MEANS = np.array(
    [269.95, 265.89, 261.15, 255.59, 248.98, 241.10,
     231.86, 226.75, 221.27, 215.39, 209.16, 202.83, 197.97],
    dtype=np.float32,
)
ERA5_T_HIGH_STDS = np.array(
    [5.72, 5.89, 6.11, 6.32, 6.41, 6.15,
     5.07, 4.12, 3.14, 2.93, 3.97, 5.86, 7.31],
    dtype=np.float32,
)


# ============================================
# 通用函数
# ============================================


def denormalize_reflectivity(ref_norm):
    return (ref_norm + 1.0) * 55.0 / 2.0 - 35.0


def setup_paper_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.unicode_minus": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
        }
    )


def parse_case_timestamp(ahi_path):
    name = ahi_path.name
    match = re.search(r"NC_H08_(\d{8})_(\d{4})", name)
    if match is None:
        match = re.search(r"(\d{8})_(\d{4})", name)
    if match is None:
        raise ValueError(f"无法从文件名解析时次: {ahi_path.name}")
    date_str, time_str = match.groups()
    return {
        "date_str": date_str,
        "time_str": time_str,
        "stamp_compact": f"{date_str}_{time_str}",
        "stamp_era5": f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}_{time_str[:2]}_{time_str[2:4]}_00",
        "title_str": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}",
    }


def find_region_slices(lat2d, lon2d):
    return find_region_slices_by_bounds(
        lat2d, lon2d, REGION_LAT_MIN, REGION_LAT_MAX, REGION_LON_MIN, REGION_LON_MAX,
    )


def find_region_slices_by_bounds(lat2d, lon2d, lat_min, lat_max, lon_min, lon_max):
    region_mask = (
        (lat2d >= lat_min) & (lat2d <= lat_max)
        & (lon2d >= lon_min) & (lon2d <= lon_max)
    )
    if not np.any(region_mask):
        raise ValueError(
            f"指定区域内没有网格点: lat=[{lat_min}, {lat_max}], lon=[{lon_min}, {lon_max}]"
        )
    row_idx, col_idx = np.where(region_mask)
    row_slice = slice(row_idx.min(), row_idx.max() + 1)
    col_slice = slice(col_idx.min(), col_idx.max() + 1)
    return row_slice, col_slice


def load_netcdf_variable(file_path, candidate_names):
    with xr.open_dataset(file_path, decode_timedelta=False) as ds:
        data_var_names = list(ds.data_vars)
        lower_name_map = {name.lower(): name for name in data_var_names}

        for candidate in candidate_names:
            key = lower_name_map.get(candidate.lower())
            if key is not None:
                return np.array(ds[key].values)

        for candidate in candidate_names:
            for var_name in data_var_names:
                if candidate.lower() in var_name.lower():
                    return np.array(ds[var_name].values)

        all_names = list(ds.variables)
        raise KeyError(
            f"未找到候选变量: {candidate_names} in {file_path}. "
            f"Available variables: {all_names}"
        )


def squeeze_spatial_field(data):
    arr = np.asarray(data)
    while arr.ndim > 2:
        arr = arr[0]
    return np.asarray(arr)


def squeeze_profile_field(data):
    arr = np.asarray(data)
    while arr.ndim > 3:
        arr = arr[0]
    return np.asarray(arr)


def get_lat_lon_from_file(file_path):
    with xr.open_dataset(file_path, decode_timedelta=False) as ds:
        lat_key = None
        lon_key = None

        for name in list(ds.coords) + list(ds.variables):
            lower_name = name.lower()
            if lat_key is None and lower_name in ["latitude", "lat", "y"]:
                lat_key = name
            if lon_key is None and lower_name in ["longitude", "lon", "x"]:
                lon_key = name

        if lat_key is None or lon_key is None:
            raise KeyError(f"无法在文件中找到经纬度: {file_path}")

        lat = np.asarray(ds[lat_key].values).squeeze()
        lon = np.asarray(ds[lon_key].values).squeeze()
        return lat, lon


def to_2d_lat_lon(lat, lon, shape_hw):
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        lat2d = lat
        lon2d = lon
    else:
        raise ValueError(f"无法识别经纬度维度: lat={lat.shape}, lon={lon.shape}")

    if lat2d.shape != shape_hw or lon2d.shape != shape_hw:
        if lat2d.T.shape == shape_hw and lon2d.T.shape == shape_hw:
            lat2d = lat2d.T
            lon2d = lon2d.T
        else:
            raise ValueError(
                f"经纬度与AHI网格不匹配: lat={lat2d.shape}, lon={lon2d.shape}, ahi={shape_hw}"
            )

    return lat2d.astype(np.float32), lon2d.astype(np.float32)


def orient_grid_and_arrays(lat2d, lon2d, *arrays):
    lat2d = np.asarray(lat2d)
    lon2d = np.asarray(lon2d)
    arrays = [np.asarray(arr) for arr in arrays]

    if lat2d[0, 0] > lat2d[-1, 0]:
        lat2d = np.flip(lat2d, axis=0)
        lon2d = np.flip(lon2d, axis=0)
        arrays = [np.flip(arr, axis=0) for arr in arrays]

    if lon2d[0, 0] > lon2d[0, -1]:
        lat2d = np.flip(lat2d, axis=1)
        lon2d = np.flip(lon2d, axis=1)
        arrays = [np.flip(arr, axis=1) for arr in arrays]

    return (lat2d.astype(np.float32), lon2d.astype(np.float32), *arrays)


def compute_extended_bt_features(ahi, t_skin):
    """计算12个BT扩展指标 (与训练脚本一致)"""
    a01 = ahi[:, 0]
    a03 = ahi[:, 2]
    a04 = ahi[:, 3]
    a05 = ahi[:, 4]
    a06 = ahi[:, 5]
    bt07 = ahi[:, 6]
    bt08 = ahi[:, 7]
    bt09 = ahi[:, 8]
    bt10 = ahi[:, 9]
    bt11 = ahi[:, 10]
    bt13 = ahi[:, 12]
    bt14 = ahi[:, 13]
    bt15 = ahi[:, 14]
    bt16 = ahi[:, 15]

    eps = 1e-6
    vci = 255 * np.sqrt(((a01 - a03) ** 2 + (a01 - a04) ** 2 + (a03 - a04) ** 2) / 3)
    bt_7_14 = bt07 - bt14
    bt_8_10 = bt08 - bt10
    bt_8_9 = bt08 - bt09
    bt_9_10 = bt09 - bt10
    bt_14_15 = bt14 - bt15
    bt_13_15 = bt13 - bt15
    bt_11_14_15 = bt11 - bt14 - bt15
    bt_16_13 = bt16 - bt13
    ratio_3_5 = (a03 - a05) / (a03 + a05 + eps)
    albedo_5_6 = a05 - a06
    tskin_bt13 = t_skin - bt13

    return np.stack(
        [vci, bt_7_14, bt_8_10, bt_8_9, bt_9_10, bt_14_15,
         bt_13_15, bt_11_14_15, bt_16_13, ratio_3_5, albedo_5_6, tskin_bt13],
        axis=1,
    ).astype(np.float32)


def normalize_geometry_features(lat, lon, saa, saz, soz, soa, terrain):
    """归一化几何信息 (与训练脚本一致)"""
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon_lat_enc = np.stack(
        [np.sin(lon_rad), np.cos(lon_rad), np.sin(lat_rad), np.cos(lat_rad)],
        axis=1,
    )

    is_land = (terrain >= 0).astype(np.float32)
    land_height = np.where(terrain >= 0, np.minimum(terrain / 5500.0, 1.0), 0.0)
    ocean_depth = np.where(terrain < 0, np.minimum(-terrain / 10000.0, 1.0), 0.0)
    terrain_enc = np.stack([land_height, ocean_depth, is_land], axis=1)

    return np.concatenate([lon_lat_enc, terrain_enc], axis=1).astype(np.float32)


def normalize_era5_rt_profile(era5_profile):
    """
    归一化ERA5 r和t廓线 (与训练脚本一致)
    era5_profile: [N, 108] 布局为 [r(14)+w(14)+q(14)+t(14)+r(13)+w(13)+q(13)+t(13)]
    返回: [N, 54] 归一化后的 r(27层) + t(27层)
    """
    normalized = era5_profile.copy()

    # r: 低层14 + 高层13 = 27层
    # r_low: 索引 0:14, r_high: 索引 56:69 (偏移 14*4=56)
    for i in range(14):
        normalized[:, i] = (era5_profile[:, i] - ERA5_RH_LOW_MEANS[i]) / ERA5_RH_LOW_STDS[i]
    for i in range(13):
        normalized[:, 14 + i] = (era5_profile[:, 56 + i] - ERA5_RH_HIGH_MEANS[i]) / ERA5_RH_HIGH_STDS[i]

    # t: 低层14 + 高层13 = 27层
    # t_low: 索引 42:56 (偏移 14*3=42), t_high: 索引 95:108 (偏移 56+13*3=95)
    for i in range(14):
        normalized[:, 27 + i] = (era5_profile[:, 42 + i] - ERA5_T_LOW_MEANS[i]) / ERA5_T_LOW_STDS[i]
    for i in range(13):
        normalized[:, 27 + 14 + i] = (era5_profile[:, 95 + i] - ERA5_T_HIGH_MEANS[i]) / ERA5_T_HIGH_STDS[i]

    return normalized.astype(np.float32)


def nearest_indices(src_coords, target_coords):
    src_coords = np.asarray(src_coords).astype(np.float32)
    target_coords = np.asarray(target_coords).astype(np.float32)
    idx = np.searchsorted(src_coords, target_coords)
    idx = np.clip(idx, 1, len(src_coords) - 1)
    left = src_coords[idx - 1]
    right = src_coords[idx]
    take_left = np.abs(target_coords - left) <= np.abs(target_coords - right)
    return np.where(take_left, idx - 1, idx)


def sample_grid_to_targets(field, src_lat, src_lon, tgt_lat2d, tgt_lon2d):
    data = np.asarray(field)
    lat_axis = data.ndim - 2
    lon_axis = data.ndim - 1

    src_lat = np.asarray(src_lat).squeeze()
    src_lon = np.asarray(src_lon).squeeze()
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        data = np.flip(data, axis=lat_axis)
    if src_lon[0] > src_lon[-1]:
        src_lon = src_lon[::-1]
        data = np.flip(data, axis=lon_axis)

    lat_idx = nearest_indices(src_lat, tgt_lat2d.ravel())
    lon_idx = nearest_indices(src_lon, tgt_lon2d.ravel())

    if data.ndim == 2:
        sampled = data[lat_idx, lon_idx]
        return sampled.reshape(tgt_lat2d.shape).astype(np.float32)

    leading = int(np.prod(data.shape[:-2]))
    data_flat = data.reshape(leading, data.shape[-2], data.shape[-1])
    sampled = data_flat[:, lat_idx, lon_idx].transpose(1, 0)
    return sampled.astype(np.float32)


def interpolate_regular_grid_to_targets(field2d, src_lat, src_lon, tgt_lat2d, tgt_lon2d):
    src_lat = np.asarray(src_lat).squeeze().astype(np.float32)
    src_lon = np.asarray(src_lon).squeeze().astype(np.float32)
    field2d = np.asarray(field2d).astype(np.float32)

    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        field2d = np.flip(field2d, axis=0)
    if src_lon[0] > src_lon[-1]:
        src_lon = src_lon[::-1]
        field2d = np.flip(field2d, axis=1)

    lat_min = float(np.nanmin(tgt_lat2d))
    lat_max = float(np.nanmax(tgt_lat2d))
    lon_min = float(np.nanmin(tgt_lon2d))
    lon_max = float(np.nanmax(tgt_lon2d))

    lat_mask = (src_lat >= lat_min - 0.1) & (src_lat <= lat_max + 0.1)
    lon_mask = (src_lon >= lon_min - 0.1) & (src_lon <= lon_max + 0.1)

    if np.any(lat_mask):
        src_lat = src_lat[lat_mask]
        field2d = field2d[lat_mask, :]
    if np.any(lon_mask):
        src_lon = src_lon[lon_mask]
        field2d = field2d[:, lon_mask]

    da = xr.DataArray(
        field2d, coords={"lat": src_lat, "lon": src_lon}, dims=("lat", "lon"),
    )
    target_lat = xr.DataArray(tgt_lat2d.astype(np.float32), dims=("y", "x"))
    target_lon = xr.DataArray(tgt_lon2d.astype(np.float32), dims=("y", "x"))
    out = da.interp(lat=target_lat, lon=target_lon, method="linear")
    return np.asarray(out.values, dtype=np.float32)


def extract_era5_profile_components(case_root, stamp_era5, tgt_lat2d, tgt_lon2d):
    """提取ERA5廓线数据并返回归一化的 r(27层) + t(27层) = 54维"""
    era5_root = Path(case_root) / "ERA5"
    low_vars = []
    high_vars = []

    for var_name in ["r", "w", "q", "t"]:
        low_path = era5_root / var_name / f"{stamp_era5}_{var_name}_low_level.nc"
        high_path = era5_root / var_name / f"{stamp_era5}_{var_name}_high_level.nc"

        low_data = squeeze_profile_field(load_netcdf_variable(low_path, [var_name]))
        high_data = squeeze_profile_field(load_netcdf_variable(high_path, [var_name]))
        low_lat, low_lon = get_lat_lon_from_file(low_path)
        high_lat, high_lon = get_lat_lon_from_file(high_path)

        low_sampled = sample_grid_to_targets(low_data, low_lat, low_lon, tgt_lat2d, tgt_lon2d)
        high_sampled = sample_grid_to_targets(high_data, high_lat, high_lon, tgt_lat2d, tgt_lon2d)
        low_vars.append(low_sampled)
        high_vars.append(high_sampled)

    # era5_profile: [N, 108] = r(14)+w(14)+q(14)+t(14) + r(13)+w(13)+q(13)+t(13)
    low = np.concatenate(low_vars, axis=1)
    high = np.concatenate(high_vars, axis=1)
    era5_profile = np.concatenate([low, high], axis=1).astype(np.float32)

    # 归一化并提取 r + t (54维)
    era5_norm = normalize_era5_rt_profile(era5_profile)
    return era5_norm[:, :54]  # 前27=r, 后27=t


def extract_skt(case_root, stamp_era5, tgt_lat2d, tgt_lon2d):
    skt_path = Path(case_root) / "ERA5" / "SKT" / f"{stamp_era5}_skt.nc"
    skt_candidates = ["skt", "SKT", "Skt"]
    skt_data = squeeze_spatial_field(load_netcdf_variable(skt_path, skt_candidates))
    skt_lat, skt_lon = get_lat_lon_from_file(skt_path)
    return sample_grid_to_targets(skt_data, skt_lat, skt_lon, tgt_lat2d, tgt_lon2d).reshape(-1)


def extract_terrain(case_root, tgt_lat2d, tgt_lon2d):
    dem_path = Path(case_root) / "DEM" / "ETOPO2v2c_f4.nc"
    candidates = ["z", "elevation", "Band1", "topo", "dem"]
    terrain = squeeze_spatial_field(load_netcdf_variable(dem_path, candidates))
    dem_lat, dem_lon = get_lat_lon_from_file(dem_path)
    terrain_on_ahi = interpolate_regular_grid_to_targets(terrain, dem_lat, dem_lon, tgt_lat2d, tgt_lon2d)
    return terrain_on_ahi.reshape(-1)


def load_case_ahi_grid(ahi_path, lat_min, lat_max, lon_min, lon_max):
    channel_names = [
        "albedo_01", "albedo_02", "albedo_03", "albedo_04",
        "albedo_05", "albedo_06",
        "tbb_07", "tbb_08", "tbb_09", "tbb_10",
        "tbb_11", "tbb_12", "tbb_13", "tbb_14", "tbb_15", "tbb_16",
    ]

    channels = []
    for name in channel_names:
        data = squeeze_spatial_field(load_netcdf_variable(ahi_path, [name]))
        channels.append(data.astype(np.float32))

    ahi_cube = np.stack(channels, axis=-1)

    lat, lon = get_lat_lon_from_file(ahi_path)
    lat2d, lon2d = to_2d_lat_lon(lat, lon, ahi_cube.shape[:2])

    row_slice, col_slice = find_region_slices_by_bounds(
        lat2d, lon2d, lat_min, lat_max, lon_min, lon_max,
    )
    ahi_cube = ahi_cube[row_slice, col_slice]
    lat2d = lat2d[row_slice, col_slice]
    lon2d = lon2d[row_slice, col_slice]

    ahi_cube = ahi_cube[::SPATIAL_STRIDE, ::SPATIAL_STRIDE]
    lat2d = lat2d[::SPATIAL_STRIDE, ::SPATIAL_STRIDE]
    lon2d = lon2d[::SPATIAL_STRIDE, ::SPATIAL_STRIDE]
    return ahi_cube, lat2d, lon2d


def load_case_geometry_angles(ahi_path, lat_min, lat_max, lon_min, lon_max):
    saa = squeeze_spatial_field(load_netcdf_variable(ahi_path, ["SAA"]))
    saz = squeeze_spatial_field(load_netcdf_variable(ahi_path, ["SAZ"]))
    soz = squeeze_spatial_field(load_netcdf_variable(ahi_path, ["SOZ"]))
    soa = squeeze_spatial_field(load_netcdf_variable(ahi_path, ["SOA"]))

    lat, lon = get_lat_lon_from_file(ahi_path)
    lat2d, lon2d = to_2d_lat_lon(lat, lon, saa.shape)
    row_slice, col_slice = find_region_slices_by_bounds(
        lat2d, lon2d, lat_min, lat_max, lon_min, lon_max,
    )

    saa = saa[row_slice, col_slice][::SPATIAL_STRIDE, ::SPATIAL_STRIDE]
    saz = saz[row_slice, col_slice][::SPATIAL_STRIDE, ::SPATIAL_STRIDE]
    soz = soz[row_slice, col_slice][::SPATIAL_STRIDE, ::SPATIAL_STRIDE]
    soa = soa[row_slice, col_slice][::SPATIAL_STRIDE, ::SPATIAL_STRIDE]
    return (
        saa.astype(np.float32), saz.astype(np.float32),
        soz.astype(np.float32), soa.astype(np.float32),
    )


def build_case_features(case_root, ahi_path, lat_min, lat_max, lon_min, lon_max):
    """
    构建Case推理所需的输入特征 (仅MLP模式)

    Returns:
        {"inputs": [N, 89], "shape_hw": (h, w), "lat2d", "lon2d", "ahi_cube", "time_info", "is_land"}
    """
    time_info = parse_case_timestamp(ahi_path)
    ahi_cube, lat2d, lon2d = load_case_ahi_grid(ahi_path, lat_min, lat_max, lon_min, lon_max)
    h, w, _ = ahi_cube.shape
    saa2d, saz2d, soz2d, soa2d = load_case_geometry_angles(
        ahi_path, lat_min, lat_max, lon_min, lon_max,
    )
    lat2d, lon2d, ahi_cube, saa2d, saz2d, soz2d, soa2d = orient_grid_and_arrays(
        lat2d, lon2d, ahi_cube, saa2d, saz2d, soz2d, soa2d,
    )

    lat_flat = lat2d.reshape(-1)
    lon_flat = lon2d.reshape(-1)
    ahi_flat = ahi_cube.reshape(-1, 16)
    saa = saa2d.reshape(-1)
    saz = saz2d.reshape(-1)
    soz = soz2d.reshape(-1)
    soa = soa2d.reshape(-1)

    terrain = extract_terrain(case_root, lat2d, lon2d)
    skt = extract_skt(case_root, time_info["stamp_era5"], lat2d, lon2d)

    ahi_normalized = (ahi_flat - CHANNEL_MEANS[None, :]) / CHANNEL_STDS[None, :]

    geo_normalized = normalize_geometry_features(lat_flat, lon_flat, saa, saz, soz, soa, terrain)

    bt_features = compute_extended_bt_features(ahi_flat, skt)
    bt_normalized = (bt_features - BT_DIFFS_EXTENDED_MEANS[None, :]) / BT_DIFFS_EXTENDED_STDS[None, :]

    era5_rt = extract_era5_profile_components(case_root, time_info["stamp_era5"], lat2d, lon2d)

    inputs = np.concatenate([ahi_normalized, geo_normalized, bt_normalized, era5_rt], axis=1).astype(np.float32)

    return {
        "inputs": inputs,
        "shape_hw": (h, w),
        "lat2d": lat2d,
        "lon2d": lon2d,
        "ahi_cube": ahi_cube,
        "time_info": time_info,
        "is_land": geo_normalized[:, 6].reshape(h, w),
    }


# ============================================
# 模型类 (与训练脚本一致)
# ============================================


class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim=89, out_dim=85, dropout=0.2):
        super().__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.output_layer = torch.nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.output_layer(x)


# ============================================
# 绘图函数
# ============================================


def add_map_subplot(fig, subplot_spec):
    if CARTOPY_AVAILABLE:
        return fig.add_subplot(subplot_spec, projection=ccrs.PlateCarree())
    return fig.add_subplot(subplot_spec)


def style_map_axis(ax, lon_range=None, lat_range=None):
    if not CARTOPY_AVAILABLE:
        return

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    if lon_range is None:
        lon_range = [CONTEXT_LON_MIN, CONTEXT_LON_MAX]
    if lat_range is None:
        lat_range = [CONTEXT_LAT_MIN, CONTEXT_LAT_MAX]

    lon_ticks = np.arange(lon_range[0], lon_range[1] + 1, 10)
    lat_ticks = np.arange(lat_range[0], lat_range[1] + 1, 10)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())

    lon_minor_ticks = np.arange(lon_range[0], lon_range[1] + 1, 5)
    lat_minor_ticks = np.arange(lat_range[0], lat_range[1] + 1, 5)
    ax.set_xticks(lon_minor_ticks, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_minor_ticks, minor=True, crs=ccrs.PlateCarree())

    ax.minorticks_on()


def plot_plan_view(ax, lon2d, lat2d, field2d, title, cbar_label, cmap, vmin=None, vmax=None, shrink=0.8):
    mesh_kwargs = {"cmap": cmap, "shading": "auto", "vmin": vmin, "vmax": vmax}
    if CARTOPY_AVAILABLE:
        mesh_kwargs["transform"] = ccrs.PlateCarree()

    mesh = ax.pcolormesh(lon2d, lat2d, field2d, **mesh_kwargs)
    ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    style_map_axis(ax)
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02, shrink=shrink)
    cbar.set_label(cbar_label, fontsize=11, fontweight="bold")
    return mesh


def add_region_box_from_grid(ax, lon2d, lat2d):
    rect = Rectangle(
        (float(np.nanmin(lon2d)), float(np.nanmin(lat2d))),
        float(np.nanmax(lon2d) - np.nanmin(lon2d)),
        float(np.nanmax(lat2d) - np.nanmin(lat2d)),
        fill=False, edgecolor="black", linewidth=2.2, linestyle="-", zorder=6,
    )
    ax.add_patch(rect)


def set_context_extent(ax):
    if CARTOPY_AVAILABLE:
        ax.set_extent(
            [CONTEXT_LON_MIN, CONTEXT_LON_MAX, CONTEXT_LAT_MIN, CONTEXT_LAT_MAX],
            crs=ccrs.PlateCarree(),
        )
    else:
        ax.set_xlim(CONTEXT_LON_MIN, CONTEXT_LON_MAX)
        ax.set_ylim(CONTEXT_LAT_MIN, CONTEXT_LAT_MAX)


def True_Color_Image(himawari_albedo_01, himawari_albedo_02, himawari_albedo_03, himawari_albedo_04):
    bdata = np.clip(np.copy(himawari_albedo_01), 0, 1)
    g1data = np.clip(np.copy(himawari_albedo_02), 0, 1)
    rdata = np.clip(np.copy(himawari_albedo_03), 0, 1)
    g2data = np.clip(np.copy(himawari_albedo_04), 0, 1)

    hybrid_g = np.clip((1 - 0.07) * g1data + 0.07 * g2data, 0, 1)

    img = np.zeros((himawari_albedo_01.shape[0], himawari_albedo_01.shape[1], 3))
    img[:, :, 0] = rdata
    img[:, :, 1] = hybrid_g
    img[:, :, 2] = bdata

    for i in range(3):
        img[:, :, i] = exposure.equalize_hist(img[:, :, i])

    return img


def plot_true_color(ax, lon2d, lat2d, ahi_cube, title):
    rgb = True_Color_Image(ahi_cube[:, :, 0], ahi_cube[:, :, 1], ahi_cube[:, :, 2], ahi_cube[:, :, 3])
    extent = [float(np.nanmin(lon2d)), float(np.nanmax(lon2d)),
              float(np.nanmin(lat2d)), float(np.nanmax(lat2d))]
    imshow_kwargs = {"extent": extent, "origin": "lower", "aspect": "auto"}
    if CARTOPY_AVAILABLE:
        imshow_kwargs["transform"] = ccrs.PlateCarree()
    ax.imshow(rgb, **imshow_kwargs)
    ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    style_map_axis(ax)


def plot_height_slice(ax, lon2d, lat2d, volume_dbz, heights, target_km, panel_label):
    idx = int(np.argmin(np.abs(heights - target_km)))
    plot_plan_view(
        ax, lon2d, lat2d, volume_dbz[:, :, idx],
        f"{panel_label} Bin {idx:02d} ({heights[idx]:.1f} km)",
        "Reflectivity (dBZ)", cmaps.MPL_jet, vmin=-40, vmax=25,
    )


def sample_line_indices(start_row, start_col, end_row, end_col, n_samples):
    row_vals = np.rint(np.linspace(start_row, end_row, n_samples)).astype(int)
    col_vals = np.rint(np.linspace(start_col, end_col, n_samples)).astype(int)
    indices = []
    for r, c in zip(row_vals, col_vals):
        if not indices or indices[-1] != (r, c):
            indices.append((r, c))
    rows = np.array([item[0] for item in indices], dtype=int)
    cols = np.array([item[1] for item in indices], dtype=int)
    return rows, cols


def format_latlon_label(lat_value, lon_value):
    return f"{lat_value:.2f}N\n{lon_value:.2f}E"


def build_region_profile_specs(lat2d, lon2d, volume_dbz):
    n_rows, n_cols = lat2d.shape
    center_row, center_col = n_rows // 2, n_cols // 2

    diag_main_rows, diag_main_cols = sample_line_indices(0, 0, n_rows - 1, n_cols - 1, max(n_rows, n_cols))
    diag_anti_rows, diag_anti_cols = sample_line_indices(0, n_cols - 1, n_rows - 1, 0, max(n_rows, n_cols))

    return [
        {
            "title": f"(a) Center Latitude ({np.nanmean(lat2d[center_row, :]):.2f}°N)",
            "section": volume_dbz[center_row, :, :],
            "axis_values": lon2d[center_row, :],
            "tick_labels": [f"{v:.2f}E" for v in lon2d[center_row, :]],
            "xlabel": "Longitude",
        },
        {
            "title": f"(b) Center Longitude ({np.nanmean(lon2d[:, center_col]):.2f}°E)",
            "section": volume_dbz[:, center_col, :],
            "axis_values": lat2d[:, center_col],
            "tick_labels": [f"{v:.2f}N" for v in lat2d[:, center_col]],
            "xlabel": "Latitude",
        },
        {
            "title": "(c) Diagonal NW-SE",
            "section": volume_dbz[diag_main_rows, diag_main_cols, :],
            "axis_values": np.arange(len(diag_main_rows), dtype=np.float32),
            "tick_labels": [format_latlon_label(lat2d[r, c], lon2d[r, c])
                            for r, c in zip(diag_main_rows, diag_main_cols)],
            "xlabel": "Path Point (Lat/Lon)",
        },
        {
            "title": "(d) Diagonal NE-SW",
            "section": volume_dbz[diag_anti_rows, diag_anti_cols, :],
            "axis_values": np.arange(len(diag_anti_rows), dtype=np.float32),
            "tick_labels": [format_latlon_label(lat2d[r, c], lon2d[r, c])
                            for r, c in zip(diag_anti_rows, diag_anti_cols)],
            "xlabel": "Path Point (Lat/Lon)",
        },
    ]


def plot_vertical_profile_section(ax, section, axis_values, heights, title, xlabel, tick_labels):
    heights_plot = heights[::-1]
    section_plot = section[:, ::-1].T

    mesh = ax.pcolormesh(axis_values, heights_plot, section_plot,
                          shading="auto", cmap=cmaps.MPL_jet, vmin=-40, vmax=25)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Height (km)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, float(np.nanmax(heights)))

    tick_count = min(6, len(axis_values))
    tick_positions = np.linspace(0, len(axis_values) - 1, tick_count).round().astype(int)
    tick_positions = np.unique(tick_positions)
    ax.set_xticks(axis_values[tick_positions])
    ax.set_xticklabels([tick_labels[idx] for idx in tick_positions], rotation=0, ha="center")

    cbar = plt.colorbar(mesh, ax=ax, pad=0.02, shrink=0.86)
    cbar.set_label("Reflectivity (dBZ)", fontsize=10, fontweight="bold")


def save_case_region_profile_figure(save_dir, prefix, lon2d, lat2d, volume_dbz, heights, time_title):
    profile_specs = build_region_profile_specs(lat2d, lon2d, volume_dbz)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    axes = axes.ravel()

    for ax, spec in zip(axes, profile_specs):
        plot_vertical_profile_section(
            ax, spec["section"], spec["axis_values"], heights,
            spec["title"], spec["xlabel"], spec["tick_labels"],
        )

    fig.suptitle(f"MLP - Region Vertical Profiles\n{time_title}", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = save_dir / f"{prefix}_region_profiles.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  保存: {save_path}")
    plt.close()


def save_case_summary_maps(save_dir, prefix, lon2d, lat2d, ahi_cube, volume_dbz, heights,
                           time_title, region_lon2d=None, region_lat2d=None, add_box=True):
    max_refl = np.max(volume_dbz, axis=2)
    cth_mask = max_refl >= -15.0
    cth_idx = np.argmax(volume_dbz, axis=2)
    cth = np.where(cth_mask, heights[cth_idx], np.nan)

    bt13 = ahi_cube[:, :, 12]

    fig = plt.figure(figsize=(24, 10.5))
    gs = fig.add_gridspec(2, 4, wspace=0.18, hspace=0.20)

    # 第一行
    ax1 = add_map_subplot(fig, gs[0, 0])
    plot_true_color(ax1, lon2d, lat2d, ahi_cube, "(a) Himawari-8 True Color")
    set_context_extent(ax1)
    if add_box and region_lon2d is not None and region_lat2d is not None:
        add_region_box_from_grid(ax1, region_lon2d, region_lat2d)

    ax2 = add_map_subplot(fig, gs[0, 1])
    plot_plan_view(ax2, lon2d, lat2d, bt13, "(b) BT13 (Brightness Temperature)",
                   "Temperature (K)", "RdYlBu_r", vmin=190, vmax=300)
    set_context_extent(ax2)
    if add_box and region_lon2d is not None and region_lat2d is not None:
        add_region_box_from_grid(ax2, region_lon2d, region_lat2d)

    ax3 = add_map_subplot(fig, gs[0, 2])
    plot_plan_view(ax3, lon2d, lat2d, max_refl, "(c) Column Maximum Reflectivity",
                   "Reflectivity (dBZ)", cmaps.MPL_jet, vmin=-40, vmax=25)
    set_context_extent(ax3)
    if add_box and region_lon2d is not None and region_lat2d is not None:
        add_region_box_from_grid(ax3, region_lon2d, region_lat2d)

    ax4 = add_map_subplot(fig, gs[0, 3])
    plot_plan_view(ax4, lon2d, lat2d, cth, "(d) Echo-top Height of Max Reflectivity",
                   "Height (km)", "viridis", vmin=0, vmax=15.4)
    set_context_extent(ax4)
    if add_box and region_lon2d is not None and region_lat2d is not None:
        add_region_box_from_grid(ax4, region_lon2d, region_lat2d)

    # 第二行：四个高度层切片
    slice_heights = [2.0, 5.0, 10.0, 15.0]
    labels = ["(e)", "(f)", "(g)", "(h)"]
    for i, (sh, label) in enumerate(zip(slice_heights, labels)):
        ax = add_map_subplot(fig, gs[1, i])
        plot_height_slice(ax, lon2d, lat2d, volume_dbz, heights, sh, label)
        set_context_extent(ax)
        if add_box and region_lon2d is not None and region_lat2d is not None:
            add_region_box_from_grid(ax, region_lon2d, region_lat2d)

    fig.suptitle(f"MLP - Case Reflectivity Field Summary\n{time_title}",
                 fontsize=16, fontweight="bold", y=0.98)
    save_path = save_dir / f"{prefix}_summary_maps.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  保存: {save_path}")
    plt.close()

    return max_refl, cth


def make_3d_scatter(ax, lon2d, lat2d, volume_dbz, heights, threshold_dbz=SCATTER_THRESHOLD_DBZ):
    mask = volume_dbz >= threshold_dbz
    if not np.any(mask):
        ax.text2D(0.35, 0.5, "No voxels above threshold", transform=ax.transAxes,
                  fontsize=13, fontweight="bold")
        return 0

    yy, xx, zz = np.where(mask)
    values = volume_dbz[yy, xx, zz]
    if len(values) > SCATTER_MAX_POINTS:
        order = np.argsort(values)[-SCATTER_MAX_POINTS:]
        yy, xx, zz, values = yy[order], xx[order], zz[order], values[order]

    xs = lon2d[yy, xx]
    ys = lat2d[yy, xx]
    zs = heights[zz]

    sc = ax.scatter(xs, ys, zs, c=values, cmap=cmaps.MPL_jet, vmin=-40, vmax=25,
                    s=6, alpha=0.75, linewidths=0)
    ax.set_xlabel("Longitude", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Latitude", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_zlabel("Height (km)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("(d) 3D Reflectivity Distribution", fontsize=13, fontweight="bold", loc="left")
    ax.view_init(elev=ELEVATION_ANGLE, azim=-61)
    cbar = plt.colorbar(sc, ax=ax, pad=0.08, shrink=0.48)
    cbar.set_label("Reflectivity (dBZ)", fontsize=10, fontweight="bold")
    return len(values)


def add_projected_region_box(ax, lon2d, lat2d, volume_dbz, threshold_dbz=SCATTER_THRESHOLD_DBZ):
    mask = volume_dbz >= threshold_dbz
    if not np.any(mask):
        return

    horizontal_mask = np.any(mask, axis=2)
    row_idx, col_idx = np.where(horizontal_mask)
    lon_sel = lon2d[row_idx, col_idx]
    lat_sel = lat2d[row_idx, col_idx]

    rect = Rectangle(
        (float(np.min(lon_sel)), float(np.min(lat_sel))),
        float(np.max(lon_sel) - np.min(lon_sel)),
        float(np.max(lat_sel) - np.min(lat_sel)),
        fill=False, edgecolor="black", linewidth=2.2, linestyle="-", zorder=6,
    )
    ax.add_patch(rect)


def save_case_height_bin_maps(save_dir, prefix, lon2d, lat2d, volume_dbz, heights,
                              time_title, bin_step=15):
    """3D透明表面渲染 - 使用 plot_surface + alpha 掩码"""
    level_indices = list(range(0, len(heights), bin_step))
    fig = plt.figure(figsize=(13.5, 10.5))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap(cmaps.MPL_jet)
    norm = mpl.colors.Normalize(vmin=-40, vmax=25)

    for level_idx in level_indices:
        plane_values = volume_dbz[:, :, level_idx]
        z_plane = np.full_like(lon2d, heights[level_idx], dtype=np.float32)
        facecolors = cmap(norm(plane_values))
        facecolors[..., 3] = np.where(plane_values >= SCATTER_THRESHOLD_DBZ, 0.78, 0.10)

        ax.plot_surface(
            lon2d, lat2d, z_plane,
            rstride=1, cstride=1,
            facecolors=facecolors, linewidth=0,
            antialiased=False, shade=False,
        )

        ax.text(
            float(np.nanmax(lon2d)), float(np.nanmax(lat2d)),
            float(heights[level_idx]), f"bin {level_idx:02d}",
            fontsize=9, color="black",
        )

    ax.set_xlabel("Longitude", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Latitude", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_zlabel("Height (km)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("(a) 3D Height-bin Reflectivity Planes", fontsize=13, fontweight="bold", loc="left")
    ax.view_init(elev=15, azim=-58)
    ax.set_zlim(0, float(np.nanmax(heights)))

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.08, shrink=0.62)
    cbar.set_label("Reflectivity (dBZ)", fontsize=10, fontweight="bold")

    sampled_bins = ", ".join(str(idx) for idx in level_indices)
    ax.text2D(
        0.02, 0.98,
        f"Bin step = {bin_step}\nSampled bins = {sampled_bins}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="gray"),
    )

    fig.suptitle(f"MLP - 3D Height-bin Planes\n{time_title}",
                 fontsize=16, fontweight="bold", y=0.98)

    save_path = save_dir / f"{prefix}_height_bin_maps_step{bin_step}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  保存: {save_path}")
    plt.close()


def save_case_3d_figure(save_dir, prefix, lon2d_context, lat2d_context,
                         volume_dbz_context, lon2d_small, lat2d_small,
                         volume_dbz_small, heights, time_title):
    max_refl = np.max(volume_dbz_context, axis=2)
    fig = plt.figure(figsize=(18, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.12)

    ax1 = add_map_subplot(fig, gs[0, 0])
    plot_plan_view(ax1, lon2d_context, lat2d_context, max_refl,
                   "(a) Column Maximum Reflectivity", "Reflectivity (dBZ)",
                   cmaps.MPL_jet, vmin=-40, vmax=25, shrink=0.6)
    set_context_extent(ax1)
    add_projected_region_box(ax1, lon2d_small, lat2d_small, volume_dbz_small)

    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    n_points = make_3d_scatter(ax2, lon2d_small, lat2d_small, volume_dbz_small, heights)
    ax2.text2D(
        0.02, 0.98,
        f"Threshold = {SCATTER_THRESHOLD_DBZ:.1f} dBZ\nDisplayed voxels = {n_points:,}",
        transform=ax2.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="gray"),
    )

    fig.suptitle(f"MLP - Case 3D Reflectivity Distribution\n{time_title}",
                 fontsize=16, fontweight="bold", y=0.97)
    save_path = save_dir / f"{prefix}_3d_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  保存: {save_path}")
    plt.close()


def save_case_3d_volume_figure(save_dir, prefix, lon2d, lat2d, volume_dbz, heights, time_title):
    """3D体积渲染图 - 多层半透明表面 + dBZ阈值等值面效果"""
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap(cmaps.MPL_jet)
    norm = mpl.colors.Normalize(vmin=-40, vmax=25)

    bin_step = 5
    level_indices = list(range(0, len(heights), bin_step))

    for level_idx in level_indices:
        plane_values = volume_dbz[:, :, level_idx]
        z_val = heights[level_idx]
        z_plane = np.full_like(lon2d, z_val, dtype=np.float32)

        facecolors = cmap(norm(plane_values))

        alpha = np.full_like(plane_values, 0.03)
        alpha[(plane_values >= -30) & (plane_values < -15)] = 0.15
        alpha[(plane_values >= -15) & (plane_values < 0)] = 0.28
        alpha[(plane_values >= 0) & (plane_values < 10)] = 0.45
        alpha[(plane_values >= 10) & (plane_values < 20)] = 0.62
        alpha[plane_values >= 20] = 0.80
        facecolors[..., 3] = alpha

        ax.plot_surface(
            lon2d, lat2d, z_plane,
            rstride=1, cstride=1,
            facecolors=facecolors, linewidth=0,
            antialiased=False, shade=False,
        )

    label_levels = [2.0, 5.0, 10.0, 15.0, 20.0]
    for target_km in label_levels:
        idx = int(np.argmin(np.abs(heights - target_km)))
        z_val = heights[idx]
        ax.text(
            float(np.nanmin(lon2d)),
            float(np.nanmax(lat2d)),
            z_val,
            f" {target_km:.0f}km",
            fontsize=8, color="#333333", alpha=0.7,
        )

    ax.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_zlabel("Height (km)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("3D Volume Rendering", fontsize=14, fontweight="bold", loc="left", pad=12)
    ax.view_init(elev=25, azim=-65)
    ax.set_zlim(0, float(np.nanmax(heights)))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, alpha=0.2)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.06, shrink=0.55, aspect=20)
    cbar.set_label("Reflectivity (dBZ)", fontsize=11, fontweight="bold")

    total_voxels = volume_dbz.size
    valid_voxels = int(np.sum(volume_dbz >= SCATTER_THRESHOLD_DBZ))
    max_dbz = float(np.max(volume_dbz))
    ax.text2D(
        0.02, 0.97,
        f"Height bin step = {bin_step}\n"
        f"Total voxels = {total_voxels:,}\n"
        f"Cloud voxels (≥ {SCATTER_THRESHOLD_DBZ:.0f} dBZ) = {valid_voxels:,}\n"
        f"Max reflectivity = {max_dbz:.1f} dBZ",
        transform=ax.transAxes, fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="gray"),
    )

    fig.suptitle(f"MLP - 3D Volume Rendering\n{time_title}",
                 fontsize=16, fontweight="bold", y=0.97)

    save_path = save_dir / f"{prefix}_3d_volume_rendering.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  保存: {save_path}")
    plt.close()


def save_case_statistics_csv(save_dir, prefix, lon2d, lat2d, volume_dbz, heights, is_land):
    max_refl = np.max(volume_dbz, axis=2)
    cth_mask = max_refl >= -15.0
    cth_idx = np.argmax(volume_dbz, axis=2)
    cth = np.where(cth_mask, heights[cth_idx], np.nan)

    stats = {
        "grid_points": int(volume_dbz.shape[0] * volume_dbz.shape[1]),
        "valid_echo_points": int(np.sum(volume_dbz >= SCATTER_THRESHOLD_DBZ)),
        "max_reflectivity_mean": float(np.nanmean(max_refl)),
        "max_reflectivity_p95": float(np.nanpercentile(max_refl, 95)),
        "cth_mean_km": float(np.nanmean(cth)),
        "cth_p95_km": (
            float(np.nanpercentile(cth[np.isfinite(cth)], 95))
            if np.any(np.isfinite(cth)) else np.nan
        ),
        "land_fraction": float(np.mean(is_land >= 0.5)),
        "lon_min": float(np.nanmin(lon2d)),
        "lon_max": float(np.nanmax(lon2d)),
        "lat_min": float(np.nanmin(lat2d)),
        "lat_max": float(np.nanmax(lat2d)),
    }

    df = pd.DataFrame([stats])
    save_path = save_dir / f"{prefix}_statistics.csv"
    df.to_csv(save_path, index=False)
    print(f"  保存: {save_path}")


# ============================================
# 主流程
# ============================================


def predict_case_volume(model, inputs, shape_hw):
    """MLP模型推理"""
    preds = []
    n_samples = len(inputs)

    with torch.no_grad():
        for start in range(0, n_samples, BATCH_SIZE):
            batch_obs = torch.from_numpy(inputs[start : start + BATCH_SIZE]).to(DEVICE)
            pred = model(batch_obs)
            preds.append(pred.cpu().numpy())

    pred_norm = np.concatenate(preds, axis=0)
    pred_dbz = denormalize_reflectivity(pred_norm)
    h, w = shape_hw
    return pred_dbz.reshape(h, w, OUT_DIM).astype(np.float32)


def main():
    setup_paper_style()

    print("=" * 60)
    print("MLP模型Case推理与三维可视化")
    print("=" * 60)
    print(f"模型: SimpleMLP (in_dim={IN_DIM}, out_dim={OUT_DIM})")
    print(f"设备: {DEVICE}")
    print(f"测试时次索引: {CASE_TIME_INDEX}")
    print(f"区域预设: {ACTIVE_REGION_NAME}")
    print(f"测试区域: {REGION_LAT_MIN:.1f}-{REGION_LAT_MAX:.1f}N, {REGION_LON_MIN:.1f}-{REGION_LON_MAX:.1f}E")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    save_dir = Path(SAVE_DIR) / ACTIVE_REGION_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[加载模型] {CHECKPOINT_PATH}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")

    model = SimpleMLP(in_dim=IN_DIM, out_dim=OUT_DIM, dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    case_root = Path(CASE_ROOT)
    ahi_files = sorted((case_root / "AHI").glob("NC_H08_*.nc"))
    print(ahi_files)
    if not ahi_files:
        raise FileNotFoundError(f"未在{case_root / 'AHI'}找到Case AHI文件")
    if CASE_TIME_INDEX < 0 or CASE_TIME_INDEX >= len(ahi_files):
        raise IndexError(f"CASE_TIME_INDEX超出范围: {CASE_TIME_INDEX}, 文件数={len(ahi_files)}")

    heights = np.linspace(20.4, 0, OUT_DIM, dtype=np.float32)

    ahi_path = ahi_files[CASE_TIME_INDEX]
    time_info = parse_case_timestamp(ahi_path)
    prefix = time_info["stamp_compact"]
    print(f"\n[Case文件数] {len(ahi_files)}")
    print(f"[处理时次] {ahi_path.name}")
    print(f"  时次: {time_info['title_str']}")

    # 构建输入数据
    context_data = build_case_features(
        case_root, ahi_path,
        CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
        CONTEXT_LON_MIN, CONTEXT_LON_MAX,
    )

    print(
        f"  上下文网格: {context_data['shape_hw'][0]} x {context_data['shape_hw'][1]} "
        f"({context_data['inputs'].shape[0]:,} points)"
    )
    print(f"  输入维度: {context_data['inputs'].shape[1]}")

    # 模型推理
    volume_dbz_context = predict_case_volume(model, context_data["inputs"], context_data["shape_hw"])

    # 提取小区域
    row_slice_small, col_slice_small = find_region_slices_by_bounds(
        context_data["lat2d"], context_data["lon2d"],
        REGION_LAT_MIN, REGION_LAT_MAX, REGION_LON_MIN, REGION_LON_MAX,
    )
    case_data = {
        "lon2d": context_data["lon2d"][row_slice_small, col_slice_small],
        "lat2d": context_data["lat2d"][row_slice_small, col_slice_small],
        "is_land": context_data["is_land"][row_slice_small, col_slice_small],
        "shape_hw": (
            row_slice_small.stop - row_slice_small.start,
            col_slice_small.stop - col_slice_small.start,
        ),
    }
    volume_dbz = volume_dbz_context[row_slice_small, col_slice_small, :]
    print(
        f"  小区域网格: {case_data['shape_hw'][0]} x {case_data['shape_hw'][1]} "
        f"({case_data['shape_hw'][0] * case_data['shape_hw'][1]:,} points)"
    )

    # 生成可视化图
    save_case_summary_maps(
        save_dir, prefix,
        context_data["lon2d"], context_data["lat2d"], context_data["ahi_cube"],
        volume_dbz_context, heights, time_info["title_str"],
        region_lon2d=case_data["lon2d"], region_lat2d=case_data["lat2d"], add_box=True,
    )
    save_case_3d_figure(
        save_dir, prefix,
        context_data["lon2d"], context_data["lat2d"], volume_dbz_context,
        case_data["lon2d"], case_data["lat2d"], volume_dbz,
        heights, time_info["title_str"],
    )
    save_case_3d_volume_figure(
        save_dir, prefix,
        case_data["lon2d"], case_data["lat2d"], volume_dbz,
        heights, time_info["title_str"],
    )
    save_case_height_bin_maps(
        save_dir, prefix,
        case_data["lon2d"], case_data["lat2d"], volume_dbz,
        heights, time_info["title_str"], bin_step=15,
    )
    save_case_region_profile_figure(
        save_dir, prefix,
        case_data["lon2d"], case_data["lat2d"], volume_dbz,
        heights, time_info["title_str"],
    )
    save_case_statistics_csv(
        save_dir, prefix,
        case_data["lon2d"], case_data["lat2d"], volume_dbz,
        heights, case_data["is_land"],
    )

    print("\n[完成] Case推理与可视化结束")


if __name__ == "__main__":
    main()
