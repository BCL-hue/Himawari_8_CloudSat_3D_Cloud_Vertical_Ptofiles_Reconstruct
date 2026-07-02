"""
MLP模型测试与可视化 - AHI + 几何信息 + BT扩展指标 + ERA5廓线
测试已训练的MLP模型并生成可视化结果

输入 (与训练脚本一致):
- AHI 16通道
- 几何信息 7维 (sin/cos编码 + 地形编码)
- BT扩展指标 12维 (VCI + 11个BT相关指数)
- ERA5廓线 54维 (r, t 各27层)

输入维度: 16 + 7 + 12 + 54 = 89
"""

import os
import time
import gc
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmaps

# ============================================
# 配置区域
# ============================================

# 数据路径
DATA_ROOT = "/mnt/g/3D_Cloud_Reflectivety_Profile_Projection/3D_Train_data/point"
CHECKPOINT_DIR = "/mnt/g/3D_Cloud_Reflectivety_Profile_Projection/Point_projection/CloudSat_GEOPROF/L_20260702_code/MLP/checkpoints"
SAVE_DIR = "/mnt/g/3D_Cloud_Reflectivety_Profile_Projection/Point_projection/CloudSat_GEOPROF/L_20260702_code/MLP/results"

# 测试参数
BATCH_SIZE = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型参数 (与训练脚本一致)
IN_DIM = 16 + 7 + 12 + 2 * 27  # AHI + geo + BT扩展 + ERA5(r,t各27层) = 89
OUT_DIM = 85  # CloudSat反射率层数 (索引20:105)
DROPOUT = 0.2

# ERA5气压层索引映射
PRESSURE_LEVELS = [
    1000,
    975,
    950,
    925,
    900,
    875,
    850,  # 0-6
    825,
    800,
    775,
    750,
    700,
    650,
    600,  # 7-13 (低层14层)
    550,
    500,
    450,
    400,
    350,
    300,
    250,  # 14-20
    225,
    200,
    175,
    150,
    125,
    100,  # 21-26 (高层13层)
]

# ERA5变量在profile中的索引
# era5_profile_low/high: [N, 6, 14/13] -> 变量顺序: r(0), u(1), v(2), w(3), q(4), t(5)
ERA5_R_IDX = 0
ERA5_T_IDX = 5

# AHI 16通道的均值和标准差
CHANNEL_MEANS = np.array(
    [
        # 可见光通道 (albedo)
        0.37,
        0.35,
        0.34,
        0.38,
        0.20,
        0.17,
        # 红外通道 (tbb)
        280.54,
        231.33,
        238.30,
        244.07,
        259.59,
        247.45,
        260.61,
        259.00,
        256.71,
        249.12,
    ]
)

CHANNEL_STDS = np.array(
    [
        # 可见光通道
        0.23,
        0.23,
        0.23,
        0.25,
        0.12,
        0.10,
        # 红外通道
        17.07,
        9.31,
        11.94,
        14.20,
        22.25,
        14.30,
        22.95,
        23.18,
        22.34,
        18.16,
    ]
)

# 12个BT扩展指标的统计量（基于全部训练数据151254样本计算）
BT_DIFFS_EXTENDED_MEANS = np.array(
    [
        10.0274,  # VCI
        21.5406,  # BT_7_14
        -12.7420,  # BT_8_10
        -6.9704,  # BT_8_9 (6.2-6.9μm)
        -5.7717,  # BT_9_10 (6.9-7.3μm)
        2.2957,  # BT_14_15
        3.9038,  # BT_13_15
        -256.1171,  # BT_11_14_15
        -11.4865,  # BT_16_13
        0.2382,  # Ratio_3_5
        0.0270,  # Albedo_5_6
        36.8777,  # Tskin_BT13
    ]
)

BT_DIFFS_EXTENDED_STDS = np.array(
    [
        6.7557,  # VCI
        11.6445,  # BT_7_14
        6.0438,  # BT_8_10
        3.1137,  # BT_8_9 (6.2-6.9μm)
        3.1410,  # BT_9_10 (6.9-7.3μm)
        1.7798,  # BT_14_15
        2.7126,  # BT_13_15
        23.5625,  # BT_11_14_15
        6.0939,  # BT_16_13
        0.2806,  # Ratio_3_5
        0.0602,  # Albedo_5_6
        24.0085,  # Tskin_BT13
    ]
)

# ============================================
# ERA5廓线数据统计量 (仅r, t)
# ============================================

# 相对湿度 - 低层 (14层)
ERA5_RH_LOW_MEANS = np.array(
    [
        73.5363,
        76.9706,
        78.5329,
        77.5105,
        76.3599,
        75.4709,
        74.5204,
        73.5544,
        72.4292,
        71.1167,
        69.6088,
        66.1056,
        63.3191,
        62.3721,
    ]
)
ERA5_RH_LOW_STDS = np.array(
    [
        16.8317,
        18.5724,
        19.2897,
        19.0874,
        19.2325,
        19.8277,
        20.6616,
        21.6271,
        22.4810,
        23.3625,
        24.1576,
        25.1004,
        26.0628,
        28.3531,
    ]
)

# 温度 - 低层 (14层)
ERA5_T_LOW_MEANS = np.array(
    [
        296.62,
        294.93,
        293.40,
        292.10,
        290.83,
        289.56,
        288.31,
        287.05,
        285.79,
        284.50,
        283.18,
        280.35,
        277.15,
        273.54,
    ]
)
ERA5_T_LOW_STDS = np.array(
    [
        7.13,
        7.07,
        7.07,
        7.10,
        7.08,
        7.01,
        6.90,
        6.78,
        6.65,
        6.51,
        6.38,
        6.16,
        5.96,
        5.73,
    ]
)

# 相对湿度 - 高层 (13层)
ERA5_RH_HIGH_MEANS = np.array(
    [
        60.66,
        57.94,
        56.65,
        57.42,
        59.29,
        61.07,
        63.14,
        63.79,
        63.20,
        60.86,
        57.69,
        56.23,
        57.08,
    ]
)
ERA5_RH_HIGH_STDS = np.array(
    [
        30.45,
        30.36,
        30.54,
        31.30,
        32.08,
        32.54,
        33.60,
        34.76,
        36.12,
        37.45,
        38.38,
        38.82,
        39.87,
    ]
)

# 温度 - 高层 (13层)
ERA5_T_HIGH_MEANS = np.array(
    [
        269.95,
        265.89,
        261.15,
        255.59,
        248.98,
        241.10,
        231.86,
        226.75,
        221.27,
        215.39,
        209.16,
        202.83,
        197.97,
    ]
)
ERA5_T_HIGH_STDS = np.array(
    [
        5.72,
        5.89,
        6.11,
        6.32,
        6.41,
        6.15,
        5.07,
        4.12,
        3.14,
        2.93,
        3.97,
        5.86,
        7.31,
    ]
)

# CloudSat反射率索引范围
REFLECTIVITY_IDX_START = 20
REFLECTIVITY_IDX_END = 105


# ============================================
# 数据处理函数
# ============================================


def apply_strategy(Geoprof_filtered):
    """根据策略处理反射率数据"""
    data = Geoprof_filtered.copy()

    data[data < -90.0] = -35.0
    data[data < -35.0] = -35.0
    data[data > 20.0] = 20.0
    normalized = 2.0 * ((data + 35.0) / 55.0) - 1.0

    return normalized


def denormalize_reflectivity(ref_norm):
    """Denormalize reflectivity values"""
    return (ref_norm + 1.0) * 55.0 / 2.0 - 35.0


# ============================================
# 数据集类 (与训练脚本一致)
# ============================================


class CloudSatDataset(Dataset):
    """
    数据集 - AHI 16通道 + 几何信息 7维 + BT扩展指标 12维 + ERA5廓线(r,t各27层)

    输入维度: 16 + 7 + 12 + 54 = 89
    """

    def __init__(self, data_path: str, split: str = "train", preload: bool = True):
        self.split = split
        self.preload = preload
        self.h5_path = os.path.join(
            data_path, split, f"point_{split}_filtered.h5"
        )

        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"数据文件不存在: {self.h5_path}")

        self.h5_file = h5py.File(self.h5_path, "r")
        self.num_samples = self.h5_file["ahi"].shape[0]

        print(f"[{split}集] 样本数: {self.num_samples}")

        self.memory_data = None
        if preload:
            self._preload_to_memory()

    def _normalize_geometry(self, geo):
        """
        归一化几何信息 - 使用sin/cos周期编码 + 精细地形编码
        geo: [lat, lon, SAA, SAZ, SOZ, SOA, terrain_elevation]
        返回: [sin(lon), cos(lon), sin(lat), cos(lat), land_height, ocean_depth, is_land] (7维)
        """
        lat = geo[0]
        lon = geo[1]
        terrain = geo[6]

        # sin/cos encoding for lat/lon (4 dimensions)
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        lon_lat_enc = np.array(
            [
                np.sin(lon_rad),
                np.cos(lon_rad),
                np.sin(lat_rad),
                np.cos(lat_rad),
            ],
            dtype=np.float32,
        )

        # 地形精细编码 (3 dimensions)
        is_land = 1.0 if terrain >= 0 else 0.0  # 陆地掩码

        if terrain >= 0:
            # 陆地: 归一化到 [0,1], 按5500m为最大值(留余量)
            land_height = min(terrain / 5500.0, 1.0)
            ocean_depth = 0.0
        else:
            # 海洋: 深度归一化到 [0,1], 按10000m为最大值
            land_height = 0.0
            ocean_depth = min(-terrain / 10000.0, 1.0)

        terrain_enc = np.array(
            [land_height, ocean_depth, is_land], dtype=np.float32
        )

        return np.concatenate([lon_lat_enc, terrain_enc])

    def _compute_extended_bt_features(self, ahi, t_skin):
        """
        计算12个BT扩展指标
        ahi: [N, 16] 原始AHI数据 (未归一化)
        t_skin: [N] 地表温度 (SKT)

        返回: [N, 12] 12个指标
        """
        # 提取各通道数据
        a01 = ahi[:, 0]  # albedo_01
        a02 = ahi[:, 1]  # albedo_02
        a03 = ahi[:, 2]  # albedo_03
        a04 = ahi[:, 3]  # albedo_04
        a05 = ahi[:, 4]  # albedo_05
        a06 = ahi[:, 5]  # albedo_06

        bt07 = ahi[:, 6]  # tbb_07
        bt08 = ahi[:, 7]  # tbb_08
        bt09 = ahi[:, 8]  # tbb_09
        bt10 = ahi[:, 9]  # tbb_10
        bt11 = ahi[:, 10]  # tbb_11
        bt12 = ahi[:, 11]  # tbb_12
        bt13 = ahi[:, 12]  # tbb_13
        bt14 = ahi[:, 13]  # tbb_14
        bt15 = ahi[:, 14]  # tbb_15
        bt16 = ahi[:, 15]  # tbb_16

        eps = 1e-6

        # 1. VCI (云检测)
        vci = 255 * np.sqrt(
            ((a01 - a03) ** 2 + (a01 - a04) ** 2 + (a03 - a04) ** 2) / 3
        )

        # 2. BT_7_14 (3.9-11.2μm 低云/雾)
        bt_7_14 = bt07 - bt14

        # 3. BT_8_10 (6.2-7.3μm 云顶高度)
        bt_8_10 = bt08 - bt10

        # 4. BT_8_9 (6.2-6.9μm 中上层水汽)
        bt_8_9 = bt08 - bt09

        # 5. BT_9_10 (6.9-7.3μm 中下层水汽)
        bt_9_10 = bt09 - bt10

        # 6. BT_14_15 (11.2-12.4μm 水汽/云光学厚度)
        bt_14_15 = bt14 - bt15

        # 7. BT_13_15 (10.4-12.4μm 卷云检测)
        bt_13_15 = bt13 - bt15

        # 8. BT_11_14_15 (8.6-11.2-12.4μm 对流云)
        bt_11_14_15 = bt11 - bt14 - bt15

        # 9. BT_16_13 (13.3-10.4μm 云顶高度)
        bt_16_13 = bt16 - bt13

        # 10. Ratio_3_5 (0.64/1.6μm 粒径判别)
        ratio_3_5 = (a03 - a05) / (a03 + a05 + eps)

        # 11. Albedo_5_6 (1.6-2.25μm 相态识别)
        albedo_5_6 = a05 - a06

        # 12. Tskin_BT13 (地表温度 - 云顶亮温)
        tskin_bt13 = t_skin - bt13

        # 拼接所有12个特征
        features = np.stack(
            [
                vci,
                bt_7_14,
                bt_8_10,
                bt_8_9,
                bt_9_10,
                bt_14_15,
                bt_13_15,
                bt_11_14_15,
                bt_16_13,
                ratio_3_5,
                albedo_5_6,
                tskin_bt13,
            ],
            axis=1,
        )

        return features  # [N, 12]

    def _normalize_extended_features(self, features):
        """归一化12个BT扩展指标"""
        return (features - BT_DIFFS_EXTENDED_MEANS) / BT_DIFFS_EXTENDED_STDS

    def _extract_era5_rt_profile(self, era5_profile_low, era5_profile_high):
        """
        提取ERA5的r和t变量，合并低层+高层全27层

        Args:
            era5_profile_low: [N, 6, 14]
            era5_profile_high: [N, 6, 13]

        Returns:
            [N, 54] r(27层) + t(27层)
        """
        era5_full = np.concatenate(
            [era5_profile_low, era5_profile_high], axis=2
        )  # [N, 6, 27]

        r = era5_full[:, ERA5_R_IDX, :]  # [N, 27]
        t = era5_full[:, ERA5_T_IDX, :]  # [N, 27]

        return np.concatenate([r, t], axis=1)  # [N, 54]

    def _normalize_era5_rt_profile(self, era5_profile):
        """
        归一化ERA5 r和t廓线 (逐层归一化)

        Args:
            era5_profile: [N, 54] 前27层为r，后27层为t

        Returns:
            [N, 54]
        """
        normalized = era5_profile.copy()

        # r: 前27层 (低层14 + 高层13)
        for i in range(14):
            normalized[:, i] = (
                era5_profile[:, i] - ERA5_RH_LOW_MEANS[i]
            ) / ERA5_RH_LOW_STDS[i]
        for i in range(13):
            normalized[:, 14 + i] = (
                era5_profile[:, 14 + i] - ERA5_RH_HIGH_MEANS[i]
            ) / ERA5_RH_HIGH_STDS[i]

        # t: 后27层 (偏移27)
        for i in range(14):
            normalized[:, 27 + i] = (
                era5_profile[:, 27 + i] - ERA5_T_LOW_MEANS[i]
            ) / ERA5_T_LOW_STDS[i]
        for i in range(13):
            normalized[:, 27 + 14 + i] = (
                era5_profile[:, 27 + 14 + i] - ERA5_T_HIGH_MEANS[i]
            ) / ERA5_T_HIGH_STDS[i]

        return normalized

    def _preload_to_memory(self):
        """预加载数据到内存"""
        start = time.time()

        # AHI原始数据
        ahi_all = self.h5_file["ahi"][:]
        ahi_normalized = (ahi_all - CHANNEL_MEANS) / CHANNEL_STDS

        # 1. 几何信息
        geo_all = self.h5_file["lat_lon_angle"][:]
        geo_normalized = np.array(
            [self._normalize_geometry(g) for g in geo_all]
        )

        # 2. BT扩展指标
        era5_2d = self.h5_file["era5_2d"][:]
        t_skin = era5_2d[:, 5]  # SKT
        extended_features = self._compute_extended_bt_features(ahi_all, t_skin)
        extended_features_norm = self._normalize_extended_features(
            extended_features
        )

        # 3. ERA5廓线 (r, t, 全27层)
        era5_low = self.h5_file["era5_profile_low"][:]
        era5_high = self.h5_file["era5_profile_high"][:]
        era5_profile = self._extract_era5_rt_profile(era5_low, era5_high)
        era5_profile_norm = self._normalize_era5_rt_profile(era5_profile)

        # 拼接所有输入组件: AHI(16) + geo(7) + BT扩展(12) + ERA5(54)
        inputs = np.concatenate(
            [ahi_normalized, geo_normalized, extended_features_norm, era5_profile_norm],
            axis=1,
        )

        # 标签
        label_raw = self.h5_file["label_georef_max"][:] / 100.0
        reflectivity = label_raw[:, REFLECTIVITY_IDX_START:REFLECTIVITY_IDX_END]

        cmask_raw = self.h5_file["label_Cmask_max"][:]
        cmask = cmask_raw[:, REFLECTIVITY_IDX_START:REFLECTIVITY_IDX_END]

        reflectivity = apply_strategy(reflectivity)

        self.h5_file.close()
        self.h5_file = None

        self.memory_data = {
            "inputs": inputs.astype(np.float32),
            "reflectivity": reflectivity.astype(np.float32),
            "cmask": cmask.astype(np.float32),
        }

        elapsed = time.time() - start
        data_size_mb = (
            sum(v.nbytes for v in self.memory_data.values()) / 1024 / 1024
        )
        print(f"  预加载完成: {elapsed:.2f}秒, {data_size_mb:.1f}MB")
        print(f"  输入维度: {inputs.shape[1]}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.memory_data is not None:
            inputs = self.memory_data["inputs"][idx]
            reflectivity = self.memory_data["reflectivity"][idx]
        else:
            # 单样本逐条读取 (未预加载时)
            ahi = self.h5_file["ahi"][idx]
            ahi_norm = (ahi - CHANNEL_MEANS) / CHANNEL_STDS

            geo = self.h5_file["lat_lon_angle"][idx]
            geo_norm = self._normalize_geometry(geo)

            ahi_all = ahi[np.newaxis, :]
            era5_2d = self.h5_file["era5_2d"][idx]
            t_skin = era5_2d[5]
            extended_features = self._compute_extended_bt_features(
                ahi_all, t_skin[np.newaxis, :]
            )[0]
            extended_features_norm = self._normalize_extended_features(
                extended_features[np.newaxis, :]
            )[0]

            era5_low = self.h5_file["era5_profile_low"][idx]  # [6, 14]
            era5_high = self.h5_file["era5_profile_high"][idx]  # [6, 13]
            era5_profile = self._extract_era5_rt_profile(
                era5_low[np.newaxis, :], era5_high[np.newaxis, :]
            )[0]
            era5_profile_norm = self._normalize_era5_rt_profile(
                era5_profile[np.newaxis, :]
            )[0]

            inputs = np.concatenate(
                [ahi_norm, geo_norm, extended_features_norm, era5_profile_norm]
            ).astype(np.float32)

            reflectivity_raw = self.h5_file["label_georef_max"][idx] / 100.0
            reflectivity = reflectivity_raw[
                REFLECTIVITY_IDX_START:REFLECTIVITY_IDX_END
            ]
            reflectivity = apply_strategy(reflectivity)

        return {
            "input": torch.from_numpy(inputs).float(),
            "reflectivity": torch.from_numpy(reflectivity).float(),
        }

    def close(self):
        if hasattr(self, "h5_file") and self.h5_file is not None:
            self.h5_file.close()

    def __del__(self):
        self.close()


# ============================================
# 模型类
# ============================================


class SimpleMLP(torch.nn.Module):
    """简单MLP模型"""

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
        output = self.output_layer(x)
        return output


# ============================================
# 绘图函数
# ============================================


def setup_paper_style():
    """设置学术论文风格的绘图参数"""
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


def plot_cross_section(ax, data, n_samples, title, vmin, vmax, cmap):
    """绘制单个剖面图"""
    n_levels = data.shape[0]
    x_edges = np.linspace(0, n_samples, n_samples + 1)
    y_edges = np.linspace(20.4, 0, n_levels + 1)

    im = ax.pcolormesh(x_edges, y_edges, data, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel("Sample Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Height (km)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 20)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, aspect=25)
    cbar.set_label("Reflectivity (dBZ)", fontsize=11, fontweight="bold")

    return im


def plot_difference_section(ax, diff_data, n_samples, title):
    """绘制差异剖面图"""
    n_levels = diff_data.shape[0]
    x_edges = np.linspace(0, n_samples, n_samples + 1)
    y_edges = np.linspace(20.4, 0, n_levels + 1)

    im = ax.pcolormesh(
        x_edges, y_edges, diff_data, cmap="RdBu_r", vmin=-20, vmax=20
    )

    ax.set_xlabel("Sample Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Height (km)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 20)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, aspect=25)
    cbar.set_label("Difference (dBZ)", fontsize=11, fontweight="bold")

    return im


def plot_reflectivity_and_peak_height_distributions(
    fig, preds_dbz, trues_dbz, heights
):
    r"""
    绘制反射率分布和峰值高度分布图

    Args:
        fig: matplotlib figure对象
        preds_dbz: 预测反射率，形状 (N, 85)
        trues_dbz: 真实反射率，形状 (N, 85)
        heights: 高度值 (km)，形状 (85,)
    """
    # 创建子图
    if len(fig.axes) < 2:
        fig.clf()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        ax1, ax2 = fig.axes[0], fig.axes[1]

    # ============================================
    # 子图1: 反射率因子分布（log坐标）
    # ============================================
    pred_flat = preds_dbz.flatten()
    true_flat = trues_dbz.flatten()

    bins = np.linspace(-35, 25, 21)
    pred_hist, _ = np.histogram(pred_flat, bins=bins)
    true_hist, _ = np.histogram(true_flat, bins=bins)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    gray_region = bin_centers < -15
    ax1.fill_between(
        bin_centers[gray_region],
        0,
        np.maximum(pred_hist[gray_region], true_hist[gray_region]),
        color="lightgray",
        alpha=0.5,
        label="< -15 dBZ (Clear air)",
    )

    ax1.plot(
        bin_centers,
        pred_hist,
        "r-",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Predicted",
        alpha=0.8,
    )
    ax1.plot(
        bin_centers,
        true_hist,
        "b--",
        linewidth=2,
        marker="s",
        markersize=4,
        label="True",
        alpha=0.8,
    )

    ax1.fill_between(
        bin_centers,
        pred_hist,
        true_hist,
        color="purple",
        alpha=0.15,
        label="Difference",
    )

    ax1.set_yscale("log")
    ax1.set_xlabel("Reflectivity (dBZ)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Frequency (log scale)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "(a) Reflectivity Distribution",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax1.set_xlim(-35, 25)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(fontsize=10)
    ax1.axvline(x=-15, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

    # ============================================
    # 子图2: 峰值高度分布
    # ============================================
    pred_peak_heights = []
    true_peak_heights = []

    for i in range(preds_dbz.shape[0]):
        pred_profile = preds_dbz[i]
        true_profile = trues_dbz[i]

        pred_valid = pred_profile > -30
        true_valid = true_profile > -30

        if pred_valid.any():
            pred_peak_idx = np.argmax(pred_profile)
            pred_peak_heights.append(heights[pred_peak_idx])
        else:
            pred_peak_heights.append(np.nan)

        if true_valid.any():
            true_peak_idx = np.argmax(true_profile)
            true_peak_heights.append(heights[true_peak_idx])
        else:
            true_peak_heights.append(np.nan)

    pred_peak_heights = np.array(pred_peak_heights)
    true_peak_heights = np.array(true_peak_heights)

    height_bins = np.linspace(0, 20.4, 41)

    pred_peak_hist, _ = np.histogram(
        pred_peak_heights[~np.isnan(pred_peak_heights)], bins=height_bins
    )
    true_peak_hist, _ = np.histogram(
        true_peak_heights[~np.isnan(true_peak_heights)], bins=height_bins
    )

    height_bin_centers = (height_bins[:-1] + height_bins[1:]) / 2

    ax2.plot(
        height_bin_centers,
        pred_peak_hist,
        "r-",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Predicted",
        alpha=0.8,
    )
    ax2.plot(
        height_bin_centers,
        true_peak_hist,
        "b--",
        linewidth=2,
        marker="s",
        markersize=4,
        label="True",
        alpha=0.8,
    )

    ax2.fill_between(
        height_bin_centers,
        pred_peak_hist,
        true_peak_hist,
        color="purple",
        alpha=0.15,
        label="Difference",
    )

    ax2.set_xlabel("Peak Height (km)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax2.set_title(
        "(b) Peak Height Distribution",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax2.set_xlim(0, 20.4)
    ax2.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax2.legend(fontsize=10)

    pred_peak_mean = np.nanmean(pred_peak_heights)
    true_peak_mean = np.nanmean(true_peak_heights)
    pred_peak_median = np.nanmedian(pred_peak_heights)
    true_peak_median = np.nanmedian(true_peak_heights)

    stats_text = (
        f"Predicted: Mean={pred_peak_mean:.2f}km, Med={pred_peak_median:.2f}km\n"
        f"True: Mean={true_peak_mean:.2f}km, Med={true_peak_median:.2f}km"
    )
    ax2.text(
        0.98,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"
        ),
    )


def plot_profile_metrics(ax1, ax2, ax3, preds_dbz, trues_dbz, heights):
    """绘制垂直廓线评估指标"""
    # RMSE by height
    rmse_by_height = np.sqrt(np.mean((preds_dbz - trues_dbz) ** 2, axis=0))
    rmse_mean = np.mean(rmse_by_height)
    rmse_max_idx = np.argmax(rmse_by_height)
    rmse_max = rmse_by_height[rmse_max_idx]
    rmse_max_height = heights[rmse_max_idx]

    ax1.plot(
        rmse_by_height,
        heights,
        color="#2E86AB",
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=5,
    )
    ax1.fill_betweenx(heights, 0, rmse_by_height, alpha=0.25, color="#2E86AB")
    ax1.set_xlabel("RMSE (dBZ)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Height (km)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "(a) RMSE vs Height", fontsize=13, fontweight="bold", loc="left"
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(0, rmse_by_height.max() * 1.1)
    ax1.axvline(
        rmse_mean, color="#E07A5F", linestyle="--", linewidth=1.8, alpha=0.9
    )
    ax1.scatter(rmse_max, rmse_max_height, color="#C73E1D", s=36, zorder=5)
    textstr = f"Mean = {rmse_mean:.2f} dBZ\nMax = {rmse_max:.2f} dBZ"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax1.text(
        0.95,
        0.95,
        textstr,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=props,
    )

    # Bias by height
    bias_by_height = np.mean(preds_dbz - trues_dbz, axis=0)
    bias_mean = np.mean(bias_by_height)
    bias_min = np.min(bias_by_height)
    bias_max = np.max(bias_by_height)
    bias_min_height = heights[np.argmin(bias_by_height)]
    bias_max_height = heights[np.argmax(bias_by_height)]

    ax2.plot(
        bias_by_height,
        heights,
        color="#A23B72",
        linewidth=2.5,
        marker="s",
        markersize=4,
        markevery=5,
    )
    ax2.axvline(x=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.fill_betweenx(heights, 0, bias_by_height, alpha=0.25, color="#A23B72")
    ax2.set_xlabel("Bias (dBZ)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Height (km)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "(b) Bias vs Height", fontsize=13, fontweight="bold", loc="left"
    )
    ax2.grid(True, alpha=0.3, linestyle="--")
    bias_text = f"Mean = {bias_mean:.2f} dBZ\nMin = {bias_min:.2f} dBZ\nMax = {bias_max:.2f} dBZ"
    ax2.text(
        0.05,
        0.95,
        bias_text,
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Correlation by height
    corr_by_height = []
    for i in range(85):
        if np.std(trues_dbz[:, i]) > 0.01:
            corr = np.corrcoef(preds_dbz[:, i], trues_dbz[:, i])[0, 1]
        else:
            corr = 0
        corr_by_height.append(corr)

    corr_mean = np.mean(corr_by_height)
    corr_min = np.min(corr_by_height)
    corr_max = np.max(corr_by_height)
    corr_min_height = heights[np.argmin(corr_by_height)]
    corr_max_height = heights[np.argmax(corr_by_height)]

    ax3.plot(
        corr_by_height,
        heights,
        color="#F18F01",
        linewidth=2.5,
        marker="^",
        markersize=4,
        markevery=5,
    )
    ax3.set_xlabel("Correlation", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Height (km)", fontsize=12, fontweight="bold")
    ax3.set_title(
        "(c) Correlation vs Height", fontsize=13, fontweight="bold", loc="left"
    )
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.set_xlim(0.4, 1.0)
    corr_text = f"Mean = {corr_mean:.3f}\nMax = {corr_max:.3f}"
    ax3.text(
        0.05,
        0.95,
        corr_text,
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def plot_scatter_comparison(ax, preds, trues):
    """绘制散点对比图"""
    p_flat = preds.flatten()
    t_flat = trues.flatten()

    ax.scatter(
        t_flat, p_flat, alpha=0.1, s=5, color="#2E86AB", edgecolors="none"
    )

    min_val = min(t_flat.min(), p_flat.min())
    max_val = max(t_flat.max(), p_flat.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", linewidth=2, alpha=0.7
    )

    r2 = np.corrcoef(t_flat, p_flat)[0, 1] ** 2
    rmse = np.sqrt(np.mean((p_flat - t_flat) ** 2))
    bias = np.mean(p_flat - t_flat)

    ax.set_xlabel("True Reflectivity (dBZ)", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Predicted Reflectivity (dBZ)", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        f"(d) Scatter (N={len(p_flat):,})",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    textstr = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} dBZ\nBias = {bias:.2f} dBZ"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )


def plot_multi_level_scatter(fig, preds, trues):
    """绘制高中低三层数点对比图"""
    level_ranges = [
        (0, 40, "High Level", "Bins 0-40 (~20-10.5km)"),
        (40, 70, "Mid Level", "Bins 40-70 (~10.5-3.3km)"),
        (70, 85, "Low Level", "Bins 70-84 (~3.3-0km)"),
    ]

    for idx, (start_bin, end_bin, level_name, height_str) in enumerate(
        level_ranges
    ):
        ax = fig.add_subplot(1, 3, idx + 1)

        p = preds[:, start_bin:end_bin].flatten()
        t = trues[:, start_bin:end_bin].flatten()

        ax.scatter(t, p, alpha=0.1, s=5, color="#2E86AB", edgecolors="none")

        min_val = min(t.min(), p.min())
        max_val = max(t.max(), p.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            alpha=0.7,
        )

        r2 = np.corrcoef(t, p)[0, 1] ** 2
        rmse = np.sqrt(np.mean((p - t) ** 2))
        bias = np.mean(p - t)

        ax.set_xlabel("True (dBZ)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Pred (dBZ)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{level_name}\n{height_str}\n$R^2$={r2:.3f}, RMSE={rmse:.2f}, Bias={bias:.2f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(-35, 20)
        ax.set_ylim(-35, 20)


def compute_profile_metrics(preds_dbz, trues_dbz):
    """逐个廓线计算RMSE和相关系数（只对真实值>-25dBZ的数据点计算）"""
    n_profiles = preds_dbz.shape[0]
    rmse_profiles = np.zeros(n_profiles, dtype=np.float32)
    corr_profiles = np.zeros(n_profiles, dtype=np.float32)
    valid_counts = np.zeros(n_profiles, dtype=np.int32)

    for i in range(n_profiles):
        p = preds_dbz[i]
        t = trues_dbz[i]
        valid = t > -25
        valid_counts[i] = np.sum(valid)

        if valid_counts[i] >= 2:
            p_valid = p[valid]
            t_valid = t[valid]
            rmse_profiles[i] = np.sqrt(np.mean((p_valid - t_valid) ** 2))

            if np.std(p_valid) > 1e-6 and np.std(t_valid) > 1e-6:
                corr_profiles[i] = np.corrcoef(p_valid, t_valid)[0, 1]
            else:
                corr_profiles[i] = np.nan
        else:
            rmse_profiles[i] = np.nan
            corr_profiles[i] = np.nan

    return rmse_profiles, corr_profiles, valid_counts


def compute_gradient_rmse_profiles_constrained(preds_dbz, trues_dbz):
    """
    逐个廓线计算梯度RMSE（带约束条件：只对真实值>-25dBZ的数据点计算）
    """
    n_profiles = preds_dbz.shape[0]
    grad_rmse_profiles = np.zeros(n_profiles, dtype=np.float32)
    valid_counts = np.zeros(n_profiles, dtype=np.int32)

    for i in range(n_profiles):
        p = preds_dbz[i]
        t = trues_dbz[i]
        valid = t > -25
        valid_counts[i] = np.sum(valid)

        if valid_counts[i] >= 3:
            p_valid = p[valid]
            t_valid = t[valid]
            pred_grad = np.diff(p_valid)
            true_grad = np.diff(t_valid)
            grad_rmse_profiles[i] = np.sqrt(
                np.mean((pred_grad - true_grad) ** 2)
            )
        else:
            grad_rmse_profiles[i] = np.nan

    return grad_rmse_profiles, valid_counts


def compute_peak_height_error_constrained(preds_dbz, trues_dbz, heights):
    """
    逐个廓线计算峰值高度误差（带约束条件：只对真实值>-25dBZ的数据点计算）
    """
    n_profiles = preds_dbz.shape[0]
    peak_errors = np.zeros(n_profiles, dtype=np.float32)
    pred_peaks_idx = np.zeros(n_profiles, dtype=np.int32)
    true_peaks_idx = np.zeros(n_profiles, dtype=np.int32)
    valid_counts = np.zeros(n_profiles, dtype=np.int32)

    for i in range(n_profiles):
        p = preds_dbz[i]
        t = trues_dbz[i]
        valid = t > -25
        valid_counts[i] = np.sum(valid)

        if valid_counts[i] >= 2:
            valid_indices = np.where(valid)[0]
            p_valid_values = p[valid]
            t_valid_values = t[valid]

            pred_peaks_idx[i] = valid_indices[np.argmax(p_valid_values)]
            true_peaks_idx[i] = valid_indices[np.argmax(t_valid_values)]

            peak_errors[i] = (
                heights[true_peaks_idx[i]] - heights[pred_peaks_idx[i]]
            )
        else:
            peak_errors[i] = np.nan
            pred_peaks_idx[i] = 0
            true_peaks_idx[i] = 0

    return peak_errors, pred_peaks_idx, true_peaks_idx, valid_counts


def compute_gradient_rmse(preds_dbz, trues_dbz):
    """
    计算梯度RMSE：先对廓线沿高度差分，再计算差分序列的RMSE
    """
    pred_grad = np.diff(preds_dbz, axis=1)
    true_grad = np.diff(trues_dbz, axis=1)

    grad_rmse_by_height = np.sqrt(np.mean((pred_grad - true_grad) ** 2, axis=0))
    grad_rmse_mean = np.mean(grad_rmse_by_height)

    return grad_rmse_by_height, grad_rmse_mean


def compute_gradient_rmse_profiles(preds_dbz, trues_dbz):
    """计算每条廓线的梯度RMSE"""
    n_profiles = preds_dbz.shape[0]
    grad_rmse_profiles = np.zeros(n_profiles, dtype=np.float32)

    pred_grad = np.diff(preds_dbz, axis=1)
    true_grad = np.diff(trues_dbz, axis=1)

    for i in range(n_profiles):
        grad_rmse_profiles[i] = np.sqrt(
            np.mean((pred_grad[i] - true_grad[i]) ** 2)
        )

    return grad_rmse_profiles


def compute_hist_peak(values, bins=50, value_range=None):
    """返回直方图最高频bin的中心值和计数"""
    counts, bin_edges = np.histogram(values, bins=bins, range=value_range)
    peak_idx = np.argmax(counts)
    peak_center = 0.5 * (bin_edges[peak_idx] + bin_edges[peak_idx + 1])
    peak_count = counts[peak_idx]
    return peak_center, peak_count


def compute_kde_curve(values, n_grid=400, value_range=None):
    """使用高斯核估计一维概率密度曲线"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.array([]), np.array([]), np.nan

    std = np.std(values, ddof=1)
    if std < 1e-8:
        std = 1e-3

    bandwidth = 1.06 * std * (len(values) ** (-1 / 5))
    bandwidth = max(bandwidth, 1e-3)

    if value_range is None:
        data_min = values.min()
        data_max = values.max()
        padding = 0.08 * (data_max - data_min + 1e-6)
        x_grid = np.linspace(data_min - padding, data_max + padding, n_grid)
    else:
        x_grid = np.linspace(value_range[0], value_range[1], n_grid)

    diff = (x_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * diff**2).sum(axis=1)
    density /= len(values) * bandwidth * np.sqrt(2 * np.pi)

    mode_x = x_grid[np.argmax(density)]
    return x_grid, density, mode_x


def plot_profile_metric_distributions(
    ax1,
    ax2,
    ax3,
    ax4,
    rmse_profiles,
    corr_profiles,
    peak_errors,
    grad_rmse_profiles,
):
    """绘制逐廓线RMSE、相关系数、峰值高度误差和梯度RMSE分布图"""
    rmse_valid = rmse_profiles[np.isfinite(rmse_profiles)]
    corr_valid = corr_profiles[np.isfinite(corr_profiles)]
    mean_rmse_profile = np.mean(rmse_valid)

    # (a) RMSE Distribution
    rmse_counts, _, _ = ax1.hist(
        rmse_valid,
        bins=50,
        color="#2E86AB",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.6,
    )
    rmse_peak_x, rmse_peak_count = compute_hist_peak(rmse_valid, bins=50)
    ax1.axvline(
        np.mean(rmse_valid), color="#C73E1D", linestyle="--", linewidth=2
    )
    ax1.axvline(
        np.median(rmse_valid), color="#5B8E7D", linestyle=":", linewidth=2
    )
    ax1.axvline(rmse_peak_x, color="#6A4C93", linestyle="-.", linewidth=2)
    ax1.scatter(
        [rmse_peak_x],
        [rmse_peak_count],
        color="#6A4C93",
        s=40,
        zorder=4,
        edgecolors="white",
    )
    ax1.set_xlabel("Per-profile RMSE (dBZ)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax1.set_title(
        "(a) RMSE Distribution", fontsize=13, fontweight="bold", loc="left"
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.text(
        0.97,
        0.97,
        f"N = {len(rmse_valid):,}\nAverage RMSE = {mean_rmse_profile:.2f} dBZ\nMedian = {np.median(rmse_valid):.2f}\nPeak = {rmse_peak_x:.2f}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"
        ),
    )

    # (b) Correlation Distribution
    corr_counts, _, _ = ax2.hist(
        corr_valid,
        bins=50,
        color="#F18F01",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.6,
    )
    corr_peak_x, corr_peak_count = compute_hist_peak(
        corr_valid, bins=50, value_range=(-1, 1)
    )
    ax2.axvline(
        np.mean(corr_valid), color="#C73E1D", linestyle="--", linewidth=2
    )
    ax2.axvline(
        np.median(corr_valid), color="#5B8E7D", linestyle=":", linewidth=2
    )
    ax2.axvline(corr_peak_x, color="#6A4C93", linestyle="-.", linewidth=2)
    ax2.scatter(
        [corr_peak_x],
        [corr_peak_count],
        color="#6A4C93",
        s=40,
        zorder=4,
        edgecolors="white",
    )
    ax2.set_xlabel("Per-profile Correlation", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax2.set_title(
        "(b) Correlation Distribution",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xlim(-1.0, 1.0)
    ax2.text(
        0.24,
        0.97,
        f"N = {len(corr_valid):,}\nMean = {np.mean(corr_valid):.3f}\nMedian = {np.median(corr_valid):.3f}\nPeak = {corr_peak_x:.3f}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"
        ),
    )

    # (c) Peak Height Error Distribution
    peak_valid = peak_errors[np.isfinite(peak_errors)]
    peak_counts, _, _ = ax3.hist(
        peak_valid,
        bins=50,
        color="#009E73",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.6,
    )
    peak_peak_x, peak_peak_count = compute_hist_peak(peak_valid, bins=50)
    ax3.axvline(
        np.mean(peak_valid), color="#C73E1D", linestyle="--", linewidth=2
    )
    ax3.axvline(
        np.median(peak_valid), color="#5B8E7D", linestyle=":", linewidth=2
    )
    ax3.axvline(
        0, color="gray", linestyle="-", linewidth=1.2, alpha=0.6
    )
    ax3.axvline(peak_peak_x, color="#6A4C93", linestyle="-.", linewidth=2)
    ax3.scatter(
        [peak_peak_x],
        [peak_peak_count],
        color="#6A4C93",
        s=40,
        zorder=4,
        edgecolors="white",
    )
    ax3.set_xlabel("Peak Height Error (km)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax3.set_title(
        "(c) Peak Height Error Distribution",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.text(
        0.97,
        0.97,
        f"N = {len(peak_valid):,}\nMean = {np.mean(peak_valid):.2f} km\nMedian = {np.median(peak_valid):.2f} km\nStd = {np.std(peak_valid):.2f} km",
        transform=ax3.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"
        ),
    )

    # (d) Gradient RMSE Distribution
    grad_valid = grad_rmse_profiles[np.isfinite(grad_rmse_profiles)]
    grad_counts, _, _ = ax4.hist(
        grad_valid,
        bins=50,
        color="#CC79A7",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.6,
    )
    grad_peak_x, grad_peak_count = compute_hist_peak(grad_valid, bins=50)
    ax4.axvline(
        np.mean(grad_valid), color="#C73E1D", linestyle="--", linewidth=2
    )
    ax4.axvline(
        np.median(grad_valid), color="#5B8E7D", linestyle=":", linewidth=2
    )
    ax4.axvline(grad_peak_x, color="#6A4C93", linestyle="-.", linewidth=2)
    ax4.scatter(
        [grad_peak_x],
        [grad_peak_count],
        color="#6A4C93",
        s=40,
        zorder=4,
        edgecolors="white",
    )
    ax4.set_xlabel("Gradient RMSE (dBZ)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax4.set_title(
        "(d) Gradient RMSE Distribution",
        fontsize=13,
        fontweight="bold",
        loc="left",
    )
    ax4.grid(True, alpha=0.3, linestyle="--")
    ax4.text(
        0.97,
        0.97,
        f"N = {len(grad_valid):,}\nMean = {np.mean(grad_valid):.2f} dBZ\nMedian = {np.median(grad_valid):.2f} dBZ\nPeak = {grad_peak_x:.2f} dBZ",
        transform=ax4.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"
        ),
    )


def plot_profile_metric_distributions_v2(
    ax1, ax2, rmse_profiles, corr_profiles
):
    """绘制更接近期刊风格的平滑概率密度图"""
    rmse_valid = rmse_profiles[np.isfinite(rmse_profiles)]
    corr_valid = corr_profiles[np.isfinite(corr_profiles)]

    rmse_x, rmse_density, rmse_mode = compute_kde_curve(rmse_valid)
    corr_x, corr_density, corr_mode = compute_kde_curve(
        corr_valid, value_range=(-1, 1)
    )

    ax1.fill_between(rmse_x, rmse_density, color="#56B4E9", alpha=0.28)
    ax1.plot(rmse_x, rmse_density, color="#0072B2", linewidth=2.6)
    ax1.axvline(
        np.mean(rmse_valid), color="#D55E00", linestyle="--", linewidth=1.8
    )
    ax1.axvline(
        np.median(rmse_valid), color="#009E73", linestyle=":", linewidth=1.8
    )
    ax1.axvline(rmse_mode, color="#CC79A7", linestyle="-.", linewidth=1.8)
    rmse_mode_y = rmse_density[np.argmax(rmse_density)]
    ax1.scatter(
        [rmse_mode],
        [rmse_mode_y],
        color="#CC79A7",
        s=42,
        zorder=4,
        edgecolors="white",
    )
    ax1.set_xlabel("Per-profile RMSE (dBZ)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax1.set_title(
        "(a) RMSE Density", fontsize=13, fontweight="bold", loc="left"
    )
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.text(
        0.97,
        0.97,
        f"N = {len(rmse_valid):,}\nMean = {np.mean(rmse_valid):.2f}\nMedian = {np.median(rmse_valid):.2f}\nMode = {rmse_mode:.2f}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.82, edgecolor="gray"
        ),
    )

    ax2.fill_between(corr_x, corr_density, color="#E69F00", alpha=0.28)
    ax2.plot(corr_x, corr_density, color="#D55E00", linewidth=2.6)
    ax2.axvline(
        np.mean(corr_valid), color="#0072B2", linestyle="--", linewidth=1.8
    )
    ax2.axvline(
        np.median(corr_valid), color="#009E73", linestyle=":", linewidth=1.8
    )
    ax2.axvline(corr_mode, color="#CC79A7", linestyle="-.", linewidth=1.8)
    corr_mode_y = corr_density[np.argmax(corr_density)]
    ax2.scatter(
        [corr_mode],
        [corr_mode_y],
        color="#CC79A7",
        s=42,
        zorder=4,
        edgecolors="white",
    )
    ax2.set_xlabel("Per-profile Correlation", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=12, fontweight="bold")
    ax2.set_title(
        "(b) Correlation Density", fontsize=13, fontweight="bold", loc="left"
    )
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.set_xlim(-1.0, 1.0)
    ax2.text(
        0.97,
        0.97,
        f"N = {len(corr_valid):,}\nMean = {np.mean(corr_valid):.3f}\nMedian = {np.median(corr_valid):.3f}\nMode = {corr_mode:.3f}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.82, edgecolor="gray"
        ),
    )


def plot_publication_density_panel(
    ax,
    values,
    xlabel,
    title,
    fill_color,
    line_color,
    value_range=None,
    fmt=".2f",
):
    """绘制适合论文展示的摘要型分布图"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    x_grid, density, mode_x = compute_kde_curve(values, value_range=value_range)
    mean_x = np.mean(values)
    median_x = np.median(values)
    q1, q3 = np.percentile(values, [25, 75])
    whisker_low, whisker_high = np.percentile(values, [5, 95])

    density_max = density.max()
    y_box = -0.14 * density_max
    box_height = 0.08 * density_max
    jitter_low = -0.30 * density_max
    jitter_high = -0.19 * density_max

    ax.fill_between(x_grid, density, color=fill_color, alpha=0.26, zorder=1)
    ax.plot(x_grid, density, color=line_color, linewidth=2.7, zorder=3)
    ax.fill_between(
        x_grid,
        0,
        density,
        where=(x_grid >= q1) & (x_grid <= q3),
        color=line_color,
        alpha=0.18,
        zorder=2,
    )

    ax.hlines(
        y_box, whisker_low, whisker_high, color="0.35", linewidth=1.6, zorder=4
    )
    ax.add_patch(
        plt.Rectangle(
            (q1, y_box - box_height / 2),
            q3 - q1,
            box_height,
            facecolor="white",
            edgecolor="0.25",
            linewidth=1.4,
            zorder=5,
        )
    )
    ax.vlines(
        median_x,
        y_box - box_height / 2,
        y_box + box_height / 2,
        color="0.15",
        linewidth=2.0,
        zorder=6,
    )

    sample_size = min(len(values), 1200)
    sample_idx = np.linspace(0, len(values) - 1, sample_size).astype(int)
    sampled = np.sort(values)[sample_idx]
    y_jitter = np.linspace(jitter_low, jitter_high, sample_size)
    ax.scatter(
        sampled,
        y_jitter,
        s=6,
        color=line_color,
        alpha=0.16,
        edgecolors="none",
        zorder=0,
    )

    ax.axvline(mean_x, color="#D55E00", linestyle="--", linewidth=1.8, zorder=4)
    ax.axvline(
        median_x, color="#009E73", linestyle=":", linewidth=1.8, zorder=4
    )
    ax.axvline(mode_x, color="#CC79A7", linestyle="-.", linewidth=1.8, zorder=4)
    mode_y = density[np.argmax(density)]
    ax.scatter(
        [mode_x],
        [mode_y],
        color="#CC79A7",
        s=48,
        edgecolors="white",
        linewidth=0.8,
        zorder=6,
    )

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.set_ylim(jitter_low - 0.02 * density_max, density_max * 1.12)

    if value_range is not None:
        ax.set_xlim(*value_range)

    ax.text(
        0.97,
        0.97,
        f"N = {len(values):,}\nMean = {format(mean_x, fmt)}\nMedian = {format(median_x, fmt)}\nMode = {format(mode_x, fmt)}\nIQR = [{format(q1, fmt)}, {format(q3, fmt)}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.8,
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.84, edgecolor="gray"
        ),
    )


def plot_profile_metric_distributions_v3(
    ax1, ax2, rmse_profiles, corr_profiles
):
    """绘制论文风格的摘要型分布图"""
    rmse_valid = rmse_profiles[np.isfinite(rmse_profiles)]
    corr_valid = corr_profiles[np.isfinite(corr_profiles)]

    plot_publication_density_panel(
        ax1,
        rmse_valid,
        xlabel="Per-profile RMSE (dBZ)",
        title="(a) RMSE Distribution Summary",
        fill_color="#56B4E9",
        line_color="#0072B2",
        fmt=".2f",
    )
    plot_publication_density_panel(
        ax2,
        corr_valid,
        xlabel="Per-profile Correlation",
        title="(b) Correlation Distribution Summary",
        fill_color="#E69F00",
        line_color="#D55E00",
        value_range=(-1.0, 1.0),
        fmt=".3f",
    )


def plot_height_level_distribution_panel(
    ax,
    values,
    xlabel,
    title,
    fill_color,
    line_color,
    value_range=None,
    fmt=".2f",
):
    """绘制单个高度层的一维分布摘要图"""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.array([0.0, 0.0], dtype=np.float64)
    elif values.size == 1:
        values = np.array([values[0], values[0]], dtype=np.float64)

    plot_publication_density_panel(
        ax,
        values,
        xlabel=xlabel,
        title=title,
        fill_color=fill_color,
        line_color=line_color,
        value_range=value_range,
        fmt=fmt,
    )


def plot_reflectivity_distributions_by_height(
    fig, preds_dbz, trues_dbz, height_bins
):
    """绘制几个代表高度层的真值/预测值/差值分布图"""
    panel_specs = [
        ("True Reflectivity (dBZ)", "#56B4E9", "#0072B2"),
        ("Predicted Reflectivity (dBZ)", "#E69F00", "#D55E00"),
        ("Difference (Pred - True, dBZ)", "#F4A6A6", "#C73E1D"),
    ]

    for row_idx, bin_idx in enumerate(height_bins):
        height_km = 20.4 * (1 - bin_idx / 84)
        true_raw = trues_dbz[:, bin_idx]
        pred_raw = preds_dbz[:, bin_idx]
        true_vals = true_raw[true_raw >= -25]
        pred_vals = pred_raw[pred_raw >= -25]
        valid_diff = (true_raw >= -25) & (pred_raw >= -25)
        diff_vals = pred_raw[valid_diff] - true_raw[valid_diff]

        value_triplets = [true_vals, pred_vals, diff_vals]
        diff_finite = diff_vals[np.isfinite(diff_vals)]
        if diff_finite.size == 0:
            diff_finite = np.array([0.0, 0.0], dtype=np.float64)
        diff_limit = float(np.percentile(np.abs(diff_finite), 99))
        diff_limit = max(diff_limit, 5.0)
        value_ranges = [(-25, 20), (-25, 20), (-diff_limit, diff_limit)]
        labels = ["(a)", "(b)", "(c)"]

        for col_idx, (
            (xlabel, fill_color, line_color),
            values,
            value_range,
            label,
        ) in enumerate(zip(panel_specs, value_triplets, value_ranges, labels)):
            ax = fig.add_subplot(len(height_bins), 3, row_idx * 3 + col_idx + 1)
            plot_height_level_distribution_panel(
                ax,
                values,
                xlabel=xlabel,
                title=f"{label} {height_km:.1f} km",
                fill_color=fill_color,
                line_color=line_color,
                value_range=value_range,
                fmt=".2f",
            )
            if col_idx == 0:
                ax.set_ylabel("Density", fontsize=12, fontweight="bold")


def compute_kl_divergence_2d(pred_hist, true_hist, epsilon=1e-10):
    """计算两个二维直方图之间的KL散度"""
    pred_p = pred_hist + epsilon
    true_p = true_hist + epsilon

    pred_p = pred_p / np.sum(pred_p)
    true_p = true_p / np.sum(true_p)

    kl_div = np.sum(true_p * np.log(true_p / pred_p))

    m = 0.5 * (true_p + pred_p)
    kl_true_m = np.sum(true_p * np.log(true_p / m))
    kl_pred_m = np.sum(pred_p * np.log(pred_p / m))
    js_div = 0.5 * (kl_true_m + kl_pred_m)

    return kl_div, js_div


def _compute_reflectivity_height_histograms(preds_dbz, trues_dbz, heights):
    """计算真值、预测值以及二者二维统计差值"""
    n_bins = preds_dbz.shape[1]
    height_grid = np.tile(heights[None, :], (preds_dbz.shape[0], 1))
    valid_true = trues_dbz >= -25
    valid_pred = preds_dbz >= -25

    refl_edges = np.linspace(-25, 20, 61)
    height_edges = np.linspace(0, 20.4, n_bins + 1)
    true_hist2d, _, _ = np.histogram2d(
        trues_dbz[valid_true].flatten(),
        height_grid[valid_true].flatten(),
        bins=[refl_edges, height_edges],
    )
    pred_hist2d, _, _ = np.histogram2d(
        preds_dbz[valid_pred].flatten(),
        height_grid[valid_pred].flatten(),
        bins=[refl_edges, height_edges],
    )
    diff_hist2d = pred_hist2d - true_hist2d
    return refl_edges, height_edges, true_hist2d, pred_hist2d, diff_hist2d


def _draw_reflectivity_height_hist2d_row(
    fig,
    axes,
    preds_dbz,
    trues_dbz,
    heights,
    row_label="",
    count_vmin=0,
    count_vmax=500,
    diff_vmin=-200,
    diff_vmax=200,
):
    """在给定axes上绘制一行二维统计图，包括KL散度"""
    refl_edges, height_edges, true_hist2d, pred_hist2d, diff_hist2d = (
        _compute_reflectivity_height_histograms(preds_dbz, trues_dbz, heights)
    )
    prefix = f"{row_label} " if row_label else ""

    kl_div, js_div = compute_kl_divergence_2d(pred_hist2d, true_hist2d)

    panel_data = [
        (
            f"(a) {prefix}True Reflectivity Distribution",
            true_hist2d,
            "RdYlBu_r",
            "Counts",
            (count_vmin, count_vmax),
        ),
        (
            f"(b) {prefix}Predicted Reflectivity Distribution",
            pred_hist2d,
            "RdYlBu_r",
            "Counts",
            (count_vmin, count_vmax),
        ),
        (
            f"(c) {prefix}Difference of (b) - (a)",
            diff_hist2d,
            "RdBu_r",
            "Count Difference",
            (diff_vmin, diff_vmax),
        ),
    ]

    for ax, (title, hist2d, cmap, cbar_label, color_limits) in zip(
        axes, panel_data
    ):
        mesh_kwargs = {"cmap": cmap, "shading": "auto"}
        if color_limits is not None:
            mesh_kwargs["vmin"] = color_limits[0]
            mesh_kwargs["vmax"] = color_limits[1]
        mesh = ax.pcolormesh(refl_edges, height_edges, hist2d.T, **mesh_kwargs)
        ax.set_xlabel("Reflectivity (dBZ)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Height (km)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
        ax.set_ylim(0, 20.4)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=25)
        cbar.set_label(cbar_label, fontsize=11, fontweight="bold")

    similarity = (1 - js_div / np.log(2)) * 100
    kl_text = f"Similarity: {similarity:.1f}%"
    for ax in axes[1:]:
        ax.text(
            0.07,
            0.97,
            kl_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.75,
                edgecolor="gray",
            ),
        )


def plot_reflectivity_height_hist2d(fig, preds_dbz, trues_dbz, heights):
    """绘制反射率-高度二维分布图"""
    axes = [fig.add_subplot(1, 3, idx + 1) for idx in range(3)]
    _draw_reflectivity_height_hist2d_row(
        fig,
        axes,
        preds_dbz,
        trues_dbz,
        heights,
        count_vmin=0,
        count_vmax=500,
        diff_vmin=-400,
        diff_vmax=400,
    )


def save_statistics_to_csv(preds_dbz, trues_dbz, save_path):
    """保存统计结果到CSV文件"""
    stats_data = []

    for bin_idx in range(85):
        p = preds_dbz[:, bin_idx]
        t = trues_dbz[:, bin_idx]
        valid = (t > -30) & (p > -30)

        if np.sum(valid) > 100:
            p_valid = p[valid]
            t_valid = t[valid]
            r2 = np.corrcoef(p_valid, t_valid)[0, 1] ** 2
            rmse = np.sqrt(np.mean((p_valid - t_valid) ** 2))
            bias = np.mean(p_valid - t_valid)
            height_km = 20.4 * (1 - bin_idx / 84)
            n_samples = np.sum(valid)

            stats_data.append(
                {
                    "Bin": bin_idx,
                    "Height_km": round(height_km, 2),
                    "N_Samples": n_samples,
                    "R2": round(r2, 4),
                    "RMSE_dBZ": round(rmse, 4),
                    "Bias_dBZ": round(bias, 4),
                }
            )

    df = pd.DataFrame(stats_data)
    df.to_csv(save_path, index=False)
    print(f"[Saved] {save_path} ({len(df)} rows)")


def save_gradient_rmse_to_csv(preds_dbz, trues_dbz, save_path):
    """保存梯度RMSE统计结果到CSV文件"""
    grad_rmse_by_height, _ = compute_gradient_rmse(preds_dbz, trues_dbz)

    stats_data = []
    for i, grad_rmse in enumerate(grad_rmse_by_height):
        height_km = 20.4 * (1 - (i + 0.5) / 84)
        stats_data.append(
            {
                "Gradient_Bin": i,
                "Height_km": round(height_km, 2),
                "Gradient_RMSE_dBZ": round(grad_rmse, 4),
            }
        )

    df = pd.DataFrame(stats_data)
    df.to_csv(save_path, index=False)
    print(f"[Saved] {save_path} ({len(df)} rows)")


def save_peak_height_error_to_csv(
    peak_errors, pred_peaks_idx, true_peaks_idx, valid_counts, save_path
):
    """保存峰值高度误差到CSV文件"""
    df = pd.DataFrame(
        {
            "Profile_Index": np.arange(len(peak_errors)),
            "Valid_Count": valid_counts,
            "Peak_Height_Error_km": peak_errors,
            "Pred_Peak_Index": pred_peaks_idx,
            "True_Peak_Index": true_peaks_idx,
        }
    )
    df.to_csv(save_path, index=False)
    print(f"[Saved] {save_path} ({len(df)} rows)")


def save_gradient_rmse_profiles_to_csv(
    grad_rmse_profiles, valid_counts, save_path
):
    """保存廓线级别梯度RMSE到CSV文件"""
    df = pd.DataFrame(
        {
            "Profile_Index": np.arange(len(grad_rmse_profiles)),
            "Valid_Count": valid_counts,
            "Gradient_RMSE_dBZ": grad_rmse_profiles,
        }
    )
    df.to_csv(save_path, index=False)
    print(f"[Saved] {save_path} ({len(df)} rows)")


def compute_profile_metrics_unfiltered(preds_dbz, trues_dbz):
    """逐个廓线计算RMSE和相关系数（未滤除版本）"""
    n_profiles = preds_dbz.shape[0]
    rmse_profiles = np.zeros(n_profiles, dtype=np.float32)
    corr_profiles = np.zeros(n_profiles, dtype=np.float32)

    for i in range(n_profiles):
        p = preds_dbz[i]
        t = trues_dbz[i]

        rmse_profiles[i] = np.sqrt(np.mean((p - t) ** 2))

        if np.std(p) > 1e-6 and np.std(t) > 1e-6:
            corr_profiles[i] = np.corrcoef(p, t)[0, 1]
        else:
            corr_profiles[i] = np.nan

    return rmse_profiles, corr_profiles


def compute_gradient_rmse_profiles_unfiltered(preds_dbz, trues_dbz):
    """逐个廓线计算梯度RMSE（未滤除版本）"""
    n_profiles = preds_dbz.shape[0]
    grad_rmse_profiles = np.zeros(n_profiles, dtype=np.float32)

    for i in range(n_profiles):
        p = preds_dbz[i]
        t = trues_dbz[i]

        pred_grad = np.diff(p)
        true_grad = np.diff(t)
        grad_rmse_profiles[i] = np.sqrt(np.mean((pred_grad - true_grad) ** 2))

    return grad_rmse_profiles


def save_summary_statistics_to_csv(
    preds_dbz,
    trues_dbz,
    rmse_profiles_filtered,
    corr_profiles_filtered,
    grad_rmse_profiles_filtered,
    peak_errors_filtered,
    rmse_profiles_unfiltered,
    corr_profiles_unfiltered,
    grad_rmse_profiles_unfiltered,
    save_path,
):
    """保存所有评估指标的统计摘要到CSV文件"""

    global_rmse = np.sqrt(np.mean((preds_dbz - trues_dbz) ** 2))
    global_bias = np.mean(preds_dbz - trues_dbz)

    def calc_stats(values, name, value_range=None):
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            return {
                f"{name}_mean": np.nan,
                f"{name}_max": np.nan,
                f"{name}_min": np.nan,
                f"{name}_median": np.nan,
                f"{name}_std": np.nan,
                f"{name}_mode": np.nan,
            }
        counts, bin_edges = np.histogram(valid, bins=50, range=value_range)
        peak_idx = np.argmax(counts)
        mode_value = 0.5 * (bin_edges[peak_idx] + bin_edges[peak_idx + 1])
        return {
            f"{name}_mean": np.mean(valid),
            f"{name}_max": np.max(valid),
            f"{name}_min": np.min(valid),
            f"{name}_median": np.median(valid),
            f"{name}_std": np.std(valid),
            f"{name}_mode": mode_value,
        }

    summary_data = {"Metric": [], "Value": []}

    summary_data["Metric"].extend(["Global_RMSE_dBZ", "Global_Bias_dBZ"])
    summary_data["Value"].extend([global_rmse, global_bias])

    for key, value in calc_stats(
        corr_profiles_unfiltered, "Corr_Unfiltered", value_range=(-1, 1)
    ).items():
        summary_data["Metric"].append(key)
        summary_data["Value"].append(value)

    for key, value in calc_stats(
        corr_profiles_filtered, "Corr_Filtered", value_range=(-1, 1)
    ).items():
        summary_data["Metric"].append(key)
        summary_data["Value"].append(value)

    for key, value in calc_stats(
        rmse_profiles_unfiltered, "RMSE_Unfiltered"
    ).items():
        summary_data["Metric"].append(key)
        summary_data["Value"].append(value)

    for key, value in calc_stats(
        rmse_profiles_filtered, "RMSE_Filtered"
    ).items():
        summary_data["Metric"].append(key)
        summary_data["Value"].append(value)

    for key, value in calc_stats(
        grad_rmse_profiles_unfiltered, "GradRMSE_Unfiltered"
    ).items():
        summary_data["Metric"].append(key)
        summary_data["Value"].append(value)

    for key, value in calc_stats(
        grad_rmse_profiles_filtered, "GradRMSE_Filtered"
    ).items():
        summary_data["Metric"].append(key)
        summary_data["Value"].append(value)

    for key, value in calc_stats(
        peak_errors_filtered, "PeakError_Filtered"
    ).items():
        summary_data["Metric"].append(key)
        summary_data["Value"].append(value)

    df = pd.DataFrame(summary_data)
    df.to_csv(save_path, index=False)
    print(f"[Saved] Summary statistics to {save_path}")

    return df


def save_formatted_summary_to_csv(
    trial_id,
    global_rmse,
    global_bias,
    corr_stats,
    rmse_stats,
    grad_rmse_stats,
    peak_error_stats,
    save_path,
    corr_stats_unfiltered=None,
    rmse_stats_unfiltered=None,
    grad_rmse_stats_unfiltered=None,
):
    """保存格式化的试验汇总CSV，便于论文表格使用"""

    def format_mean_std_mode(mean, std, mode, decimals=3):
        return (
            f"({mean:.{decimals}f} ± {std:.{decimals}f}), {mode:.{decimals}f}"
        )

    row_data = {
        "Trial": trial_id,
        "Global_RMSE_dBZ": round(global_rmse, 3),
        "Global_Bias_dBZ": round(global_bias, 3),
        "Correlation_Filtered": format_mean_std_mode(
            corr_stats["mean"], corr_stats["std"], corr_stats["mode"]
        ),
        "RMSE_dBZ_Filtered": format_mean_std_mode(
            rmse_stats["mean"], rmse_stats["std"], rmse_stats["mode"]
        ),
        "GradRMSE_dBZ_Filtered": format_mean_std_mode(
            grad_rmse_stats["mean"],
            grad_rmse_stats["std"],
            grad_rmse_stats["mode"],
        ),
        "PeakError_km_Filtered": format_mean_std_mode(
            peak_error_stats["mean"],
            peak_error_stats["std"],
            peak_error_stats["mode"],
        ),
    }

    if corr_stats_unfiltered is not None:
        row_data["Correlation_Unfiltered"] = format_mean_std_mode(
            corr_stats_unfiltered["mean"],
            corr_stats_unfiltered["std"],
            corr_stats_unfiltered["mode"],
        )
    if rmse_stats_unfiltered is not None:
        row_data["RMSE_dBZ_Unfiltered"] = format_mean_std_mode(
            rmse_stats_unfiltered["mean"],
            rmse_stats_unfiltered["std"],
            rmse_stats_unfiltered["mode"],
        )
    if grad_rmse_stats_unfiltered is not None:
        row_data["GradRMSE_dBZ_Unfiltered"] = format_mean_std_mode(
            grad_rmse_stats_unfiltered["mean"],
            grad_rmse_stats_unfiltered["std"],
            grad_rmse_stats_unfiltered["mode"],
        )

    if save_path.exists():
        df_new = pd.DataFrame([row_data])
        df_existing = pd.read_csv(save_path, encoding="utf-8-sig")

        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = np.nan

        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new[col] = np.nan

        if trial_id in df_existing["Trial"].values:
            idx = df_existing[df_existing["Trial"] == trial_id].index[0]
            for col in df_new.columns:
                if col != "Trial":
                    df_existing.loc[idx, col] = df_new[col].values[0]
            df_existing.to_csv(save_path, index=False, encoding="utf-8-sig")
        else:
            pd.concat([df_existing, df_new], ignore_index=True).to_csv(
                save_path, index=False, encoding="utf-8-sig"
            )
    else:
        df = pd.DataFrame([row_data])
        df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[Saved] Formatted summary to {save_path}")


# ============================================
# 主测试流程
# ============================================

# 设置绘图风格
setup_paper_style()

# 打印测试信息
print("=" * 60)
print("MLP模型测试与可视化 - CloudSat反射率预测")
print("=" * 60)
print(f"设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"\n输入维度: {IN_DIM} (AHI(16) + 几何(7) + BT扩展(12) + ERA5(r,t,54))")
print(f"输入组件: AHI(16) + 几何(7) + BT扩展(12) + ERA5(r,t,full)")

# 创建保存目录
save_dir = Path(SAVE_DIR)
save_dir.mkdir(parents=True, exist_ok=True)

# 加载模型
checkpoint_path = Path(CHECKPOINT_DIR) / "best_model.pt"

print(f"\n[加载模型] {checkpoint_path}")
if not checkpoint_path.exists():
    print(f"错误: 模型文件不存在: {checkpoint_path}")
    exit(1)

checkpoint = torch.load(
    checkpoint_path, map_location=DEVICE, weights_only=False
)
print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")

model = SimpleMLP(in_dim=IN_DIM, out_dim=OUT_DIM, dropout=DROPOUT).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"  模型参数量: {total_params:,}")

# 加载测试数据
print(f"\n[加载数据] 测试集...")
test_dataset = CloudSatDataset(DATA_ROOT, "test", preload=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 生成预测
print(f"\n[生成预测] 处理 {len(test_dataset)} 个样本...")

all_preds = []
all_trues = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch["input"].to(DEVICE)
        target = batch["reflectivity"]

        pred = model(inputs).cpu()

        all_preds.append(pred.numpy())
        all_trues.append(target.numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_trues = np.concatenate(all_trues, axis=0)

# 读取陆地/海洋掩码
with h5py.File(test_dataset.h5_path, "r") as h5_geo:
    is_land = h5_geo["lat_lon_angle"][:, 6] >= 0

# 转换为dBZ
preds_dbz = denormalize_reflectivity(all_preds)
trues_dbz = denormalize_reflectivity(all_trues)
heights = np.linspace(20.4, 0, 85)

# 计算统计指标
rmse = np.sqrt(np.mean((preds_dbz - trues_dbz) ** 2))
bias = np.mean(preds_dbz - trues_dbz)
corr = np.corrcoef(preds_dbz.flatten(), trues_dbz.flatten())[0, 1]

print(f"\n[整体精度指标的统计结果]")
print(f"  RMSE: {rmse:.3f} dBZ")
print(f"  Bias: {bias:.3f} dBZ")
print(f"  Correlation: {corr:.3f}")

# 计算峰值高度误差（带约束条件）
peak_errors, pred_peaks_idx, true_peaks_idx, peak_valid_counts = (
    compute_peak_height_error_constrained(preds_dbz, trues_dbz, heights)
)
peak_error_mean = np.mean(np.abs(peak_errors[np.isfinite(peak_errors)]))
peak_error_std = np.std(peak_errors[np.isfinite(peak_errors)])
peak_error_median = np.median(peak_errors[np.isfinite(peak_errors)])
print(f"\n[峰值高度误差统计]")
print(f"  平均绝对误差: {peak_error_mean:.3f} km")
print(f"  标准差: {peak_error_std:.3f} km")
print(f"  中位数: {peak_error_median:.3f} km")

# 计算梯度RMSE
grad_rmse_by_height, grad_rmse_mean = compute_gradient_rmse(
    preds_dbz, trues_dbz
)
grad_rmse_max = np.max(grad_rmse_by_height)
grad_rmse_max_height = 20.4 * (1 - np.argmax(grad_rmse_by_height) / 84)
print(f"\n[梯度RMSE统计]")
print(f"  平均梯度RMSE: {grad_rmse_mean:.3f} dBZ")
print(
    f"  最大梯度RMSE: {grad_rmse_max:.3f} dBZ (位于{grad_rmse_max_height:.1f} km)"
)

# 定义配色和剖面范围
cmap_reflectivity = cmaps.MPL_jet
vmin, vmax = -40, 25

# 自动生成所有剖面，每100个样本一个
SECTION_LENGTH = 100
total_samples = len(preds_dbz)
num_sections = (total_samples + SECTION_LENGTH - 1) // SECTION_LENGTH

# ============================================
# 1. 创建剖面对比图
# ============================================
print(f"\n[生成可视化]")
print(f"  剖面长度: {SECTION_LENGTH}")
print(f"  剖面数量: {num_sections}")

for i in range(0,num_sections,100):
    start_idx = i * SECTION_LENGTH
    n_samples = min(SECTION_LENGTH, total_samples - start_idx)
    section_name = f"Samples {start_idx}-{start_idx + n_samples - 1}"

    pred_sec = preds_dbz[start_idx : start_idx + n_samples, :].T
    true_sec = trues_dbz[start_idx : start_idx + n_samples, :].T
    diff_sec = pred_sec - true_sec
    actual_n = n_samples

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4))
    fig.suptitle(
        f"{section_name} - Cross-Section Comparison",
        fontsize=15,
        fontweight="bold",
        y=0.95,
    )

    plot_cross_section(
        axes[0], true_sec, actual_n, "True", vmin, vmax, cmap_reflectivity
    )
    plot_cross_section(
        axes[1], pred_sec, actual_n, "Predicted", vmin, vmax, cmap_reflectivity
    )
    plot_difference_section(axes[2], diff_sec, actual_n, "Difference")

    # 计算该 section 的统计指标
    p_flat = pred_sec.flatten()
    t_flat = true_sec.flatten()
    valid = (t_flat > -30) & (p_flat > -30)
    if np.sum(valid) > 0:
        p_valid = p_flat[valid]
        t_valid = t_flat[valid]
        rmse_sec = np.sqrt(np.mean((p_valid - t_valid) ** 2))
        corr_sec = np.corrcoef(p_valid, t_valid)[0, 1]
        bias_sec = np.mean(p_valid - t_valid)
    else:
        rmse_sec = np.sqrt(np.mean((p_flat - t_flat) ** 2))
        corr_sec = np.corrcoef(p_flat, t_flat)[0, 1]
        bias_sec = np.mean(p_flat - t_flat)

    # 计算梯度RMSE
    pred_sec_flat = preds_dbz[start_idx : start_idx + n_samples, :]
    true_sec_flat = trues_dbz[start_idx : start_idx + n_samples, :]
    grad_rmse_profiles = compute_gradient_rmse_profiles(
        pred_sec_flat, true_sec_flat
    )
    grad_rmse_mean_sec = np.mean(grad_rmse_profiles)
    grad_rmse_median_sec = np.median(grad_rmse_profiles)

    # 计算逐廓线RMSE
    profile_rmse = np.sqrt(
        np.mean((pred_sec_flat - true_sec_flat) ** 2, axis=1)
    )
    profile_rmse_mean = np.mean(profile_rmse)
    profile_rmse_median = np.median(profile_rmse)

    stats_text = (
        f"RMSE = {rmse_sec:.2f} dBZ\n"
        f"Corr = {corr_sec:.3f}\n"
        f"Bias = {bias_sec:.2f} dBZ\n"
        f"GradRMSE = {grad_rmse_median_sec:.2f} dBZ\n"
        f"ProfRMSE = {profile_rmse_median:.2f} dBZ"
    )
    axes[1].text(
        0.02,
        0.98,
        stats_text,
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.95, edgecolor="gray"
        ),
    )

    plt.tight_layout()
    save_path = save_dir / f"section_{start_idx:06d}_{start_idx + n_samples - 1:06d}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  [{i+1}/{num_sections}] 保存: {save_path}")
    plt.close()

# ============================================
# 2. 创建综合评估图
# ============================================
fig = plt.figure(figsize=(21, 4))
fig.gs = fig.add_gridspec(1, 4, wspace=0.17)

ax1 = fig.add_subplot(fig.gs[0, 0])
ax2 = fig.add_subplot(fig.gs[0, 1])
ax3 = fig.add_subplot(fig.gs[0, 2])
ax4 = fig.add_subplot(fig.gs[0, 3])

plot_profile_metrics(ax1, ax2, ax3, preds_dbz, trues_dbz, heights)
plot_scatter_comparison(ax4, preds_dbz, trues_dbz)

fig.suptitle(
    "Model Performance Analysis",
    fontsize=16,
    fontweight="bold",
)
save_path = save_dir / "performance_analysis.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"  保存: {save_path}")
plt.close()

# ============================================
# 3. 创建高中低三层数点对比图
# ============================================
fig_multi = plt.figure(figsize=(15, 4))
plot_multi_level_scatter(fig_multi, preds_dbz, trues_dbz)
fig_multi.suptitle(
    "High/Mid/Low Level Scatter Comparison",
    fontsize=15,
    fontweight="bold",
    y=1.07,
)
save_path_multi = save_dir / "multi_level_scatter.png"
plt.savefig(save_path_multi, dpi=200, bbox_inches="tight", facecolor="white")
print(f"  保存: {save_path_multi}")
plt.close()

# ============================================
# 3.5. 反射率分布和峰值高度分布图
# ============================================
fig_refl_peak = plt.figure(figsize=(14, 5))
plot_reflectivity_and_peak_height_distributions(
    fig_refl_peak, preds_dbz, trues_dbz, heights
)
fig_refl_peak.suptitle(
    "Reflectivity and Peak Height Distributions",
    fontsize=15,
    fontweight="bold",
)
save_path_refl_peak = save_dir / "reflectivity_peak_distributions.png"
plt.savefig(
    save_path_refl_peak, dpi=200, bbox_inches="tight", facecolor="white"
)
print(f"  保存: {save_path_refl_peak}")
plt.close()

# ============================================
# 4. 逐廓线分布图
# ============================================
rmse_profiles, corr_profiles, valid_counts = compute_profile_metrics(
    preds_dbz, trues_dbz
)

# 计算廓线级别的梯度RMSE（带约束条件）
grad_rmse_profiles, grad_valid_counts = (
    compute_gradient_rmse_profiles_constrained(preds_dbz, trues_dbz)
)

fig_profile_dist = plt.figure(figsize=(25, 4.8))
gs = fig_profile_dist.add_gridspec(1, 4, wspace=0.25)
ax_rmse = fig_profile_dist.add_subplot(gs[0, 0])
ax_corr = fig_profile_dist.add_subplot(gs[0, 1])
ax_peak = fig_profile_dist.add_subplot(gs[0, 2])
ax_grad = fig_profile_dist.add_subplot(gs[0, 3])

plot_profile_metric_distributions(
    ax_rmse,
    ax_corr,
    ax_peak,
    ax_grad,
    rmse_profiles,
    corr_profiles,
    peak_errors,
    grad_rmse_profiles,
)
fig_profile_dist.suptitle(
    "Per-profile Metric Distributions",
    fontsize=15,
)
save_path_profile_dist = save_dir / "profile_metric_distributions.png"
plt.savefig(
    save_path_profile_dist, dpi=200, bbox_inches="tight", facecolor="white"
)
print(f"  保存: {save_path_profile_dist}")
plt.close()

fig_profile_dist_v2, (ax_rmse_v2, ax_corr_v2) = plt.subplots(
    1, 2, figsize=(14, 4.8)
)
plot_profile_metric_distributions_v2(
    ax_rmse_v2, ax_corr_v2, rmse_profiles, corr_profiles
)
fig_profile_dist_v2.suptitle(
    "Per-profile RMSE and Correlation Density Distributions",
    fontsize=15,
)
save_path_profile_dist_v2 = save_dir / "profile_metric_distributions_v2.png"
plt.savefig(
    save_path_profile_dist_v2, dpi=250, bbox_inches="tight", facecolor="white"
)
print(f"  保存: {save_path_profile_dist_v2}")
plt.close()

fig_profile_dist_v3, (ax_rmse_v3, ax_corr_v3) = plt.subplots(
    1, 2, figsize=(14.5, 5.1)
)
plot_profile_metric_distributions_v3(
    ax_rmse_v3, ax_corr_v3, rmse_profiles, corr_profiles
)
fig_profile_dist_v3.suptitle(
    "Per-profile RMSE and Correlation Publication-style Summary",
    fontsize=15,
)
save_path_profile_dist_v3 = save_dir / "profile_metric_distributions_v3.png"
plt.savefig(
    save_path_profile_dist_v3, dpi=300, bbox_inches="tight", facecolor="white"
)
print(f"  保存: {save_path_profile_dist_v3}")
plt.close()

# ============================================
# 5. 指定高度层分布图
# ============================================
fig_height_dist = plt.figure(figsize=(14.5, 12.5))
plot_reflectivity_distributions_by_height(
    fig_height_dist, preds_dbz, trues_dbz, height_bins=[10, 42, 74]
)
fig_height_dist.suptitle(
    "Height-specific Reflectivity Distributions",
    fontsize=15,
)
save_path_height_dist = save_dir / "reflectivity_distributions_by_height.png"
plt.savefig(
    save_path_height_dist, dpi=300, bbox_inches="tight", facecolor="white"
)
print(f"  保存: {save_path_height_dist}")
plt.close()

# ============================================
# 6. 反射率-高度二维分布图
# ============================================
fig_hist2d = plt.figure(figsize=(15.5, 4.8))
plot_reflectivity_height_hist2d(fig_hist2d, preds_dbz, trues_dbz, heights)
fig_hist2d.suptitle(
    "Reflectivity-Height 2D Distributions",
    fontsize=15,
)
save_path_hist2d = save_dir / "reflectivity_height_hist2d.png"
plt.savefig(save_path_hist2d, dpi=300, bbox_inches="tight", facecolor="white")
print(f"  保存: {save_path_hist2d}")
plt.close()

fig_hist2d_split, axes_split = plt.subplots(2, 3, figsize=(15.5, 9.0))
_draw_reflectivity_height_hist2d_row(
    fig_hist2d_split,
    axes_split[0],
    preds_dbz[~is_land],
    trues_dbz[~is_land],
    heights,
    row_label="Ocean",
    count_vmin=0,
    count_vmax=300,
    diff_vmin=-300,
    diff_vmax=300,
)
_draw_reflectivity_height_hist2d_row(
    fig_hist2d_split,
    axes_split[1],
    preds_dbz[is_land],
    trues_dbz[is_land],
    heights,
    row_label="Land",
    count_vmin=0,
    count_vmax=300,
    diff_vmin=-300,
    diff_vmax=300,
)
fig_hist2d_split.suptitle(
    "Reflectivity-Height 2D Distributions by Surface Type",
    fontsize=15,
    y=0.94,
)
save_path_hist2d_split = save_dir / "reflectivity_height_hist2d_land_ocean.png"
plt.savefig(
    save_path_hist2d_split, dpi=300, bbox_inches="tight", facecolor="white"
)
print(f"  保存: {save_path_hist2d_split}")
plt.close()

# ============================================
# 7. 保存统计结果到CSV
# ============================================
profile_stats_df = pd.DataFrame(
    {
        "Profile_Index": np.arange(len(rmse_profiles)),
        "Valid_Count": valid_counts,
        "RMSE_dBZ": rmse_profiles,
        "Correlation": corr_profiles,
    }
)
profile_csv_path = save_dir / "statistics_by_profile.csv"
profile_stats_df.to_csv(profile_csv_path, index=False)
print(f"[Saved] {profile_csv_path} ({len(profile_stats_df)} rows)")

csv_path = save_dir / "statistics_by_bin.csv"
save_statistics_to_csv(preds_dbz, trues_dbz, csv_path)

# 保存梯度RMSE (按高度层)
grad_csv_path = save_dir / "gradient_rmse_by_height.csv"
save_gradient_rmse_to_csv(preds_dbz, trues_dbz, grad_csv_path)

# 保存廓线级别梯度RMSE
grad_profile_csv_path = save_dir / "gradient_rmse_profiles.csv"
save_gradient_rmse_profiles_to_csv(
    grad_rmse_profiles, grad_valid_counts, grad_profile_csv_path
)

# 保存峰值高度误差
peak_csv_path = save_dir / "peak_height_errors.csv"
save_peak_height_error_to_csv(
    peak_errors,
    pred_peaks_idx,
    true_peaks_idx,
    peak_valid_counts,
    peak_csv_path,
)

# 计算未滤除版本的指标
rmse_profiles_unfiltered, corr_profiles_unfiltered = (
    compute_profile_metrics_unfiltered(preds_dbz, trues_dbz)
)
grad_rmse_profiles_unfiltered = compute_gradient_rmse_profiles_unfiltered(
    preds_dbz, trues_dbz
)

# 保存汇总统计CSV
summary_csv_path = save_dir / "summary_statistics.csv"
save_summary_statistics_to_csv(
    preds_dbz,
    trues_dbz,
    rmse_profiles,
    corr_profiles,
    grad_rmse_profiles,
    peak_errors,
    rmse_profiles_unfiltered,
    corr_profiles_unfiltered,
    grad_rmse_profiles_unfiltered,
    summary_csv_path,
)

# 保存格式化汇总CSV（用于论文表格）
global_rmse = np.sqrt(np.mean((preds_dbz - trues_dbz) ** 2))
global_bias = np.mean(preds_dbz - trues_dbz)


def calc_stats_for_format(values, value_range=None):
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return {k: np.nan for k in ["mean", "std", "mode"]}
    counts, bin_edges = np.histogram(valid, bins=50, range=value_range)
    peak_idx = np.argmax(counts)
    mode_value = 0.5 * (bin_edges[peak_idx] + bin_edges[peak_idx + 1])
    return {
        "mean": np.mean(valid),
        "std": np.std(valid),
        "mode": mode_value,
    }


# 过滤版本统计量
corr_stats = calc_stats_for_format(corr_profiles, value_range=(-1, 1))
rmse_stats = calc_stats_for_format(rmse_profiles)
grad_rmse_stats = calc_stats_for_format(grad_rmse_profiles)
peak_error_stats = calc_stats_for_format(peak_errors)

# 未过滤版本统计量
corr_stats_unfiltered = calc_stats_for_format(corr_profiles_unfiltered, value_range=(-1, 1))
rmse_stats_unfiltered = calc_stats_for_format(rmse_profiles_unfiltered)
grad_rmse_stats_unfiltered = calc_stats_for_format(grad_rmse_profiles_unfiltered)

formatted_summary_path = save_dir / "formatted_summary.csv"
save_formatted_summary_to_csv(
    "MLP",
    global_rmse,
    global_bias,
    corr_stats,
    rmse_stats,
    grad_rmse_stats,
    peak_error_stats,
    formatted_summary_path,
    corr_stats_unfiltered=corr_stats_unfiltered,
    rmse_stats_unfiltered=rmse_stats_unfiltered,
    grad_rmse_stats_unfiltered=grad_rmse_stats_unfiltered,
)

# ============================================
# 测试完成
# ============================================
print("\n" + "=" * 60)
print("测试完成!")
print(f"结果保存到: {save_dir}")
print("=" * 60)

# ============================================
# 释放内存
# ============================================
print(f"\n[释放内存]")

del model
del all_preds, all_trues
del preds_dbz, trues_dbz
del test_dataset, test_loader

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"  CUDA缓存已清理")

print(f"  内存已释放\n")
