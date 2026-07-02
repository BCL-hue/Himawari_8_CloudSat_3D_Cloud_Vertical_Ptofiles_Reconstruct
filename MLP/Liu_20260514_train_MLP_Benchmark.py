"""
MLP模型训练脚本 - AHI + 几何信息 + BT扩展指标 + ERA5廓线 预测CloudSat反射率
包含: 数据集、模型、训练流程

输入:
- AHI 16通道
- 几何信息 7维 (sin/cos编码 + 地形编码)
- BT扩展指标 12维 (VCI + 11个BT相关指数)
- ERA5廓线 54维 (r, t 各27层)

损失函数: L1
"""

import os
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================
# 随机数种子设置 - 确保实验可复现
# ============================================


def set_seed(seed=42):
    """设置随机数种子以确保实验可复现"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================
# 配置区域
# ============================================

# 数据路径
DATA_ROOT = "/mnt/g/3D_Cloud_Reflectivety_Profile_Projection/3D_Train_data/point"
SAVE_DIR = "/mnt/g/3D_Cloud_Reflectivety_Profile_Projection/Point_projection/CloudSat_GEOPROF/L_20260702_code/MLP/checkpoints"

# 训练参数
BATCH_SIZE = 256
NUM_EPOCHS = 300
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 20
LOG_INTERVAL = 50

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型参数
IN_DIM = 16 + 7 + 12 + 2 * 27  # AHI + geo + BT扩展 + ERA5(r,t各27层)
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


def get_pressure_index(pressure_value):
    """获取气压值在PRESSURE_LEVELS中的索引"""
    for i, p in enumerate(PRESSURE_LEVELS):
        if p == pressure_value:
            return i
    raise ValueError(f"气压值 {pressure_value} 不在PRESSURE_LEVELS中")


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
# 修改水汽通道差为：BT_8_9 (6.2-6.9μm), BT_9_10 (6.9-7.3μm)
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
# ERA5廓线数据统计量 (变量顺序: r, w, q, t)
# era5_profile_low: [N, 6, 14] -> r(0), u(1), v(2), w(3), q(4), t(5)
# era5_profile_high: [N, 6, 13]
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

# 垂直速度 - 低层 (14层)
ERA5_W_LOW_MEANS = np.array(
    [
        -0.0325,
        -0.0358,
        -0.0395,
        -0.0439,
        -0.0485,
        -0.0523,
        -0.0550,
        -0.0564,
        -0.0571,
        -0.0576,
        -0.0584,
        -0.0593,
        -0.0594,
        -0.0615,
    ]
)
ERA5_W_LOW_STDS = np.array(
    [
        0.1173,
        0.1495,
        0.1906,
        0.2276,
        0.2553,
        0.2743,
        0.2866,
        0.2953,
        0.3013,
        0.3051,
        0.3095,
        0.3184,
        0.3259,
        0.3305,
    ]
)

# 比湿 - 低层 (14层)
ERA5_Q_LOW_MEANS = np.array(
    [
        0.01364,
        0.01337,
        0.01285,
        0.01209,
        0.01137,
        0.01072,
        0.01011,
        0.00952,
        0.00894,
        0.00834,
        0.00773,
        0.00648,
        0.00532,
        0.00437,
    ]
)
ERA5_Q_LOW_STDS = np.array(
    [
        0.00560,
        0.00553,
        0.00531,
        0.00495,
        0.00464,
        0.00440,
        0.00421,
        0.00405,
        0.00389,
        0.00372,
        0.00353,
        0.00311,
        0.00269,
        0.00240,
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

# 垂直速度 - 高层 (13层)
ERA5_W_HIGH_MEANS = np.array(
    [
        -0.0652,
        -0.0711,
        -0.0778,
        -0.0847,
        -0.0893,
        -0.0865,
        -0.0738,
        -0.0631,
        -0.0504,
        -0.0378,
        -0.0255,
        -0.0136,
        -0.0063,
    ]
)
ERA5_W_HIGH_STDS = np.array(
    [
        0.3284,
        0.3225,
        0.3180,
        0.3121,
        0.2980,
        0.2702,
        0.2329,
        0.2109,
        0.1859,
        0.1566,
        0.1182,
        0.0739,
        0.0398,
    ]
)

# 比湿 - 高层 (13层) - 从训练数据精确计算
ERA5_Q_HIGH_MEANS = np.array(
    [
        0.00358769,
        0.00275112,
        0.00198693,
        0.00135575,
        0.00084591,
        0.00046079,
        0.00020475,
        0.00012165,
        0.00006435,
        0.00002953,
        0.00001195,
        0.00000483,
        0.00000280,
    ]
)
ERA5_Q_HIGH_STDS = np.array(
    [
        0.00219481,
        0.00179884,
        0.00136113,
        0.00096180,
        0.00061545,
        0.00033737,
        0.00014706,
        0.00008619,
        0.00004463,
        0.00001978,
        0.00000712,
        0.00000204,
        0.00000082,
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
    normalized = 2 * (data + 35.0) / 55.0 - 1.0

    return normalized


# ============================================
# 设置随机数种子
# ============================================
set_seed(42)


# ============================================
# 数据集类 (AHI + 几何信息 + BT扩展指标 + ERA5廓线)
# ============================================


class CloudSatDataset(Dataset):
    """
    数据集 - AHI 16通道 + 几何信息 7维 + BT扩展指标 12维 + ERA5廓线(r,t各27层)

    输入维度: 16 + 7 + 12 + 54 = 89

    BT扩展指标 (12个):
    1. VCI - 云检测
    2-9. BT_7_14, BT_8_10, BT_9_8, BT_10_9, BT_14_15, BT_13_15, BT_11_14_15, BT_16_13
    10. Ratio_3_5 - 粒径判别
    11. Albedo_5_6 - 相态识别
    12. Tskin_BT13 - 地表温度-云顶温差
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
            cmask = self.memory_data["cmask"][idx]
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

            cmask_raw = self.h5_file["label_Cmask_max"][idx]
            cmask = cmask_raw[REFLECTIVITY_IDX_START:REFLECTIVITY_IDX_END]

        return {
            "input": torch.from_numpy(inputs).float(),
            "reflectivity": torch.from_numpy(reflectivity).float(),
            "cmask": torch.from_numpy(cmask).float(),
        }

    def close(self):
        if hasattr(self, "h5_file") and self.h5_file is not None:
            self.h5_file.close()

    def __del__(self):
        self.close()


# ============================================
# 模型类
# ============================================


class SimpleMLP(nn.Module):
    """简单MLP模型"""

    def __init__(self, in_dim=89, out_dim=85, dropout=0.1):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # 输入层
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 隐藏层1
        self.layer2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 隐藏层2
        self.layer3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 隐藏层3
        self.layer4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 输出层
        self.output_layer = nn.Linear(128, out_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        output = self.output_layer(x)
        return output


# ============================================
# 主训练流程
# ============================================

# ============================================
# 打印训练信息
# ============================================
print("=" * 60)
print("MLP模型训练 - CloudSat反射率预测")
print("=" * 60)
print(f"设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"\n输入维度: {IN_DIM} (AHI(16) + 几何(7) + BT扩展(12) + ERA5(r,t,54))")
print(f"损失函数: L1")

# ============================================
# 创建保存目录
# ============================================
save_dir = Path(SAVE_DIR)
save_dir.mkdir(parents=True, exist_ok=True)

# ============================================
# 加载数据集
# ============================================
print("\n" + "-" * 40)
print("加载数据集")
print("-" * 40)

train_dataset = CloudSatDataset(DATA_ROOT, "train", preload=True)
val_dataset = CloudSatDataset(DATA_ROOT, "val", preload=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# ============================================
# 创建模型
# ============================================
print("\n" + "-" * 40)
print("创建模型")
print("-" * 40)

model = SimpleMLP(in_dim=IN_DIM, out_dim=OUT_DIM, dropout=DROPOUT).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")

# ============================================
# 创建优化器和学习率调度器
# ============================================
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)

# ============================================
# 初始化损失函数
# ============================================
print("\n" + "-" * 40)
print("损失函数配置")
print("-" * 40)
print("  使用: L1损失 (nn.L1Loss)")

criterion = nn.L1Loss()

# ============================================
# 初始化训练状态
# ============================================
best_val_loss = float("inf")
patience_counter = 0
history = {
    "train_loss": [],
    "val_loss": [],
    "val_rmse": [],
    "lr": [],
}

# ============================================
# 打印训练配置
# ============================================
print("\n" + "-" * 40)
print("开始训练")
print("-" * 40)
print(f"Epochs: {NUM_EPOCHS}, LR: {LR}, Batch Size: {BATCH_SIZE}")
print(f"早停耐心值: {PATIENCE}")
print("=" * 60)

# ============================================
# 训练循环
# ============================================

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()

    # ----------------------------
    # 训练阶段
    # ----------------------------
    model.train()
    train_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch["input"].to(DEVICE)
        target = batch["reflectivity"].to(DEVICE)

        pred = model(inputs)

        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    avg_train_loss = train_loss / num_batches

    # ----------------------------
    # 验证阶段
    # ----------------------------
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(DEVICE)
            target = batch["reflectivity"].to(DEVICE)

            pred = model(inputs)

            loss = criterion(pred, target)

            val_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    avg_val_loss = val_loss / len(val_loader)

    # 计算RMSE
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    rmse = torch.sqrt(((all_preds - all_targets) ** 2).mean()).item()

    # ----------------------------
    # 学习率调度
    # ----------------------------
    scheduler.step()

    # ----------------------------
    # 记录历史
    # ----------------------------
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["val_rmse"].append(rmse)
    history["lr"].append(optimizer.param_groups[0]["lr"])

    # ----------------------------
    # 打印epoch总结
    # ----------------------------
    epoch_time = time.time() - epoch_start_time
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} 总结:")
    print(f"  训练Loss: {avg_train_loss:.4f}")
    print(f"  验证Loss: {avg_val_loss:.4f} (RMSE: {rmse:.4f})")
    print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"  时间: {epoch_time:.1f}s")

    # ----------------------------
    # 保存checkpoint
    # ----------------------------
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_val_loss,
        "history": history,
    }
    torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch + 1}.pt")

    # ----------------------------
    # 早停检查
    # ----------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(checkpoint, save_dir / "best_model.pt")
        print(f"  [OK] 验证损失改善: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"  [X] 验证损失未改善 ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\n早停触发，停止训练")
            break

# ============================================
# 训练完成
# ============================================
print("\n" + "=" * 60)
print("训练完成!")
print(f"最佳验证损失: {best_val_loss:.4f}")
print(f"模型已保存到: {save_dir}")
print("=" * 60)

plt.figure()
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve (L1 Loss)")
plt.legend()
plt.xlim(0, NUM_EPOCHS)
plt.grid()
plt.tight_layout()
plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches="tight")
plt.clf()
plt.close()
