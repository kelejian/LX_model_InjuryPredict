import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Union

HIC = np.array([i for i in range(1, 1500)])
Dmax = np.array([i*0.01 for i in range(0, 16501)])
Nij = np.array([i * 0.001 for i in range(0, 2001)])


def AIS_cal_head(
    HIC15: Union[float, np.ndarray], 
    ais_level: int = 3,
    prob_thresholds: list = [0.02, 0.05, 0.15, 0.4, 0.75]
) -> np.ndarray:
    """
    根据头 HIC15 值计算头部 AIS 等级。
    
    Args:
        HIC15 (Union[float, np.ndarray]): HIC15 值。
        ais_level (int): 使用的风险曲线 P(AIS≥n) 中的 n，范围 1-5。
        prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
            例如 [0.02, 0.05, 0.15, 0.4, 0.75] 表示:
            P < 0.02: AIS=0, 0.02≤P<0.05: AIS=1, 0.05≤P<0.15: AIS=2,
            0.15≤P<0.4: AIS=3, 0.4≤P<0.75: AIS=4, P≥0.75: AIS=5

    Returns:
        float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
    """
    if np.issubdtype(type(HIC15), np.number):
        is_single_value = True
    else:
        is_single_value = False 
    HIC15 = np.atleast_1d(HIC15)
    HIC15 = np.clip(HIC15, 1, 2500)

    # 定义系数
    coefficients = np.array([
        [1.54, 0.00650],  # P(AIS≥1)
        [2.49, 0.00483],  # P(AIS≥2)
        [3.39, 0.00372],  # P(AIS≥3)
        [4.90, 0.00351],  # P(AIS≥4)
        [7.82, 0.00429]   # P(AIS≥5)
    ])
    
    if ais_level < 1 or ais_level > 5:
        raise ValueError("头部 AIS 等级应在 1-5 之间")
    
    c1, c2 = coefficients[ais_level - 1]
    HIC_inv = 200 / HIC15
    prob = 1.0 / (1.0 + np.exp(c1 + HIC_inv - c2 * HIC15))

    # 根据概率阈值确定 AIS 等级
    AIS = np.zeros_like(HIC15, dtype=int)
    for i, thresh in enumerate(prob_thresholds):
        AIS = np.where(prob >= thresh, i + 1, AIS)

    if is_single_value:
        return AIS[0]
    return AIS

def AIS_cal_chest(
    Dmax: Union[float, np.ndarray], 
    ais_level: int = 3,
    prob_thresholds: list = [0.05, 0.1, 0.2, 0.3, 0.4]
) -> np.ndarray:
    """
    根据胸部压缩量 Dmax (mm) 计算胸部 AIS 等级。

    Args:
        Dmax (Union[float, np.ndarray]): 胸部最大压缩量 (mm)。
        ais_level (int): 使用的风险曲线 P(AIS≥n) 中的 n，范围 2-5。
        prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
            例如 [0.05, 0.1, 0.2, 0.3, 0.4] 表示:
            P < 0.05: AIS=0, 0.05≤P<0.1: AIS=1, 0.1≤P<0.2: AIS=2,
            0.2≤P<0.3: AIS=3, 0.3≤P<0.4: AIS=4, P≥0.4: AIS=5

    Returns:
        float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
    """
    if np.issubdtype(type(Dmax), np.number):
        is_single_value = True
    else:
        is_single_value = False
    Dmax = np.atleast_1d(Dmax)
    Dmax = np.clip(Dmax, 0.0, 500.0)

    coefficients = np.array([
        [1.8706, 0.04439],  # P(AIS≥2)
        [3.7124, 0.04750],  # P(AIS≥3)
        [5.0952, 0.04750],  # P(AIS≥4)
        [8.8274, 0.04590]   # P(AIS≥5)
    ])
    
    if ais_level < 2 or ais_level > 5:
        raise ValueError("胸部 AIS 等级应在 2-5 之间")
    
    c1, c2 = coefficients[ais_level - 2]
    prob = 1.0 / (1.0 + np.exp(c1 - c2 * Dmax))

    # 根据概率阈值确定 AIS 等级
    AIS = np.zeros_like(Dmax, dtype=int)
    for i, thresh in enumerate(prob_thresholds):
        AIS = np.where(prob >= thresh, i + 1, AIS)

    if is_single_value:
        return AIS[0]
    return AIS

def AIS_cal_neck(
    Nij: Union[float, np.ndarray], 
    ais_level: int = 3,
    prob_thresholds: list = [0.05, 0.1, 0.2, 0.3, 0.4]
) -> np.ndarray:
    """
    根据颈部伤害指数 Nij 计算颈部 AIS 等级。

    Args:
        Nij (Union[float, np.ndarray]): Nij 值。
        ais_level (int): 使用的风险曲线 P(AIS≥n) 中的 n，范围 2-5。
        prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
            例如 [0.05, 0.1, 0.2, 0.3, 0.4] 表示:
            P < 0.05: AIS=0, 0.05≤P<0.1: AIS=1, 0.1≤P<0.2: AIS=2,
            0.2≤P<0.3: AIS=3, 0.3≤P<0.4: AIS=4, P≥0.4: AIS=5

    Returns:
        float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
    """
    if np.issubdtype(type(Nij), np.number):
        is_single_value = True
    else:
        is_single_value = False
    Nij = np.atleast_1d(Nij)
    Nij = np.clip(Nij, 0, 50.0)

    coefficients = np.array([
        [2.054, 1.195],  # P(AIS≥2)
        [3.227, 1.969],  # P(AIS≥3)
        [2.693, 1.195],  # P(AIS≥4)
        [3.817, 1.195]   # P(AIS≥5)
    ])
    
    if ais_level < 2 or ais_level > 5:
        raise ValueError("颈部 AIS 等级应在 2-5 之间")
    
    c1, c2 = coefficients[ais_level - 2]
    prob = 1.0 / (1.0 + np.exp(c1 - c2 * Nij))

    # 根据概率阈值确定 AIS 等级
    AIS = np.zeros_like(Nij, dtype=int)
    for i, thresh in enumerate(prob_thresholds):
        AIS = np.where(prob >= thresh, i + 1, AIS)

    if is_single_value:
        return AIS[0]
    return AIS


def plot_ais_risk_curve(
    body_part: str,
    ais_level: int,
    prob_thresholds: list,
    figsize: tuple = (10, 7),
    save_path: str = None
):
    """
    绘制指定部位的 AIS 风险概率曲线 P(AIS≥n)，并标注指定概率值的交点。

    Args:
        body_part (str): 部位名称，可选 'head', 'chest', 'neck'
        ais_level (int): 指定的 AIS 等级 n（绘制 P(AIS≥n) 曲线）
        prob_thresholds (list): 概率阈值列表，如 [0.05, 0.1, 0.2, 0.3, 0.5]
        figsize (tuple): 图形大小
        save_path (str): 保存路径，若为 None 则不保存
    """
    # 根据部位选择参数
    if body_part.lower() == 'head':
        x_values = np.linspace(1, 2500, 2500)
        x_label = 'HIC15'
        title = f'Head Injury Risk Curve: P(AIS≥{ais_level})'
        # 头部系数 (AIS 1-5)
        coefficients = np.array([
            [1.54, 0.00650],  # P(AIS≥1)
            [2.49, 0.00483],  # P(AIS≥2)
            [3.39, 0.00372],  # P(AIS≥3)
            [4.90, 0.00351],  # P(AIS≥4)
            [7.82, 0.00429]   # P(AIS≥5)
        ])
        if ais_level < 1 or ais_level > 5:
            raise ValueError("头部 AIS 等级应在 1-5 之间")
        c1, c2 = coefficients[ais_level - 1]
        HIC_inv = 200 / x_values
        prob = 1.0 / (1.0 + np.exp(c1 + HIC_inv - c2 * x_values))
        
    elif body_part.lower() == 'chest':
        x_values = np.linspace(0, 165, 1650)
        x_label = 'Dmax (mm)'
        title = f'Chest Injury Risk Curve: P(AIS≥{ais_level})'
        # 胸部系数 (AIS 2-5)
        coefficients = np.array([
            [1.8706, 0.04439],  # P(AIS≥2)
            [3.7124, 0.04750],  # P(AIS≥3)
            [5.0952, 0.04750],  # P(AIS≥4)
            [8.8274, 0.04590]   # P(AIS≥5)
        ])
        if ais_level < 2 or ais_level > 5:
            raise ValueError("胸部 AIS 等级应在 2-5 之间")
        c1, c2 = coefficients[ais_level - 2]
        prob = 1.0 / (1.0 + np.exp(c1 - c2 * x_values))
        
    elif body_part.lower() == 'neck':
        x_values = np.linspace(0, 5, 500)
        x_label = 'Nij'
        title = f'Neck Injury Risk Curve: P(AIS≥{ais_level})'
        # 颈部系数 (AIS 2-5)
        coefficients = np.array([
            [2.054, 1.195],  # P(AIS≥2)
            [3.227, 1.969],  # P(AIS≥3)
            [2.693, 1.195],  # P(AIS≥4)
            [3.817, 1.195]   # P(AIS≥5)
        ])
        if ais_level < 2 or ais_level > 5:
            raise ValueError("颈部 AIS 等级应在 2-5 之间")
        c1, c2 = coefficients[ais_level - 2]
        prob = 1.0 / (1.0 + np.exp(c1 - c2 * x_values))
    else:
        raise ValueError("body_part 必须是 'head', 'chest' 或 'neck'")

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制风险曲线
    ax.plot(x_values, prob, 'b-', linewidth=2, label=f'P(AIS≥{ais_level})')
    
    # 绘制概率阈值水平线和交点
    colors = plt.cm.tab10(np.linspace(0, 1, len(prob_thresholds)))
    for i, p_thresh in enumerate(prob_thresholds):
        # 绘制水平线
        ax.axhline(y=p_thresh, color=colors[i], linestyle='--', alpha=0.7)
        
        # 找到交点（使用线性插值）
        idx = np.where(prob >= p_thresh)[0]
        if len(idx) > 0:
            first_idx = idx[0]
            if first_idx > 0:
                # 线性插值求交点
                x_cross = np.interp(p_thresh, [prob[first_idx-1], prob[first_idx]], 
                                    [x_values[first_idx-1], x_values[first_idx]])
            else:
                x_cross = x_values[first_idx]
            
            # 绘制交点
            ax.plot(x_cross, p_thresh, 'o', color=colors[i], markersize=10)
            
            # 标注交点坐标（字体放大）
            ax.annotate(f'({x_cross:.3f}, {p_thresh:.2f})', 
                        xy=(x_cross, p_thresh),
                        xytext=(15, 10), textcoords='offset points',
                        fontsize=12, fontweight='bold',
                        color=colors[i],
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # 设置图形属性
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Probability', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, ax


# 使用示例
if __name__ == "__main__":

    # ========== 测试 AIS 计算函数 ==========
    print("=" * 50)
    print("测试 AIS 计算函数")
    print("=" * 50)
    
    # 测试头部 AIS 计算 (使用 P(AIS>=3) 曲线)
    # 根据图中交点: HIC15 ≈ 519 -> P=0.02, 632 -> P=0.05, 830 -> P=0.15, 1116 -> P=0.4, 1457 -> P=0.75
    print("\n--- 头部 AIS 计算测试 (ais_level=3) ---")
    test_hic_values = [100,200, 500, 600, 700, 900, 1200, 1500]
    expected_head_ais = [0, 1, 2, 3, 3, 4, 4, 5]
    for hic, expected in zip(test_hic_values, expected_head_ais):
        result = AIS_cal_head(hic, ais_level=3, prob_thresholds=[0.02, 0.05, 0.15, 0.4, 0.75])
        status = "✓" if result == expected else "✗"
        print(f"HIC15={hic}: AIS={result} (期望={expected}) {status}")
    
    # 测试数组输入
    hic_array = np.array([100,200, 500, 600, 700, 900, 1200, 1500])
    result_array = AIS_cal_head(hic_array, ais_level=3, prob_thresholds=[0.02, 0.05, 0.15, 0.4, 0.75])
    print(f"数组测试: HIC15={hic_array} -> AIS={result_array}")
    
    # 测试胸部 AIS 计算 (使用 P(AIS>=3) 曲线)
    # 根据图中交点: Dmax ≈ 16 -> P=0.05, 32 -> P=0.1, 49 -> P=0.2, 62 -> P=0.3, 74 -> P=0.4
    print("\n--- 胸部 AIS 计算测试 (ais_level=3) ---")
    test_dmax_values = [10, 20, 40, 55,62, 70, 80]
    expected_chest_ais = [0, 1, 2, 3, 4, 5, 5]
    for dmax, expected in zip(test_dmax_values, expected_chest_ais):
        result = AIS_cal_chest(dmax, ais_level=3, prob_thresholds=[0.05, 0.1, 0.2, 0.3, 0.4])
        status = "✓" if result == expected else "✗"
        print(f"Dmax={dmax}mm: AIS={result} (期望={expected}) {status}")
    dmax_array = np.array([10, 20, 40, 55,62, 70, 80])
    result_array = AIS_cal_chest(dmax_array, ais_level=3, prob_thresholds=[0.05, 0.1, 0.2, 0.3, 0.4])
    print(f"数组测试: Dmax={dmax_array}mm -> AIS={result_array}")
    
    # 测试颈部 AIS 计算 (使用 P(AIS>=3) 曲线)
    # 根据图中交点: Nij ≈ 0.15 -> P=0.05, 0.70 -> P=0.1, 1.15 -> P=0.2, 1.45 -> P=0.3, 1.68 -> P=0.4
    print("\n--- 颈部 AIS 计算测试 (ais_level=3) ---")
    test_nij_values = [0.1, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
    expected_neck_ais = [0, 1, 2, 3, 4, 5, 5]
    for nij, expected in zip(test_nij_values, expected_neck_ais):
        result = AIS_cal_neck(nij, ais_level=3, prob_thresholds=[0.05, 0.1, 0.2, 0.3, 0.4])
        status = "✓" if result == expected else "✗"
        print(f"Nij={nij}: AIS={result} (期望={expected}) {status}")
    nij_array = np.array([0.1, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0])
    result_array = AIS_cal_neck(nij_array, ais_level=3, prob_thresholds=[0.05, 0.1, 0.2, 0.3, 0.4])
    print(f"数组测试: Nij={nij_array} -> AIS={result_array}")
    
    print("\n" + "=" * 50)
    print("绘制风险曲线")
    print("=" * 50)

    # 绘制头部 P(AIS>=3) 曲线
    plot_ais_risk_curve('head', ais_level=3, prob_thresholds=[0.02, 0.05, 0.15, 0.4, 0.75]) # 3 [0.02, 0.05, 0.15, 0.4, 0.75], 分别以这些概率为阈值，找出对应的 HIC15 值，作为AIS分级阈值，0~0.02: AIS0, 0.02~0.05: AIS=1, 0.05~0.15: AIS=2, 0.15~0.4: AIS=3, 0.4~0.75: AIS=4, 0.75~1.0: AIS=5
    
    # 绘制胸部 P(AIS>=3) 曲线
    plot_ais_risk_curve('chest', ais_level=3, prob_thresholds=[0.05, 0.1, 0.2, 0.3, 0.4]) # 3 [0.05, 0.1, 0.2, 0.3, 0.4], 分别以这些概率为阈值，找出对应的 Dmax 值，作为AIS分级阈值，0~0.05: AIS0, 0.05~0.1: AIS=1, 0.1~0.2: AIS=2, 0.2~0.3: AIS=3, 0.3~0.4: AIS=4, 0.4~1.0: AIS=5
    
    # 绘制颈部 P(AIS>=3) 曲线
    plot_ais_risk_curve('neck', ais_level=3, prob_thresholds=[0.05, 0.1, 0.2, 0.3, 0.4])  # 3 [0.05, 0.1, 0.2, 0.3, 0.4], 分别以这些概率为阈值，找出对应的 Nij 值，作为AIS分级阈值，0~0.05: AIS0, 0.05~0.1: AIS=1, 0.1~0.2: AIS=2, 0.2~0.3: AIS=3, 0.3~0.4: AIS=4, 0.4~1.0: AIS=5

