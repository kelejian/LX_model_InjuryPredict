import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Union

HIC15 = np.array([i for i in range(1, 1500)])
Dmax = np.array([i*0.01 for i in range(0, 8000)])
Nij = np.array([i * 0.001 for i in range(0, 1500)])


def AIS_cal_head(
    HIC15: Union[float, np.ndarray], 
    prob_thresholds: list = [0.01, 0.05, 0.1, 0.2, 0.3]
) -> np.ndarray:
    """
    根据头部 HIC15 值计算头部 AIS 等级。
    使用公式: P_head(AIS3+) = Φ((ln(HIC15) - 7.45231) / 0.73998)
    其中 Φ 是累积正态分布函数
    
    Args:
        HIC15 (Union[float, np.ndarray]): HIC15 值。
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
    HIC15 = np.atleast_1d(HIC15).astype(float)
    HIC15 = np.clip(HIC15, 1, 2500)

    # P_head(AIS3+) = Φ((ln(HIC15) - 7.45231) / 0.73998)
    z = (np.log(HIC15) - 7.45231) / 0.73998
    prob = norm.cdf(z)

    # 根据概率阈值确定 AIS 等级
    AIS = np.zeros_like(HIC15, dtype=int)
    for i, thresh in enumerate(prob_thresholds):
        AIS = np.where(prob >= thresh, i + 1, AIS)

    if is_single_value:
        return AIS[0]
    return AIS


def AIS_cal_chest(
    Dmax: Union[float, np.ndarray], 
    OT: Union[int, np.ndarray],
    prob_thresholds: list = [0.02, 0.06, 0.15, 0.25, 0.4]
) -> np.ndarray:
    """
    根据胸部压缩量 Dmax (mm) 计算胸部 AIS 等级。
    使用公式: P_chest_defl(AIS3+) = 1 / (1 + e^(10.5456 - 1.568 * ChestDefl^0.4612))

    Args:
        Dmax (Union[float, np.ndarray]): 胸部最大压缩量 (mm)。
        OT (Union[int, np.ndarray],) : 假人类别, 1为5th 女性, 2为50th男性, 3为95th男性
        prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
            例如 [0.06, 0.1, 0.2, 0.3, 0.4] 表示:
            P < 0.06: AIS=0, 0.06≤P<0.1: AIS=1, 0.1≤P<0.2: AIS=2,
            0.2≤P<0.3: AIS=3, 0.3≤P<0.4: AIS=4, P≥0.4: AIS=5

    Returns:
        float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
    """
    if np.issubdtype(type(Dmax), np.number):
        is_single_value = True
    else:
        is_single_value = False
    Dmax = np.atleast_1d(Dmax).astype(float)
    Dmax = np.clip(Dmax, 0.0, 500.0)

    # OT= 1时，Scaling_Factor=221/182.9; OT=2时，Scaling_Factor=1.0; OT=3时，Scaling_Factor=221/246.38
    Scaling_Factor = np.where(OT == 1, 221.0 / 182.9, 
                              np.where(OT == 2, 1.0,
                                       np.where(OT == 3, 221.0 / 246.38, 1.0)))
    # 根据缩放因子调整 Dmax
    Dmax_eq = Dmax * Scaling_Factor
        
    # P_chest_defl(AIS3+) = 1 / (1 + e^(10.5456 - 1.568 * ChestDefl^0.4612))
    prob = 1.0 / (1.0 + np.exp(10.5456 - 1.568 * np.power(Dmax_eq, 0.4612)))
    # 根据概率阈值确定 AIS 等级
    AIS = np.zeros_like(Dmax_eq, dtype=int)
    for i, thresh in enumerate(prob_thresholds):
        AIS = np.where(prob >= thresh, i + 1, AIS)

    if is_single_value:
        return AIS[0]
    return AIS


def AIS_cal_neck(
    Nij: Union[float, np.ndarray], 
    prob_thresholds: list = [0.06, 0.1, 0.2, 0.3, 0.4]
) -> np.ndarray:
    """
    根据颈部伤害指数 Nij 计算颈部 AIS 等级。
    使用公式: P_neck_Nij(AIS3+) = 1 / (1 + e^(3.2269 - 1.9688 * Nij))

    Args:
        Nij (Union[float, np.ndarray]): Nij 值。
        prob_thresholds (list): 概率阈值列表，用于划分 AIS 等级。
            例如 [0.06, 0.1, 0.2, 0.3, 0.4] 表示:
            P < 0.06: AIS=0, 0.06≤P<0.1: AIS=1, 0.1≤P<0.2: AIS=2,
            0.2≤P<0.3: AIS=3, 0.3≤P<0.4: AIS=4, P≥0.4: AIS=5

    Returns:
        float or np.ndarray: 计算出的 AIS 等级, 和输入的损伤值形状一致。
    """
    if np.issubdtype(type(Nij), np.number):
        is_single_value = True
    else:
        is_single_value = False
    Nij = np.atleast_1d(Nij).astype(float)
    Nij = np.clip(Nij, 0, 50.0)

    # P_neck_Nij(AIS3+) = 1 / (1 + e^(3.2269 - 1.9688 * Nij))
    prob = 1.0 / (1.0 + np.exp(3.2269 - 1.9688 * Nij))

    # 根据概率阈值确定 AIS 等级
    AIS = np.zeros_like(Nij, dtype=int)
    for i, thresh in enumerate(prob_thresholds):
        AIS = np.where(prob >= thresh, i + 1, AIS)

    if is_single_value:
        return AIS[0]
    return AIS


def plot_ais_risk_curve(
    body_part: str,
    prob_thresholds: list,
    figsize: tuple = (10, 7),
    save_path: str = None
):
    """
    绘制指定部位的 AIS 风险概率曲线 P(AIS≥3+)，并标注指定概率值的交点。

    Args:
        body_part (str): 部位名称，可选 'head', 'chest', 'neck'
        prob_thresholds (list): 概率阈值列表，如 [0.05, 0.1, 0.2, 0.3, 0.5]
        figsize (tuple): 图形大小
        save_path (str): 保存路径，若为 None 则不保存
    """
    # 根据部位选择参数
    if body_part.lower() == 'head':
        x_values = HIC15
        x_label = 'HIC15'
        title = 'Head Injury Risk Curve: P(AIS≥3+)'
        # P_head(AIS3+) = Φ((ln(HIC15) - 7.45231) / 0.73998)
        z = (np.log(x_values) - 7.45231) / 0.73998
        prob = norm.cdf(z)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_values, prob, 'b-', linewidth=2, label='Head')
        
        # 绘制概率阈值水平线和交点
        colors = plt.cm.tab10(np.linspace(0, 1, len(prob_thresholds)))
        for i, p_thresh in enumerate(prob_thresholds):
            ax.axhline(y=p_thresh, color=colors[i], linestyle='--', alpha=0.7)
            # 在虚线左侧标注概率值
            ax.annotate(f'{p_thresh:.3f}', 
                        xy=(x_values[0], p_thresh),
                        xytext=(-5, 0), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        color=colors[i], ha='right', va='center')
            
            idx = np.where(prob >= p_thresh)[0]
            if len(idx) > 0:
                first_idx = idx[0]
                if first_idx > 0:
                    x_cross = np.interp(p_thresh, [prob[first_idx-1], prob[first_idx]], 
                                        [x_values[first_idx-1], x_values[first_idx]])
                else:
                    x_cross = x_values[first_idx]
                ax.plot(x_cross, p_thresh, 'o', color=colors[i], markersize=10)
                ax.annotate(f'{x_cross:.1f}', 
                            xy=(x_cross, p_thresh),
                            xytext=(5, 10), textcoords='offset points',
                            fontsize=11, fontweight='bold',
                            color=colors[i],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.legend(fontsize=12, loc='lower right')
        
    elif body_part.lower() == 'chest':
        x_values = Dmax
        x_label = 'Dmax (mm)'
        title = 'Chest Injury Risk Curve: P(AIS≥3+)'
        
        # 三种 OT 的缩放因子
        ot_configs = [
            {'OT': 1, 'name': '5th Female', 'scale': 221.0 / 182.9, 'color': 'r', 'linestyle': '-'},
            {'OT': 2, 'name': '50th Male', 'scale': 1.0, 'color': 'b', 'linestyle': '-'},
            {'OT': 3, 'name': '95th Male', 'scale': 221.0 / 246.38, 'color': 'g', 'linestyle': '-'}
        ]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 概率阈值颜色
        thresh_colors = plt.cm.Greys(np.linspace(0.4, 0.7, len(prob_thresholds)))
        
        # 先绘制所有水平阈值线并标注概率值
        for i, p_thresh in enumerate(prob_thresholds):
            ax.axhline(y=p_thresh, color=thresh_colors[i], linestyle='--', alpha=0.7, linewidth=1)
            # 在虚线左侧标注概率值
            ax.annotate(f'{p_thresh:.2f}', 
                        xy=(x_values[0], p_thresh),
                        xytext=(-5, 0), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        color=thresh_colors[i], ha='right', va='center')
        
        # 绘制三条曲线及交点
        for ot_cfg in ot_configs:
            scale = ot_cfg['scale']
            # 等效 Dmax
            Dmax_eq = x_values * scale
            prob = 1.0 / (1.0 + np.exp(10.5456 - 1.568 * np.power(Dmax_eq, 0.4612)))
            
            # 绘制曲线
            ax.plot(x_values, prob, color=ot_cfg['color'], linestyle=ot_cfg['linestyle'], 
                    linewidth=2, label=f"{ot_cfg['name']}")
            
            # 绘制与阈值的交点
            for i, p_thresh in enumerate(prob_thresholds):
                idx = np.where(prob >= p_thresh)[0]
                if len(idx) > 0:
                    first_idx = idx[0]
                    if first_idx > 0:
                        x_cross = np.interp(p_thresh, [prob[first_idx-1], prob[first_idx]], 
                                            [x_values[first_idx-1], x_values[first_idx]])
                    else:
                        x_cross = x_values[first_idx]
                    
                    # 绘制交点
                    ax.plot(x_cross, p_thresh, 'o', color=ot_cfg['color'], markersize=8)
                    
                    # 标注横坐标
                    offset_y = 5
                    ax.annotate(f'{x_cross:.1f}', 
                                xy=(x_cross, p_thresh),
                                xytext=(3, offset_y), textcoords='offset points',
                                fontsize=11, fontweight='bold',
                                color=ot_cfg['color'],
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax.legend(fontsize=12, loc='upper left')
        
    elif body_part.lower() == 'neck':
        x_values = Nij
        x_label = 'Nij'
        title = 'Neck Injury Risk Curve: P(AIS≥3+)'
        # P_neck_Nij(AIS3+) = 1 / (1 + e^(3.2269 - 1.9688 * Nij))
        prob = 1.0 / (1.0 + np.exp(3.2269 - 1.9688 * x_values))
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_values, prob, 'b-', linewidth=2, label='Neck')
        
        # 绘制概率阈值水平线和交点
        colors = plt.cm.tab10(np.linspace(0, 1, len(prob_thresholds)))
        for i, p_thresh in enumerate(prob_thresholds):
            ax.axhline(y=p_thresh, color=colors[i], linestyle='--', alpha=0.7)
            # 在虚线左侧标注概率值
            ax.annotate(f'{p_thresh:.2f}', 
                        xy=(x_values[0], p_thresh),
                        xytext=(-5, 0), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        color=colors[i], ha='right', va='center')
            
            idx = np.where(prob >= p_thresh)[0]
            if len(idx) > 0:
                first_idx = idx[0]
                if first_idx > 0:
                    x_cross = np.interp(p_thresh, [prob[first_idx-1], prob[first_idx]], 
                                        [x_values[first_idx-1], x_values[first_idx]])
                else:
                    x_cross = x_values[first_idx]
                ax.plot(x_cross, p_thresh, 'o', color=colors[i], markersize=10)
                ax.annotate(f'{x_cross:.3f}', 
                            xy=(x_cross, p_thresh),
                            xytext=(5, 10), textcoords='offset points',
                            fontsize=11, fontweight='bold',
                            color=colors[i],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.legend(fontsize=12, loc='lower right')
        
    else:
        raise ValueError("body_part 必须是 'head', 'chest' 或 'neck'")

    # 设置图形属性
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Probability', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig, ax


# 使用示例
if __name__ == "__main__":
    # 绘制头部 P(AIS>=3+) 曲线
    plot_ais_risk_curve('head', prob_thresholds=[0.01, 0.05, 0.1, 0.2, 0.3])
    
    # 绘制胸部 P(AIS>=3+) 曲线
    plot_ais_risk_curve('chest', prob_thresholds=[0.02, 0.06, 0.15, 0.25, 0.4])
    
    # 绘制颈部 P(AIS>=3+) 曲线
    plot_ais_risk_curve('neck', prob_thresholds=[0.06, 0.1, 0.2, 0.3, 0.4])

