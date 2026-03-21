"""
visualize.py - SG-ABM仿真结果可视化脚本

该脚本用于加载仿真生成的 Excel 结果文件，并基于预设的样式库生成专业的可视化图表。
支持折线图、雷达图、组合图等多种展示形式，直观展现市场关键指标的演化轨迹。

作者：Liang
日期：2026年
版本：1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置字体格式和大小为 22
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 22

def load_data(sheet_name='时间序列数据'):
    """从本地 data 目录或 batch_results 加载最新数据"""
    data_path = Path("data/latest_results.xlsx")
    
    if not data_path.exists():
        # 尝试在 batch_results 中寻找最新的文件
        batch_dir = Path("batch_results")
        if batch_dir.exists():
            files = list(batch_dir.glob("**/*.xlsx"))
            if files:
                # 按修改时间排序
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                data_path = files[0]
                print(f"[INFO] 检测到最新仿真结果: {data_path}")
    
    if not data_path.exists():
        print(f"[ERROR] 未找到数据文件: {data_path}")
        print("[TIPS] 请先执行 run_simulation.py 脚本以生成仿真数据。")
        return pd.DataFrame()
        
    return pd.read_excel(data_path, sheet_name=sheet_name)

def normalize_series(series):
    """归一化到 0-1 范围"""
    min_val, max_val = series.min(), series.max()
    return (series - min_val) / (max_val - min_val) if max_val > min_val else series * 0 + 0.5

def plot_time_series():
    """绘制 50 年多指标趋势图 (Origami 风格)"""
    df = load_data('时间序列数据')
    if df.empty: return
    
    # 计算派生指标
    total_capacity_mw = 343901.5
    potential_gen = (total_capacity_mw * 0.1) * 8760 / 10000
    df['penetration_rate'] = (df['green_power_volume'] / df['total_green_demand']) * 100
    df['capacity_utilization'] = (df['green_power_volume'] / 10000) / potential_gen * 100
    df['cumulative_policy_cost'] = df.groupby('scenario_id')['total_policy_cost'].cumsum()

    target_scenario = 'S3'
    scenario_df = df[df['scenario_id'] == target_scenario].sort_values('year')
    
    metrics = {
        'green_power_volume': '绿电交易量', 'carbon_reduction': '碳减排量',
        'cumulative_policy_cost': '累计政策成本', 'clearing_price': '平均出清价格',
        'penetration_rate': '绿电渗透率', 'capacity_utilization': '产能利用率'
    }
    colors = ['#f57c6e', '#f2b56e', '#fbe79e', '#84c3b7', '#88d7da', '#71b8ed']

    fig, ax = plt.subplots(figsize=(16, 10))
    years = scenario_df['year'].values
    
    for idx, (col, label) in enumerate(metrics.items()):
        vals = normalize_series(scenario_df[col])
        color = colors[idx]
        
        # 填充与线条
        ax.add_patch(Polygon(np.column_stack([np.concatenate([[years[0]], years, [years[-1]]]), 
                                             np.concatenate([[0], vals, [0]])]), 
                             facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.15))
        ax.plot(years, vals, color=color, linewidth=5, zorder=idx+10)
        
        # 内嵌标签
        mid_idx = len(years) // 2
        ax.text(years[mid_idx], vals.iloc[mid_idx], label, ha='center', va='center', 
                fontsize=20, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel('时间 (年份)', fontsize=22, fontweight='bold')
    ax.set_ylim(-0.05, 1.1)
    ax.set_yticks([])
    ax.spines[['left', 'right', 'top']].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/origami_time_series.png", dpi=300)
    print("[INFO] 已生成: outputs/origami_time_series.png")

# ... 类似逻辑可以添加 radar 和 line chart 整合 ...

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    plot_time_series()
    # 更多绘图函数调用...
