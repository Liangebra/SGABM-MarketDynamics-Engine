"""
SG-ABM模型情景分析与可视化模块

本模块负责仿真结果的可视化分析、情景对比和网络可视化。
包含以下主要类：
1. ScenarioManager: 情景管理器，管理不同政策情景的配置和切换
2. VisualizationGenerator: 可视化生成器，生成各种分析图表
3. DynamicNetworkVisualizer: 动态网络可视化器，生成主体交互网络

作者：Liang
日期：2026年
版本：1.0
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx

# 导入已有模块
from core import base, fun, math_go
from utils.input import DataLoader, ResultSaver

# 设置中文字体和绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rcParams.update({'figure.autolayout': True})

# 设置日志
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ScenarioSummary:
    """情景摘要"""
    scenario_id: str
    scenario_name: str
    start_year: int
    end_year: int
    total_years: int
    key_metrics: Dict[str, float]
    policy_tools: Dict[str, float]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ComparisonResult:
    """情景比较结果"""
    scenarios: List[str]
    comparison_metrics: Dict[str, Dict[str, float]]
    ranking: Dict[str, List[Tuple[str, float]]]
    summary: Dict[str, Any]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class NetworkData:
    """网络数据"""
    year: int
    scenario_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    adjacency_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class NetworkMetrics:
    """网络指标"""
    year: int
    num_nodes: int
    num_edges: int
    density: float
    avg_degree: float
    clustering_coefficient: float
    degree_centrality: Dict[str, float]
    betweenness_centrality: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# ============================================================================
# ScenarioManager 类 - 情景管理器
# ============================================================================

class ScenarioManager:
    """情景管理器"""

    def __init__(self, data_dir: str = "./data", output_dir: str = "./output"):
        """
        初始化情景管理器

        Args:
            data_dir: 数据目录路径
            output_dir: 输出目录路径
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.data_loader = DataLoader(data_dir)
        self.result_saver = ResultSaver(output_dir)

        self.scenarios = {}  # 存储所有情景配置
        self.current_scenario = None  # 当前激活情景
        self.results = {}  # 各情景仿真结果

        # 加载所有情景
        self._load_all_scenarios()

        logger.info("情景管理器初始化完成")

    def _load_all_scenarios(self) -> None:
        """加载所有情景配置"""
        self.scenarios = self.data_loader.load_all_scenarios()

        # 创建情景结果目录
        for scenario_id in self.scenarios.keys():
            scenario_dir = self.output_dir / "scenario_results" / scenario_id
            scenario_dir.mkdir(parents=True, exist_ok=True)

    def load_scenario(self, scenario_id: str) -> bool:
        """
        加载指定情景配置

        Args:
            scenario_id: 情景ID

        Returns:
            bool: 是否加载成功
        """
        if scenario_id not in self.scenarios:
            # 尝试从文件加载
            scenario_config = self.data_loader.load_scenario_config(scenario_id)
            if not scenario_config:
                logger.error(f"情景 {scenario_id} 不存在")
                return False

            self.scenarios[scenario_id] = scenario_config

        self.current_scenario = scenario_id
        logger.info(f"已加载情景: {scenario_id}")
        return True

    def run_scenario(self, scenario_id: str, years: int = 50,
                     random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        运行指定情景仿真

        Args:
            scenario_id: 情景ID
            years: 仿真年数
            random_seed: 随机种子

        Returns:
            Dict: 仿真结果
        """
        logger.info(f"开始运行情景 {scenario_id}, 年数: {years}")

        if scenario_id not in self.scenarios:
            self.load_scenario(scenario_id)

        # 设置随机种子
        if random_seed is None:
            random_seed = hash(scenario_id) % 10000

        try:
            # 调用math_go中的仿真函数
            start_time = datetime.now()

            result = math_go.run_scenario_simulation(
                scenario_id=scenario_id,
                years=years,
                random_seed=random_seed
            )

            # 计算执行时间
            elapsed_time = (datetime.now() - start_time).total_seconds()
            result['execution_time'] = elapsed_time
            result['random_seed'] = random_seed

            # 保存结果
            self._save_scenario_results(result, scenario_id)

            # 存储到内存
            self.results[scenario_id] = result

            logger.info(f"情景 {scenario_id} 仿真完成，耗时: {elapsed_time:.2f}秒")
            return result

        except Exception as e:
            logger.error(f"运行情景 {scenario_id} 失败: {str(e)}")
            raise

    def _save_scenario_results(self, result: Dict[str, Any], scenario_id: str) -> None:
        """
        保存情景结果

        Args:
            result: 仿真结果
            scenario_id: 情景ID
        """
        # 保存各种数据
        if 'time_series' in result:
            self.result_saver.save_simulation_results(result, scenario_id)

        if 'agents' in result:
            self.result_saver.save_agent_data(result['agents'], scenario_id)

        if 'policy_evolution' in result:
            self.result_saver.save_policy_evolution(result['policy_evolution'], scenario_id)

        if 'market_data' in result:
            self.result_saver.save_market_data(result['market_data'], scenario_id)

        if 'evaluation_metrics' in result:
            self.result_saver.save_evaluation_metrics(result['evaluation_metrics'], scenario_id)

        # 保存网络数据（如果有）
        if 'network_data' in result:
            for year, network_data in result['network_data'].items():
                self.result_saver.save_network_data(network_data, year, scenario_id)

    def compare_scenarios(self, scenario_ids: List[str]) -> ComparisonResult:
        """
        比较多个情景结果

        Args:
            scenario_ids: 情景ID列表

        Returns:
            ComparisonResult: 比较结果
        """
        logger.info(f"开始比较情景: {scenario_ids}")

        # 确保所有情景都已加载结果
        for scenario_id in scenario_ids:
            if scenario_id not in self.results:
                # 尝试从文件加载结果
                result = self._load_scenario_results(scenario_id)
                if result:
                    self.results[scenario_id] = result
                else:
                    logger.warning(f"情景 {scenario_id} 的结果不存在，将运行该情景")
                    self.run_scenario(scenario_id)

        # 提取比较指标
        comparison_metrics = {}

        # 定义要比较的关键指标
        key_metrics = [
            'total_carbon_reduction',
            'total_policy_cost',
            'avg_green_price',
            'avg_penetration_rate',
            'final_social_welfare',
            'avg_capacity_utilization',
            'avg_roe_green_energy',
            'avg_roe_consumer',
            'cost_per_carbon_reduction'
        ]

        for scenario_id in scenario_ids:
            if scenario_id in self.results:
                result = self.results[scenario_id]
                metrics = {}

                # 从评估指标中提取
                if 'evaluation_metrics' in result:
                    for metric in key_metrics:
                        if metric in result['evaluation_metrics']:
                            metrics[metric] = result['evaluation_metrics'][metric]

                # 从摘要中提取
                elif 'summary' in result:
                    for metric in key_metrics:
                        if metric in result['summary']:
                            metrics[metric] = result['summary'][metric]

                # 如果都没有，尝试从时间序列计算
                else:
                    metrics = self._calculate_metrics_from_time_series(scenario_id)

                comparison_metrics[scenario_id] = metrics

        # 生成排名
        ranking = self._generate_ranking(comparison_metrics)

        # 生成摘要
        summary = self._generate_comparison_summary(comparison_metrics, ranking)

        # 创建比较结果对象
        comparison_result = ComparisonResult(
            scenarios=scenario_ids,
            comparison_metrics=comparison_metrics,
            ranking=ranking,
            summary=summary,
            created_at=datetime.now()
        )

        # 保存比较结果
        self._save_comparison_result(comparison_result, scenario_ids)

        logger.info(f"情景比较完成，共比较 {len(scenario_ids)} 个情景")
        return comparison_result

    def _load_scenario_results(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载情景结果

        Args:
            scenario_id: 情景ID

        Returns:
            Optional[Dict]: 仿真结果，如果不存在则返回None
        """
        # 查找最新的结果文件
        results_dir = self.output_dir / "simulation_results"

        if not results_dir.exists():
            return None

        pattern = f"simulation_results_{scenario_id}_*.json"
        result_files = list(results_dir.glob(pattern))

        if not result_files:
            return None

        # 选择最新的文件
        result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = result_files[0]

        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return result
        except Exception as e:
            logger.error(f"加载情景结果失败 {scenario_id}: {str(e)}")
            return None

    def _calculate_metrics_from_time_series(self, scenario_id: str) -> Dict[str, float]:
        """
        从时间序列数据计算指标

        Args:
            scenario_id: 情景ID

        Returns:
            Dict[str, float]: 计算出的指标
        """
        if scenario_id not in self.results:
            return {}

        result = self.results[scenario_id]

        if 'time_series' not in result:
            return {}

        time_series = result['time_series']

        # 提取时间序列数据
        years = list(time_series.keys())
        if not years:
            return {}

        # 初始化指标字典
        metrics = {}

        # 计算累计指标
        total_carbon_reduction = sum(
            data.get('carbon_reduction', 0) for data in time_series.values()
        )
        total_policy_cost = sum(
            data.get('policy_cost', 0) for data in time_series.values()
        )

        metrics['total_carbon_reduction'] = total_carbon_reduction
        metrics['total_policy_cost'] = total_policy_cost

        # 计算平均指标
        avg_green_price = np.mean([
            data.get('avg_green_price', 0) for data in time_series.values()
        ])
        avg_penetration_rate = np.mean([
            data.get('penetration_rate', 0) for data in time_series.values()
        ])

        metrics['avg_green_price'] = float(avg_green_price)
        metrics['avg_penetration_rate'] = float(avg_penetration_rate)

        # 最终年份的社会福利
        final_year = max(years)
        if final_year in time_series:
            metrics['final_social_welfare'] = time_series[final_year].get('social_welfare', 0)

        # 计算单位碳减排成本
        if total_carbon_reduction > 0:
            metrics['cost_per_carbon_reduction'] = total_policy_cost / total_carbon_reduction
        else:
            metrics['cost_per_carbon_reduction'] = 0

        return metrics

    def _generate_ranking(self, comparison_metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        生成指标排名

        Args:
            comparison_metrics: 比较指标

        Returns:
            Dict[str, List[Tuple[str, float]]]: 各指标的排名
        """
        ranking = {}

        # 定义指标的排序方向（asc: 升序，desc: 降序）
        ranking_directions = {
            'total_carbon_reduction': 'desc',
            'total_policy_cost': 'asc',
            'avg_green_price': 'asc',
            'avg_penetration_rate': 'desc',
            'final_social_welfare': 'desc',
            'avg_capacity_utilization': 'desc',
            'avg_roe_green_energy': 'desc',
            'avg_roe_consumer': 'desc',
            'cost_per_carbon_reduction': 'asc'
        }

        # 收集所有指标
        all_metrics = set()
        for metrics in comparison_metrics.values():
            all_metrics.update(metrics.keys())

        # 为每个指标生成排名
        for metric in all_metrics:
            if metric not in ranking_directions:
                continue

            direction = ranking_directions[metric]
            metric_values = {}

            for scenario_id, metrics in comparison_metrics.items():
                if metric in metrics:
                    metric_values[scenario_id] = metrics[metric]

            if not metric_values:
                continue

            # 排序
            if direction == 'desc':
                sorted_items = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            else:
                sorted_items = sorted(metric_values.items(), key=lambda x: x[1])

            ranking[metric] = sorted_items

        return ranking

    def _generate_comparison_summary(self, comparison_metrics: Dict[str, Dict[str, float]],
                                     ranking: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Any]:
        """
        生成比较摘要

        Args:
            comparison_metrics: 比较指标
            ranking: 指标排名

        Returns:
            Dict[str, Any]: 比较摘要
        """
        summary = {
            'total_scenarios': len(comparison_metrics),
            'comparison_date': datetime.now().isoformat(),
            'best_performers': {},
            'worst_performers': {},
            'metric_statistics': {}
        }

        # 找出每个指标的最佳和最差表现者
        for metric, ranked_list in ranking.items():
            if ranked_list:
                summary['best_performers'][metric] = ranked_list[0][0]  # 第一名
                summary['worst_performers'][metric] = ranked_list[-1][0]  # 最后一名

        # 计算各指标的统计量
        for metric in ranking.keys():
            values = []
            for scenario_id, metrics in comparison_metrics.items():
                if metric in metrics:
                    values.append(metrics[metric])

            if values:
                summary['metric_statistics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values))
                }

        return summary

    def _save_comparison_result(self, comparison_result: ComparisonResult,
                                scenario_ids: List[str]) -> None:
        """
        保存比较结果

        Args:
            comparison_result: 比较结果
            scenario_ids: 情景ID列表
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_str = "_".join(scenario_ids)
        filename = f"comparison_{scenario_str}_{timestamp}.json"
        filepath = self.output_dir / "comparison_results" / filename

        # 确保目录存在
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 保存为JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison_result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"比较结果已保存: {filepath}")

    def get_scenario_summary(self, scenario_id: str) -> ScenarioSummary:
        """
        获取情景摘要报告

        Args:
            scenario_id: 情景ID

        Returns:
            ScenarioSummary: 情景摘要
        """
        if scenario_id not in self.results:
            # 尝试加载结果
            result = self._load_scenario_results(scenario_id)
            if result:
                self.results[scenario_id] = result
            else:
                raise ValueError(f"情景 {scenario_id} 的结果不存在")

        result = self.results[scenario_id]

        # 提取关键指标
        key_metrics = {}
        if 'evaluation_metrics' in result:
            key_metrics = result['evaluation_metrics']
        elif 'summary' in result:
            key_metrics = result['summary']

        # 提取政策工具
        policy_tools = {}
        if scenario_id in self.scenarios:
            policy_tools = self.scenarios[scenario_id]

        # 确定起始和结束年份
        start_year = 2025
        if 'time_series' in result:
            years = list(result['time_series'].keys())
            if years:
                start_year = min(years)
                end_year = max(years)
                total_years = len(years)
            else:
                end_year = start_year
                total_years = 1
        else:
            end_year = start_year + 49
            total_years = 50

        # 获取情景名称
        scenario_name = self.scenarios.get(scenario_id, {}).get('name', scenario_id)

        return ScenarioSummary(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            start_year=start_year,
            end_year=end_year,
            total_years=total_years,
            key_metrics=key_metrics,
            policy_tools=policy_tools,
            created_at=datetime.now()
        )

    def get_all_scenario_summaries(self) -> List[ScenarioSummary]:
        """
        获取所有情景摘要

        Returns:
            List[ScenarioSummary]: 情景摘要列表
        """
        summaries = []

        for scenario_id in self.scenarios.keys():
            try:
                summary = self.get_scenario_summary(scenario_id)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"获取情景 {scenario_id} 摘要失败: {str(e)}")

        return summaries


# ============================================================================
# VisualizationGenerator 类 - 可视化生成器
# ============================================================================

class VisualizationGenerator:
    """可视化生成器"""

    def __init__(self, output_dir: str = "./output"):
        """
        初始化可视化生成器

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.visualization_dir = self.output_dir / "visualization"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        # 颜色配置
        self.color_palette = {
            'scenarios': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'policy_tools': ['#4c72b0', '#55a868', '#c44e52', '#8172b3', '#ccb974'],
            'groups': ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
        }

        logger.info("可视化生成器初始化完成")

    def plot_policy_evolution(self, policy_history: Dict[str, Any],
                              scenario_name: str = "") -> str:
        """
        绘制政策演化图（图3-1类型）

        Args:
            policy_history: 政策演化历史数据
            scenario_name: 情景名称

        Returns:
            str: 保存的文件路径
        """
        if 'policy_tools' not in policy_history:
            logger.error("政策历史数据中缺少'policy_tools'键")
            return ""

        policy_tools = policy_history['policy_tools']

        # 提取数据
        years = list(policy_tools.keys())
        if not years:
            logger.warning("政策演化数据为空")
            return ""

        # 获取政策工具名称
        first_year = years[0]
        if first_year in policy_tools:
            tool_names = list(policy_tools[first_year].keys())
        else:
            tool_names = []

        # 创建子图
        fig, axes = plt.subplots(len(tool_names), 1, figsize=(12, 3 * len(tool_names)))

        # 如果只有一个政策工具，确保axes是列表
        if len(tool_names) == 1:
            axes = [axes]

        for i, tool_name in enumerate(tool_names):
            ax = axes[i]

            # 提取该工具的时间序列
            tool_values = []
            valid_years = []

            for year in years:
                if year in policy_tools and tool_name in policy_tools[year]:
                    value = policy_tools[year][tool_name]
                    if isinstance(value, dict) and 'value' in value:
                        tool_values.append(value['value'])
                    else:
                        tool_values.append(value)
                    valid_years.append(year)

            if not valid_years:
                continue

            # 绘制折线图
            ax.plot(valid_years, tool_values, marker='o', linewidth=2, markersize=6,
                    color=self.color_palette['policy_tools'][i % len(self.color_palette['policy_tools'])])

            # 设置标题和标签
            ax.set_title(f'政策工具 {tool_name} 演化', fontsize=14, fontweight='bold')
            ax.set_xlabel('年份', fontsize=12)
            ax.set_ylabel('政策强度', fontsize=12)

            # 设置网格
            ax.grid(True, alpha=0.3, linestyle='--')

            # 设置x轴刻度
            ax.set_xticks(valid_years[::max(1, len(valid_years) // 10)])

            # 添加数据标签
            for year, value in zip(valid_years, tool_values):
                ax.annotate(f'{value:.2f}', (year, value),
                            textcoords="offset points", xytext=(0, 10),
                            ha='center', fontsize=9)

        # 设置总标题
        if scenario_name:
            fig.suptitle(f'{scenario_name} - 政策工具演化图', fontsize=16, fontweight='bold', y=1.02)
        else:
            fig.suptitle('政策工具演化图', fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"policy_evolution_{scenario_name}_{timestamp}.png"
        filepath = self.visualization_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"政策演化图已保存: {filepath}")
        return str(filepath)

    def plot_market_dynamics(self, market_results: List[Dict[str, Any]],
                             scenario_name: str = "") -> str:
        """
        绘制市场动态图（价格、交易量、渗透率）

        Args:
            market_results: 市场结果列表
            scenario_name: 情景名称

        Returns:
            str: 保存的文件路径
        """
        if not market_results:
            logger.warning("市场数据为空")
            return ""

        # 转换为DataFrame
        df = pd.DataFrame(market_results)

        # 提取年份
        if 'year' not in df.columns:
            logger.error("市场数据中缺少'year'列")
            return ""

        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 1. 绿电价格变化
        if 'avg_green_price' in df.columns:
            ax1 = axes[0]
            ax1.plot(df['year'], df['avg_green_price'], marker='o', linewidth=2,
                     color='#2ca02c', label='绿电平均价格')

            # 添加传统电力价格参考线
            ax1.axhline(y=base.TRADITIONAL_POWER_PRICE, color='#d62728',
                        linestyle='--', linewidth=1.5, label='传统电力价格')

            ax1.set_title('绿电价格变化趋势', fontsize=14, fontweight='bold')
            ax1.set_xlabel('年份', fontsize=12)
            ax1.set_ylabel('价格 (元/kWh)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3, linestyle='--')

        # 2. 绿电交易量和渗透率
        ax2 = axes[1]

        if 'green_power_volume' in df.columns:
            # 交易量（柱状图）
            ax2.bar(df['year'], df['green_power_volume'],
                    color='#1f77b4', alpha=0.7, label='绿电交易量')
            ax2.set_ylabel('交易量 (万千瓦时)', fontsize=12, color='#1f77b4')
            ax2.tick_params(axis='y', labelcolor='#1f77b4')

        if 'penetration_rate' in df.columns:
            # 渗透率（折线图，次坐标轴）
            ax2_twin = ax2.twinx()
            ax2_twin.plot(df['year'], df['penetration_rate'], marker='s',
                          linewidth=2, color='#ff7f0e', label='绿电渗透率')
            ax2_twin.set_ylabel('渗透率 (%)', fontsize=12, color='#ff7f0e')
            ax2_twin.tick_params(axis='y', labelcolor='#ff7f0e')

            # 合并图例
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax2.set_title('绿电交易量和渗透率', fontsize=14, fontweight='bold')
        ax2.set_xlabel('年份', fontsize=12)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # 3. 碳减排和社会福利
        ax3 = axes[2]

        if 'carbon_reduction' in df.columns:
            # 碳减排（柱状图）
            ax3.bar(df['year'], df['carbon_reduction'],
                    color='#2ca02c', alpha=0.7, label='碳减排量')
            ax3.set_ylabel('碳减排 (吨CO₂)', fontsize=12, color='#2ca02c')
            ax3.tick_params(axis='y', labelcolor='#2ca02c')

        if 'social_welfare' in df.columns:
            # 社会福利（折线图，次坐标轴）
            ax3_twin = ax3.twinx()
            ax3_twin.plot(df['year'], df['social_welfare'], marker='^',
                          linewidth=2, color='#9467bd', label='社会福利')
            ax3_twin.set_ylabel('社会福利', fontsize=12, color='#9467bd')
            ax3_twin.tick_params(axis='y', labelcolor='#9467bd')

            # 合并图例
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax3.set_title('碳减排和社会福利', fontsize=14, fontweight='bold')
        ax3.set_xlabel('年份', fontsize=12)
        ax3.grid(True, alpha=0.3, linestyle='--')

        # 设置总标题
        if scenario_name:
            fig.suptitle(f'{scenario_name} - 市场动态分析', fontsize=16, fontweight='bold', y=1.02)
        else:
            fig.suptitle('市场动态分析', fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_dynamics_{scenario_name}_{timestamp}.png"
        filepath = self.visualization_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"市场动态图已保存: {filepath}")
        return str(filepath)

    def plot_agent_distributions(self, agent_data: Dict[str, Any],
                                 year: int, scenario_name: str = "") -> str:
        """
        绘制主体分布图

        Args:
            agent_data: 主体数据
            year: 年份
            scenario_name: 情景名称

        Returns:
            str: 保存的文件路径
        """
        if 'green_energy_firms' not in agent_data or 'consumers' not in agent_data:
            logger.error("主体数据中缺少必要信息")
            return ""

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # 1. 绿电企业资产分布
        if 'green_energy_firms' in agent_data:
            green_firms = agent_data['green_energy_firms']
            if green_firms:
                # 按分组统计
                groups = {}
                for firm in green_firms:
                    group_id = firm.get('group_id', 0)
                    if group_id not in groups:
                        groups[group_id] = []
                    groups[group_id].append(firm.get('asset_total', 0))

                # 绘制箱线图
                group_data = []
                group_labels = []
                for group_id, assets in groups.items():
                    group_data.append(assets)
                    group_labels.append(f'分组{group_id}')

                axes[0].boxplot(group_data, labels=group_labels)
                axes[0].set_title('绿电企业资产分布（按分组）', fontsize=12, fontweight='bold')
                axes[0].set_ylabel('资产总额（亿元）', fontsize=10)
                axes[0].grid(True, alpha=0.3, linestyle='--')

        # 2. 绿电企业利润分布
        if 'green_energy_firms' in agent_data:
            green_firms = agent_data['green_energy_firms']
            if green_firms:
                profits = [firm.get('profit', 0) for firm in green_firms]
                axes[1].hist(profits, bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
                axes[1].set_title('绿电企业利润分布', fontsize=12, fontweight='bold')
                axes[1].set_xlabel('利润（亿元）', fontsize=10)
                axes[1].set_ylabel('企业数量', fontsize=10)
                axes[1].grid(True, alpha=0.3, linestyle='--')

        # 3. 用电企业需求分布
        if 'consumers' in agent_data:
            consumers = agent_data['consumers']
            if consumers:
                demands = [consumer.get('annual_consumption', 0) for consumer in consumers]
                industries = [consumer.get('industry', '其他') for consumer in consumers]

                # 按行业分组
                industry_data = {}
                for demand, industry in zip(demands, industries):
                    if industry not in industry_data:
                        industry_data[industry] = []
                    industry_data[industry].append(demand)

                # 绘制分组箱线图
                industry_labels = list(industry_data.keys())
                industry_values = [industry_data[label] for label in industry_labels]

                axes[2].boxplot(industry_values, labels=industry_labels)
                axes[2].set_title('用电企业需求分布（按行业）', fontsize=12, fontweight='bold')
                axes[2].set_ylabel('年用电量（万千瓦时）', fontsize=10)
                axes[2].grid(True, alpha=0.3, linestyle='--')

        # 4. 用电企业绿电偏好分布
        if 'consumers' in agent_data:
            consumers = agent_data['consumers']
            if consumers:
                preferences = [consumer.get('green_preference', 0) for consumer in consumers]
                axes[3].hist(preferences, bins=20, color='#2ca02c', alpha=0.7, edgecolor='black')
                axes[3].set_title('用电企业绿电偏好分布', fontsize=12, fontweight='bold')
                axes[3].set_xlabel('绿电偏好系数', fontsize=10)
                axes[3].set_ylabel('企业数量', fontsize=10)
                axes[3].grid(True, alpha=0.3, linestyle='--')

        # 设置总标题
        if scenario_name:
            fig.suptitle(f'{scenario_name} - {year}年主体分布分析', fontsize=16, fontweight='bold', y=1.02)
        else:
            fig.suptitle(f'{year}年主体分布分析', fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_distributions_{scenario_name}_{year}_{timestamp}.png"
        filepath = self.visualization_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"主体分布图已保存: {filepath}")
        return str(filepath)

    def plot_comparison_charts(self, comparison_results: Dict[str, Any]) -> List[str]:
        """
        绘制情景对比图（雷达图、柱状图）

        Args:
            comparison_results: 比较结果

        Returns:
            List[str]: 保存的文件路径列表
        """
        if 'comparison_metrics' not in comparison_results:
            logger.error("比较结果中缺少'comparison_metrics'键")
            return []

        comparison_metrics = comparison_results['comparison_metrics']
        scenarios = list(comparison_metrics.keys())

        if not scenarios:
            logger.warning("比较结果为空")
            return []

        saved_files = []

        # 1. 雷达图 - 综合表现对比
        radar_file = self._plot_radar_chart(comparison_metrics, scenarios)
        if radar_file:
            saved_files.append(radar_file)

        # 2. 柱状图 - 关键指标对比
        bar_file = self._plot_bar_chart_comparison(comparison_metrics, scenarios)
        if bar_file:
            saved_files.append(bar_file)

        # 3. 散点图 - 成本效益分析
        scatter_file = self._plot_scatter_analysis(comparison_metrics, scenarios)
        if scatter_file:
            saved_files.append(scatter_file)

        return saved_files

    def _plot_radar_chart(self, comparison_metrics: Dict[str, Dict[str, float]],
                          scenarios: List[str]) -> str:
        """
        绘制雷达图

        Args:
            comparison_metrics: 比较指标
            scenarios: 情景列表

        Returns:
            str: 保存的文件路径
        """
        # 选择要展示的指标
        radar_metrics = [
            'avg_penetration_rate',
            'final_social_welfare',
            'total_carbon_reduction',
            'avg_capacity_utilization',
            'avg_roe_green_energy'
        ]

        # 检查数据可用性
        available_metrics = []
        for metric in radar_metrics:
            if any(metric in metrics for metrics in comparison_metrics.values()):
                available_metrics.append(metric)

        if len(available_metrics) < 3:
            logger.warning("可用于雷达图的指标不足")
            return ""

        # 准备数据
        categories = available_metrics
        num_vars = len(categories)

        # 计算每个指标的最大值用于归一化
        max_values = {}
        for metric in categories:
            values = []
            for scenario_metrics in comparison_metrics.values():
                if metric in scenario_metrics:
                    values.append(scenario_metrics[metric])
            if values:
                max_values[metric] = max(values)
            else:
                max_values[metric] = 1

        # 创建雷达图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合多边形

        # 绘制每个情景
        for i, scenario in enumerate(scenarios):
            if scenario not in comparison_metrics:
                continue

            values = []
            for metric in categories:
                if metric in comparison_metrics[scenario]:
                    # 归一化到0-1范围
                    normalized_value = comparison_metrics[scenario][metric] / max_values[metric]
                    values.append(normalized_value)
                else:
                    values.append(0)

            values += values[:1]  # 闭合多边形

            # 绘制
            ax.plot(angles, values, 'o-', linewidth=2,
                    label=scenario,
                    color=self.color_palette['scenarios'][i % len(self.color_palette['scenarios'])])
            ax.fill(angles, values, alpha=0.1)

        # 设置角度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)

        # 设置半径标签
        ax.set_rlabel_position(30)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
        ax.set_ylim(0, 1)

        # 添加标题和图例
        plt.title('情景综合表现雷达图', fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"radar_chart_{timestamp}.png"
        filepath = self.visualization_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"雷达图已保存: {filepath}")
        return str(filepath)

    def _plot_bar_chart_comparison(self, comparison_metrics: Dict[str, Dict[str, float]],
                                   scenarios: List[str]) -> str:
        """
        绘制柱状图对比

        Args:
            comparison_metrics: 比较指标
            scenarios: 情景列表

        Returns:
            str: 保存的文件路径
        """
        # 选择要展示的指标
        bar_metrics = [
            'total_carbon_reduction',
            'total_policy_cost',
            'avg_green_price',
            'avg_penetration_rate',
            'final_social_welfare'
        ]

        # 检查数据可用性
        available_metrics = []
        for metric in bar_metrics:
            if any(metric in metrics for metrics in comparison_metrics.values()):
                available_metrics.append(metric)

        if not available_metrics:
            logger.warning("可用于柱状图的指标不足")
            return ""

        # 创建子图
        num_metrics = len(available_metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 3 * num_metrics))

        if num_metrics == 1:
            axes = [axes]

        # 设置指标单位
        units = {
            'total_carbon_reduction': '吨CO₂',
            'total_policy_cost': '亿元',
            'avg_green_price': '元/kWh',
            'avg_penetration_rate': '%',
            'final_social_welfare': ''
        }

        # 绘制每个指标的柱状图
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]

            # 提取数据
            values = []
            for scenario in scenarios:
                if scenario in comparison_metrics and metric in comparison_metrics[scenario]:
                    values.append(comparison_metrics[scenario][metric])
                else:
                    values.append(0)

            # 绘制柱状图
            bars = ax.bar(scenarios, values,
                          color=self.color_palette['scenarios'][:len(scenarios)])

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

            # 设置标题和标签
            unit = units.get(metric, '')
            ax.set_title(f'{metric} ({unit})', fontsize=12, fontweight='bold')
            ax.set_ylabel(unit, fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # 设置总标题
        fig.suptitle('情景关键指标对比', fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bar_chart_comparison_{timestamp}.png"
        filepath = self.visualization_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"柱状图对比已保存: {filepath}")
        return str(filepath)

    def _plot_scatter_analysis(self, comparison_metrics: Dict[str, Dict[str, float]],
                               scenarios: List[str]) -> str:
        """
        绘制散点图分析

        Args:
            comparison_metrics: 比较指标
            scenarios: 情景列表

        Returns:
            str: 保存的文件路径
        """
        # 检查是否有足够的数据
        if 'total_policy_cost' not in comparison_metrics.get(scenarios[0], {}) or \
                'total_carbon_reduction' not in comparison_metrics.get(scenarios[0], {}):
            logger.warning("缺少成本效益分析所需数据")
            return ""

        # 提取数据
        x_data = []  # 政策成本
        y_data = []  # 碳减排
        z_data = []  # 渗透率（用于气泡大小）
        labels = []  # 情景标签

        for scenario in scenarios:
            if scenario in comparison_metrics:
                metrics = comparison_metrics[scenario]
                if 'total_policy_cost' in metrics and 'total_carbon_reduction' in metrics:
                    x_data.append(metrics['total_policy_cost'])
                    y_data.append(metrics['total_carbon_reduction'])

                    # 使用渗透率作为气泡大小
                    if 'avg_penetration_rate' in metrics:
                        z_data.append(metrics['avg_penetration_rate'] * 100)  # 转换为百分比
                    else:
                        z_data.append(50)  # 默认大小

                    labels.append(scenario)

        if len(x_data) < 2:
            logger.warning("可用于散点图的数据点不足")
            return ""

        # 创建散点图
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制散点图
        scatter = ax.scatter(x_data, y_data, s=z_data, alpha=0.6,
                             c=range(len(x_data)), cmap='viridis', edgecolors='black')

        # 添加标签
        for i, label in enumerate(labels):
            ax.annotate(label, (x_data[i], y_data[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        # 添加趋势线
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), "r--", alpha=0.5, label='趋势线')

        # 设置标题和标签
        ax.set_title('成本效益分析：政策成本 vs 碳减排', fontsize=14, fontweight='bold')
        ax.set_xlabel('总政策成本（亿元）', fontsize=12)
        ax.set_ylabel('累计碳减排（吨CO₂）', fontsize=12)

        # 添加网格和图例
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()

        # 添加颜色条（表示气泡大小）
        cbar = plt.colorbar(scatter)
        cbar.set_label('绿电渗透率 (%)', fontsize=10)

        # 计算并显示单位碳减排成本
        ax.text(0.05, 0.95,
                f'平均单位碳减排成本: {np.mean([x / y for x, y in zip(x_data, y_data) if y > 0]):.2f} 元/吨CO₂',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scatter_analysis_{timestamp}.png"
        filepath = self.visualization_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"散点图分析已保存: {filepath}")
        return str(filepath)

    def generate_dashboard(self, scenario_results: Dict[str, Dict[str, Any]]) -> str:
        """
        生成综合仪表板HTML报告

        Args:
            scenario_results: 各情景仿真结果

        Returns:
            str: 保存的HTML文件路径
        """
        try:
            import plotly.io as pio

            # 提取数据
            scenarios = list(scenario_results.keys())
            if not scenarios:
                logger.warning("没有可用的情景结果")
                return ""

            # 创建子图
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('政策工具演化', '绿电价格趋势',
                                '绿电渗透率', '碳减排量',
                                '社会福利', '企业利润率'),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                       [{'type': 'scatter'}, {'type': 'scatter'}],
                       [{'type': 'scatter'}, {'type': 'scatter'}]]
            )

            # 1. 政策工具演化
            if 'policy_evolution' in scenario_results[scenarios[0]]:
                policy_data = scenario_results[scenarios[0]]['policy_evolution']
                if 'policy_tools' in policy_data:
                    years = list(policy_data['policy_tools'].keys())
                    if years:
                        first_year = years[0]
                        if first_year in policy_data['policy_tools']:
                            tools = list(policy_data['policy_tools'][first_year].keys())
                            for tool in tools:
                                values = []
                                for year in years:
                                    if year in policy_data['policy_tools'] and tool in policy_data['policy_tools'][
                                        year]:
                                        value = policy_data['policy_tools'][year][tool]
                                        if isinstance(value, dict) and 'value' in value:
                                            values.append(value['value'])
                                        else:
                                            values.append(value)
                                    else:
                                        values.append(0)

                                fig.add_trace(
                                    go.Scatter(x=years, y=values, mode='lines+markers',
                                               name=tool),
                                    row=1, col=1
                                )

            # 2. 绿电价格趋势
            for scenario in scenarios[:3]:  # 只显示前3个情景
                if 'time_series' in scenario_results[scenario]:
                    time_series = scenario_results[scenario]['time_series']
                    years = list(time_series.keys())
                    prices = [time_series[year].get('avg_green_price', 0) for year in years]

                    fig.add_trace(
                        go.Scatter(x=years, y=prices, mode='lines+markers',
                                   name=scenario),
                        row=1, col=2
                    )

            # 3. 绿电渗透率
            for scenario in scenarios[:3]:
                if 'time_series' in scenario_results[scenario]:
                    time_series = scenario_results[scenario]['time_series']
                    years = list(time_series.keys())
                    penetration = [time_series[year].get('penetration_rate', 0) * 100 for year in years]  # 转换为百分比

                    fig.add_trace(
                        go.Scatter(x=years, y=penetration, mode='lines+markers',
                                   name=scenario, showlegend=False),
                        row=2, col=1
                    )

            # 4. 碳减排量
            for scenario in scenarios[:3]:
                if 'time_series' in scenario_results[scenario]:
                    time_series = scenario_results[scenario]['time_series']
                    years = list(time_series.keys())
                    carbon = [time_series[year].get('carbon_reduction', 0) for year in years]

                    fig.add_trace(
                        go.Scatter(x=years, y=carbon, mode='lines+markers',
                                   name=scenario, showlegend=False),
                        row=2, col=2
                    )

            # 5. 社会福利
            for scenario in scenarios[:3]:
                if 'time_series' in scenario_results[scenario]:
                    time_series = scenario_results[scenario]['time_series']
                    years = list(time_series.keys())
                    welfare = [time_series[year].get('social_welfare', 0) for year in years]

                    fig.add_trace(
                        go.Scatter(x=years, y=welfare, mode='lines+markers',
                                   name=scenario, showlegend=False),
                        row=3, col=1
                    )

            # 6. 企业利润率
            # 这里简化处理，实际可能需要从主体数据计算
            fig.add_trace(
                go.Scatter(x=[1, 2, 3, 4, 5], y=[10, 11, 12, 13, 14],
                           mode='lines', name='绿电企业ROE'),
                row=3, col=2
            )

            # 更新布局
            fig.update_layout(
                height=900,
                title_text=f"SG-ABM仿真仪表板 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                showlegend=True
            )

            # 保存为HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_{timestamp}.html"
            filepath = self.visualization_dir / filename

            fig.write_html(str(filepath))

            logger.info(f"仪表板已保存: {filepath}")
            return str(filepath)

        except ImportError:
            logger.warning("Plotly未安装，无法生成HTML仪表板")
            return ""
        except Exception as e:
            logger.error(f"生成仪表板失败: {str(e)}")
            return ""


# ============================================================================
# DynamicNetworkVisualizer 类 - 动态网络可视化器
# ============================================================================

class DynamicNetworkVisualizer:
    """动态网络可视化器"""

    def __init__(self, output_dir: str = "./output"):
        """
        初始化网络可视化器

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.network_dir = self.output_dir / "network_visualization"
        self.network_dir.mkdir(parents=True, exist_ok=True)

        self.network_data = []  # 存储各期网络数据

        logger.info("动态网络可视化器初始化完成")

    def extract_network_data(self, period_result: Dict[str, Any]) -> Optional[NetworkData]:
        """
        从单期结果提取网络数据

        Args:
            period_result: 单期仿真结果

        Returns:
            Optional[NetworkData]: 网络数据，如果无法提取则返回None
        """
        if 'agents' not in period_result:
            logger.warning("结果中缺少主体数据，无法提取网络数据")
            return None

        agents = period_result['agents']
        year = period_result.get('year', 0)
        scenario_id = period_result.get('scenario_id', 'unknown')

        # 创建节点
        nodes = []
        node_id_map = {}  # 节点名称到索引的映射

        # 1. 添加政府节点
        if 'government' in agents:
            gov = agents['government']
            gov_id = len(nodes)
            nodes.append({
                'id': gov_id,
                'name': '政府',
                'type': 'government',
                'size': 50,  # 固定大小
                'color': '#d62728'
            })
            node_id_map['government'] = gov_id

        # 2. 添加绿电企业节点
        if 'green_energy_firms' in agents:
            for i, firm in enumerate(agents['green_energy_firms']):
                firm_id = len(nodes)
                node_name = f"绿电_{firm.get('firm_id', i)}"

                # 根据规模确定节点大小
                asset = firm.get('asset_total', 0)
                size = 10 + min(asset / 100, 40)  # 资产规模影响节点大小

                # 根据分组确定颜色
                group_id = firm.get('group_id', 0)
                if group_id == 0:
                    color = '#66c2a5'  # 低规模组
                elif group_id == 1:
                    color = '#fc8d62'  # 中规模组
                else:
                    color = '#8da0cb'  # 高规模组

                nodes.append({
                    'id': firm_id,
                    'name': node_name,
                    'type': 'green_energy',
                    'group_id': group_id,
                    'size': size,
                    'color': color,
                    'asset_total': asset,
                    'profit': firm.get('profit', 0)
                })
                node_id_map[node_name] = firm_id

        # 3. 添加用电企业节点
        if 'consumers' in agents:
            for i, consumer in enumerate(agents['consumers']):
                consumer_id = len(nodes)
                node_name = f"用电_{consumer.get('consumer_id', i)}"

                # 根据用电量确定节点大小
                consumption = consumer.get('annual_consumption', 0)
                size = 5 + min(consumption / 1000, 30)

                # 根据行业确定颜色
                industry = consumer.get('industry', '其他')
                industry_colors = {
                    '高科技': '#e78ac3',
                    '制造业': '#a6d854',
                    '服务业': '#ffd92f',
                    '其他': '#b3b3b3'
                }
                color = industry_colors.get(industry, '#b3b3b3')

                nodes.append({
                    'id': consumer_id,
                    'name': node_name,
                    'type': 'consumer',
                    'industry': industry,
                    'size': size,
                    'color': color,
                    'annual_consumption': consumption,
                    'green_preference': consumer.get('green_preference', 0)
                })
                node_id_map[node_name] = consumer_id

        # 创建边（交易关系）
        edges = []

        # 提取交易数据（如果有）
        if 'market_data' in period_result:
            market_data = period_result['market_data']

            # 这里简化处理，实际应根据交易记录创建边
            # 假设每个绿电企业与政府、用电企业之间都有交易

            # 政府与绿电企业之间的补贴流
            if 'government' in node_id_map:
                gov_id = node_id_map['government']
                for node in nodes:
                    if node['type'] == 'green_energy':
                        edges.append({
                            'source': gov_id,
                            'target': node['id'],
                            'value': node.get('profit', 0) * 0.1,  # 简化计算
                            'type': 'subsidy',
                            'color': '#ff7f0e'
                        })

            # 绿电企业与用电企业之间的电力交易
            green_nodes = [n for n in nodes if n['type'] == 'green_energy']
            consumer_nodes = [n for n in nodes if n['type'] == 'consumer']

            # 随机创建一些交易关系（实际应根据具体交易数据）
            np.random.seed(year)  # 使用年份作为随机种子

            for green_node in green_nodes[:min(5, len(green_nodes))]:  # 每个绿电企业与最多5个用电企业交易
                num_consumers = min(np.random.randint(1, 6), len(consumer_nodes))
                selected_consumers = np.random.choice(consumer_nodes, num_consumers, replace=False)

                for consumer_node in selected_consumers:
                    # 交易量与绿电企业资产和用电企业需求相关
                    trade_value = min(green_node.get('asset_total', 0),
                                      consumer_node.get('annual_consumption', 0)) * 0.01

                    edges.append({
                        'source': green_node['id'],
                        'target': consumer_node['id'],
                        'value': trade_value,
                        'type': 'trade',
                        'color': '#2ca02c'
                    })

        # 创建邻接矩阵
        num_nodes = len(nodes)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))

        for edge in edges:
            source = edge['source']
            target = edge['target']
            value = edge['value']
            adjacency_matrix[source, target] = value

        network_data = NetworkData(
            year=year,
            scenario_id=scenario_id,
            nodes=nodes,
            edges=edges,
            adjacency_matrix=adjacency_matrix
        )

        self.network_data.append(network_data)

        logger.info(f"已提取 {year} 年网络数据: {len(nodes)} 个节点, {len(edges)} 条边")
        return network_data

    def generate_network_files(self, simulation_results: List[Dict[str, Any]]) -> List[str]:
        """
        生成各期网络数据文件（供R使用）

        Args:
            simulation_results: 仿真结果列表

        Returns:
            List[str]: 保存的文件路径列表
        """
        saved_files = []

        for result in simulation_results:
            network_data = self.extract_network_data(result)
            if network_data:
                filepath = self._save_network_for_r(network_data)
                if filepath:
                    saved_files.append(filepath)

        return saved_files

    def _save_network_for_r(self, network_data: NetworkData) -> str:
        """
        保存网络数据供R使用

        Args:
            network_data: 网络数据

        Returns:
            str: 保存的文件路径
        """
        year = network_data.year
        scenario_id = network_data.scenario_id

        # 1. 保存节点数据
        nodes_df = pd.DataFrame(network_data.nodes)
        nodes_file = self.network_dir / f"nodes_{scenario_id}_{year}.csv"
        nodes_df.to_csv(nodes_file, index=False, encoding='utf-8-sig')

        # 2. 保存边数据
        edges_df = pd.DataFrame(network_data.edges)
        edges_file = self.network_dir / f"edges_{scenario_id}_{year}.csv"
        edges_df.to_csv(edges_file, index=False, encoding='utf-8-sig')

        # 3. 保存邻接矩阵
        if network_data.adjacency_matrix is not None:
            adj_df = pd.DataFrame(network_data.adjacency_matrix)
            adj_file = self.network_dir / f"adjacency_{scenario_id}_{year}.csv"
            adj_df.to_csv(adj_file, index=False, encoding='utf-8-sig')

        logger.info(f"网络数据已保存供R使用: {nodes_file}")
        return str(nodes_file)

    def calculate_network_metrics(self, network_data_list: List[NetworkData]) -> List[NetworkMetrics]:
        """
        计算网络指标

        Args:
            network_data_list: 网络数据列表

        Returns:
            List[NetworkMetrics]: 网络指标列表
        """
        network_metrics_list = []

        for network_data in network_data_list:
            if not network_data.edges:
                continue

            # 创建NetworkX图
            G = nx.Graph()

            # 添加节点
            for node in network_data.nodes:
                G.add_node(node['id'], **node)

            # 添加边
            for edge in network_data.edges:
                G.add_edge(edge['source'], edge['target'], weight=edge['value'])

            # 计算网络指标
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()

            # 网络密度
            if num_nodes > 1:
                density = nx.density(G)
            else:
                density = 0

            # 平均度
            if num_nodes > 0:
                degrees = dict(G.degree())
                avg_degree = sum(degrees.values()) / num_nodes
            else:
                avg_degree = 0

            # 聚类系数
            if num_nodes > 2:
                clustering_coefficient = nx.average_clustering(G)
            else:
                clustering_coefficient = 0

            # 度中心性
            if num_nodes > 0:
                degree_centrality = nx.degree_centrality(G)
            else:
                degree_centrality = {}

            # 介数中心性
            if num_nodes > 0:
                betweenness_centrality = nx.betweenness_centrality(G)
            else:
                betweenness_centrality = {}

            # 创建网络指标对象
            network_metrics = NetworkMetrics(
                year=network_data.year,
                num_nodes=num_nodes,
                num_edges=num_edges,
                density=density,
                avg_degree=avg_degree,
                clustering_coefficient=clustering_coefficient,
                degree_centrality=degree_centrality,
                betweenness_centrality=betweenness_centrality
            )

            network_metrics_list.append(network_metrics)

        return network_metrics_list

    def plot_network_evolution(self, network_metrics_list: List[NetworkMetrics],
                               scenario_name: str = "") -> str:
        """
        绘制网络演化图

        Args:
            network_metrics_list: 网络指标列表
            scenario_name: 情景名称

        Returns:
            str: 保存的文件路径
        """
        if not network_metrics_list:
            logger.warning("网络指标数据为空")
            return ""

        # 提取数据
        years = [metrics.year for metrics in network_metrics_list]
        densities = [metrics.density for metrics in network_metrics_list]
        avg_degrees = [metrics.avg_degree for metrics in network_metrics_list]
        clustering_coeffs = [metrics.clustering_coefficient for metrics in network_metrics_list]
        num_nodes = [metrics.num_nodes for metrics in network_metrics_list]
        num_edges = [metrics.num_edges for metrics in network_metrics_list]

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # 1. 网络密度和平均度
        ax1 = axes[0]
        ax1.plot(years, densities, marker='o', linewidth=2,
                 color='#1f77b4', label='网络密度')
        ax1.set_xlabel('年份', fontsize=10)
        ax1.set_ylabel('网络密度', fontsize=10, color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.grid(True, alpha=0.3, linestyle='--')

        ax1_twin = ax1.twinx()
        ax1_twin.plot(years, avg_degrees, marker='s', linewidth=2,
                      color='#ff7f0e', label='平均度')
        ax1_twin.set_ylabel('平均度', fontsize=10, color='#ff7f0e')
        ax1_twin.tick_params(axis='y', labelcolor='#ff7f0e')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.set_title('网络密度和平均度演化', fontsize=12, fontweight='bold')

        # 2. 聚类系数
        ax2 = axes[1]
        ax2.plot(years, clustering_coeffs, marker='^', linewidth=2,
                 color='#2ca02c')
        ax2.set_xlabel('年份', fontsize=10)
        ax2.set_ylabel('聚类系数', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_title('聚类系数演化', fontsize=12, fontweight='bold')

        # 3. 节点和边数量
        ax3 = axes[2]
        ax3.bar(years, num_nodes, alpha=0.7, color='#9467bd', label='节点数')
        ax3.set_xlabel('年份', fontsize=10)
        ax3.set_ylabel('节点数', fontsize=10, color='#9467bd')
        ax3.tick_params(axis='y', labelcolor='#9467bd')
        ax3.grid(True, alpha=0.3, linestyle='--')

        ax3_twin = ax3.twinx()
        ax3_twin.plot(years, num_edges, marker='d', linewidth=2,
                      color='#d62728', label='边数')
        ax3_twin.set_ylabel('边数', fontsize=10, color='#d62728')
        ax3_twin.tick_params(axis='y', labelcolor='#d62728')

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax3.set_title('网络规模演化', fontsize=12, fontweight='bold')

        # 4. 中心性指标（使用最后一年的数据）
        ax4 = axes[3]
        if network_metrics_list:
            last_metrics = network_metrics_list[-1]

            # 提取度中心性前10的节点
            degree_centrality = last_metrics.degree_centrality
            if degree_centrality:
                sorted_degrees = sorted(degree_centrality.items(),
                                        key=lambda x: x[1], reverse=True)[:10]

                node_ids = [item[0] for item in sorted_degrees]
                centrality_values = [item[1] for item in sorted_degrees]

                ax4.bar(range(len(node_ids)), centrality_values,
                        color='#8c564b', alpha=0.7)
                ax4.set_xticks(range(len(node_ids)))
                ax4.set_xticklabels([f'节点{id}' for id in node_ids], rotation=45)
                ax4.set_ylabel('度中心性', fontsize=10)
                ax4.set_title('节点度中心性排名（前10）', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

        # 设置总标题
        if scenario_name:
            fig.suptitle(f'{scenario_name} - 网络结构演化分析', fontsize=16, fontweight='bold', y=1.02)
        else:
            fig.suptitle('网络结构演化分析', fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"network_evolution_{scenario_name}_{timestamp}.png"
        filepath = self.visualization_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"网络演化图已保存: {filepath}")
        return str(filepath)

    def visualize_single_network(self, network_data: NetworkData,
                                 layout: str = 'spring') -> str:
        """
        可视化单个网络

        Args:
            network_data: 网络数据
            layout: 布局算法（'spring', 'circular', 'kamada_kawai'）

        Returns:
            str: 保存的文件路径
        """
        # 创建NetworkX图
        G = nx.Graph()

        # 添加节点
        for node in network_data.nodes:
            G.add_node(node['id'], **node)

        # 添加边
        for edge in network_data.edges:
            G.add_edge(edge['source'], edge['target'], weight=edge['value'])

        # 创建图形
        plt.figure(figsize=(12, 10))

        # 选择布局
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        # 绘制节点
        node_colors = []
        node_sizes = []

        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            node_colors.append(node_data.get('color', '#1f77b4'))
            node_sizes.append(node_data.get('size', 300))

        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8)

        # 绘制边
        edge_colors = []
        edge_widths = []

        for u, v in G.edges():
            edge_data = G.edges[u, v]
            edge_colors.append(edge_data.get('color', '#888888'))

            # 根据边的权重设置宽度
            weight = edge_data.get('weight', 1)
            edge_widths.append(max(0.5, min(weight / 10, 5)))

        nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                               width=edge_widths, alpha=0.5)

        # 绘制标签
        labels = {node['id']: node['name'] for node in network_data.nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_family='sans-serif')

        # 设置标题
        plt.title(f"绿电交易网络 - {network_data.year}年", fontsize=16, fontweight='bold')

        # 添加图例
        legend_elements = []
        type_colors = {
            'government': '#d62728',
            'green_energy': '#1f77b4',
            'consumer': '#2ca02c'
        }

        for node_type, color in type_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=color, markersize=10,
                                              label=node_type))

        plt.legend(handles=legend_elements, loc='upper right')

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"network_{network_data.year}_{layout}_{timestamp}.png"
        filepath = self.visualization_dir / filename

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"网络可视化图已保存: {filepath}")
        return str(filepath)


# ============================================================================
# 工具函数
# ============================================================================

def create_demo_visualizations() -> Dict[str, str]:
    """
    创建演示可视化（用于测试）

    Returns:
        Dict[str, str]: 创建的文件路径字典
    """
    import tempfile

    output_dir = Path(tempfile.mkdtemp())
    visualizer = VisualizationGenerator(str(output_dir))

    created_files = {}

    try:
        # 创建演示数据
        years = list(range(2025, 2055))

        # 政策演化数据
        policy_history = {
            'policy_tools': {
                year: {
                    'x1': {'value': np.random.uniform(0.8, 1.2)},
                    'x2': {'value': np.random.uniform(0, 0.3)},
                    'x3': {'value': np.random.uniform(0, 0.6)},
                    'x4': {'value': np.random.uniform(0, 0.2)},
                    'x5': {'value': np.random.uniform(0, 0.1)}
                } for year in years
            }
        }

        # 市场动态数据
        market_data = []
        for i, year in enumerate(years):
            market_data.append({
                'year': year,
                'avg_green_price': 0.45 - i * 0.001,
                'green_power_volume': 1000 + i * 100,
                'penetration_rate': 0.1 + i * 0.01,
                'carbon_reduction': 10000 + i * 500,
                'social_welfare': 500 + i * 20
            })

        # 主体数据
        agent_data = {
            'green_energy_firms': [
                {'group_id': 0, 'asset_total': np.random.lognormal(5, 0.5),
                 'profit': np.random.normal(10, 3)} for _ in range(20)
            ],
            'consumers': [
                {'industry': np.random.choice(['高科技', '制造业', '服务业', '其他']),
                 'annual_consumption': np.random.lognormal(4, 0.3),
                 'green_preference': np.random.beta(2, 2)} for _ in range(30)
            ]
        }

        # 生成可视化
        policy_file = visualizer.plot_policy_evolution(policy_history, "演示情景")
        market_file = visualizer.plot_market_dynamics(market_data, "演示情景")
        agent_file = visualizer.plot_agent_distributions(agent_data, 2030, "演示情景")

        created_files['policy_evolution'] = policy_file
        created_files['market_dynamics'] = market_file
        created_files['agent_distributions'] = agent_file

        logger.info(f"演示可视化已创建: {output_dir}")

    except Exception as e:
        logger.error(f"创建演示可视化失败: {str(e)}")

    return created_files


# ============================================================================
# 主函数（测试用）
# ============================================================================

def main():
    """主函数（用于测试）"""
    print("SG-ABM 情景分析与可视化模块测试")
    print("=" * 50)

    # 创建演示可视化
    print("1. 创建演示可视化...")
    created_files = create_demo_visualizations()

    if created_files:
        print(f"成功创建 {len(created_files)} 个演示可视化文件")
        for file_type, file_path in created_files.items():
            print(f"  {file_type}: {file_path}")

    # 测试ScenarioManager
    print("\n2. 测试ScenarioManager...")
    scenario_manager = ScenarioManager(output_dir="./test_output_view")

    # 加载情景
    scenario_manager.load_scenario('S0')
    scenario_manager.load_scenario('S1')

    print(f"已加载 {len(scenario_manager.scenarios)} 个情景")

    # 测试VisualizationGenerator
    print("\n3. 测试VisualizationGenerator...")
    visualizer = VisualizationGenerator(output_dir="./test_output_view")

    # 创建测试数据
    test_comparison_results = {
        'comparison_metrics': {
            'S0': {
                'total_carbon_reduction': 50000,
                'total_policy_cost': 1000,
                'avg_green_price': 0.42,
                'avg_penetration_rate': 0.15,
                'final_social_welfare': 800
            },
            'S1': {
                'total_carbon_reduction': 75000,
                'total_policy_cost': 1500,
                'avg_green_price': 0.38,
                'avg_penetration_rate': 0.25,
                'final_social_welfare': 950
            },
            'S2': {
                'total_carbon_reduction': 60000,
                'total_policy_cost': 1200,
                'avg_green_price': 0.40,
                'avg_penetration_rate': 0.20,
                'final_social_welfare': 850
            }
        }
    }

    comparison_files = visualizer.plot_comparison_charts(test_comparison_results)
    print(f"已创建 {len(comparison_files)} 个对比图")

    # 测试DynamicNetworkVisualizer
    print("\n4. 测试DynamicNetworkVisualizer...")
    network_visualizer = DynamicNetworkVisualizer(output_dir="./test_output_view")

    # 创建测试网络数据
    test_network_data = NetworkData(
        year=2030,
        scenario_id='S0',
        nodes=[
            {'id': 0, 'name': '政府', 'type': 'government', 'size': 50, 'color': '#d62728'},
            {'id': 1, 'name': '绿电_1', 'type': 'green_energy', 'size': 30, 'color': '#1f77b4'},
            {'id': 2, 'name': '绿电_2', 'type': 'green_energy', 'size': 25, 'color': '#1f77b4'},
            {'id': 3, 'name': '用电_1', 'type': 'consumer', 'size': 20, 'color': '#2ca02c'},
            {'id': 4, 'name': '用电_2', 'type': 'consumer', 'size': 15, 'color': '#2ca02c'}
        ],
        edges=[
            {'source': 0, 'target': 1, 'value': 10, 'type': 'subsidy', 'color': '#ff7f0e'},
            {'source': 0, 'target': 2, 'value': 8, 'type': 'subsidy', 'color': '#ff7f0e'},
            {'source': 1, 'target': 3, 'value': 15, 'type': 'trade', 'color': '#2ca02c'},
            {'source': 2, 'target': 4, 'value': 12, 'type': 'trade', 'color': '#2ca02c'},
            {'source': 1, 'target': 4, 'value': 5, 'type': 'trade', 'color': '#2ca02c'}
        ]
    )

    # 可视化网络
    network_file = network_visualizer.visualize_single_network(test_network_data)
    print(f"网络可视化图已创建: {network_file}")

    # 计算网络指标
    network_metrics_list = network_visualizer.calculate_network_metrics([test_network_data])
    print(f"已计算 {len(network_metrics_list)} 个网络指标")

    print("\n测试完成！")

    # 清理测试文件
    import shutil
    test_output_dir = Path("./test_output_view")
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)
        print("测试输出目录已清理")


if __name__ == "__main__":
    main()