"""
SG-ABM模型输入输出与数据管理模块

本模块负责数据的读取、验证、保存和批量处理功能。
包含以下主要类：
1. DataLoader: 数据加载器，负责从各种格式读取输入数据
2. ResultSaver: 结果保存器，负责仿真结果的持久化存储
3. DataValidator: 数据验证器，负责输入数据的验证和清洗
4. BatchProcessor: 批量处理器，支持多情景批量运行

作者：Liang
日期：2026年
版本：1.0
"""

import os
import json
import yaml
import pickle
import csv
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
import warnings

# 导入已有模块
from core import base, fun, math_go

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ValidationResult:
    """数据验证结果"""
    is_valid: bool
    message: str
    issues: List[Dict[str, Any]]
    statistics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class DistributionTest:
    """分布检验结果"""
    distribution_type: str
    test_statistic: float
    p_value: float
    is_normal: bool
    skewness: float
    kurtosis: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class OutlierReport:
    """异常值检测报告"""
    outlier_indices: List[int]
    outlier_values: List[float]
    outlier_percentage: float
    detection_method: str
    threshold_lower: float
    threshold_upper: float

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_records: int
    missing_records: int
    duplicate_records: int
    outlier_records: int
    data_types: Dict[str, str]
    descriptive_stats: Dict[str, Dict[str, float]]
    validation_time: datetime

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# ============================================================================
# DataLoader 类 - 数据加载器
# ============================================================================

class DataLoader:
    """数据加载器"""

    def __init__(self, data_dir: str = "./data"):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self._cache = {}  # 数据缓存
        self._create_data_directories()

    def _create_data_directories(self) -> None:
        """创建必要的数据目录结构"""
        # 只保留必要的输出目录
        directories = [
            self.data_dir / "processed"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"基础数据目录结构已就绪: {self.data_dir}")

    def load_green_energy_data(self) -> pd.DataFrame:
        """
        加载绿电企业数据（基于 base.py 中预设的参数初始化）

        不再从外部文件加载，直接使用 base.GREEN_ENERGY_GROUPS 中定义的基准参数。

        Returns:
            pd.DataFrame: 初始化的绿电企业数据
        """
        rows = []
        for group_id, group_params in base.GREEN_ENERGY_GROUPS.items():
            count = group_params.get('count', 1)
            # 提取基础参数并移除元信息
            base_row = group_params.copy()
            base_row.pop('name', None)
            base_row.pop('count', None)

            # 根据 count 数量创建主体
            for i in range(count):
                row = base_row.copy()
                row['firm_id'] = f"{group_id}_{i+1}"
                row['group_id'] = group_id
                rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"成功基于预设参数初始化绿电企业数据，主体总数: {len(df)}")
        return df

    def load_consumer_data(self) -> pd.DataFrame:
        """
        加载用电企业数据（基于 base.py 中预设的参数初始化）

        不再从外部文件加载，直接使用 base.CONSUMER_GROUPS 中定义的基准参数。

        Returns:
            pd.DataFrame: 初始化的用电企业数据
        """
        rows = []
        for group_name, group_params in base.CONSUMER_GROUPS.items():
            count = group_params.get('count', 1)
            # 基础参数副本
            base_row = group_params.copy()
            base_row.pop('count', None)

            # 为该分组创建指定数量的主体数据
            for i in range(count):
                row = base_row.copy()
                row['consumer_id'] = f"{group_name}_{i+1}"
                # 初始碳配额计算
                row['carbon_quota'] = row['annual_consumption'] * row.get('carbon_quota_factor', 0.6)
                rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"成功基于预设参数初始化用电企业数据，主体总数: {len(df)}")
        return df

    def load_policy_texts(self) -> List[Dict[str, Any]]:
        """
        加载政府政策文本数据（基于 base.py 中的预设关键词生成）

        不再从外部文件加载，直接模拟文本分析结果。

        Returns:
            List[Dict]: 模拟的政策文本数据
        """
        simulated_texts = [{
            'file_name': 'policy_2025_summary.txt',
            'year': 2025,
            'content': " ".join(base.GOV_SENTIMENT_KEYWORDS),
            'sentiment_keywords': base.GOV_SENTIMENT_KEYWORDS,
            'is_simulated': True
        }]
        logger.info(f"使用预设关键词模拟加载政策文本数据")
        return simulated_texts

    def load_validation_data(self) -> Dict[int, Dict[str, float]]:
        """
        加载验证数据（已精简为不进行外部加载）

        Returns:
            Dict: 空字典（不再进行历史数据比对验证）
        """
        return {}

    def load_scenario_config(self, scenario_id: str) -> Dict[str, Any]:
        """
        加载情景配置（基于 base.py 中预设的情景）

        Args:
            scenario_id: 情景ID

        Returns:
            Dict: 情景配置
        """
        # 直接从 base 模块中加载预设情景
        if scenario_id in base.SCENARIO_TARGETS:
            scenario_config = base.SCENARIO_TARGETS[scenario_id].copy()
            scenario_config['id'] = scenario_id
            scenario_config['name'] = base.get_scenario_name(scenario_id)
            return scenario_config

        # 默认返回基准情景 S0 的配置
        logger.warning(f"情景 {scenario_id} 未找到，使用默认 S0 配置")
        return {
            'id': scenario_id,
            'name': f'情景{scenario_id}',
            'x1': 1.0,
            'x2': 0.0,
            'x3': 0.0,
            'x4': 0.0,
            'x5': 0.0
        }

    def load_all_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        加载所有情景配置

        Returns:
            Dict[str, Dict]: 所有情景配置
        """
        scenarios = {}

        # 首先从base模块加载
        for scenario_id in base.SCENARIO_TARGETS:
            scenarios[scenario_id] = self.load_scenario_config(scenario_id)

        # 然后从文件加载
        scenario_dir = self.data_dir / "scenarios"

        if scenario_dir.exists():
            for scenario_file in scenario_dir.glob("*.json"):
                scenario_id = scenario_file.stem
                if scenario_id not in scenarios:
                    try:
                        scenario_config = self.load_scenario_config(scenario_id)
                        scenarios[scenario_id] = scenario_config
                    except Exception as e:
                        logger.error(f"加载情景文件失败 {scenario_file}: {str(e)}")

        logger.info(f"成功加载 {len(scenarios)} 个情景配置")
        return scenarios

    def clear_cache(self) -> None:
        """清空数据缓存"""
        self._cache.clear()
        logger.info("数据缓存已清空")


# ============================================================================
# ResultSaver 类 - 结果保存器
# ============================================================================

class ResultSaver:
    """结果保存器"""

    def __init__(self, output_dir: str = "./output"):
        """
        初始化结果保存器

        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self._create_output_directories()

    def _create_output_directories(self) -> None:
        """创建输出目录结构"""
        directories = [
            self.output_dir,
            self.output_dir / "simulation_results",
            self.output_dir / "agent_data",
            self.output_dir / "policy_evolution",
            self.output_dir / "market_data",
            self.output_dir / "network_data",
            self.output_dir / "visualization",
            self.output_dir / "reports",
            self.output_dir / "logs",
            self.output_dir / "cache"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"输出目录结构已创建: {self.output_dir}")

    def save_simulation_results(self, results: Dict[str, Any], scenario_id: str) -> str:
        """
        保存仿真结果到CSV文件

        Args:
            results: 仿真结果字典
            scenario_id: 情景ID

        Returns:
            str: 保存的文件路径
        """
        # 提取时间序列数据
        if 'time_series' not in results:
            logger.error("仿真结果中缺少'time_series'键")
            return ""

        time_series = results['time_series']

        # 转换为DataFrame
        df_list = []

        for year, data in time_series.items():
            row = {'year': year}
            row.update(data)
            df_list.append(row)

        df = pd.DataFrame(df_list)

        # 确保年份列在第一列
        cols = ['year'] + [col for col in df.columns if col != 'year']
        df = df[cols]

        # 保存到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results_{scenario_id}_{timestamp}.csv"
        filepath = self.output_dir / "simulation_results" / filename

        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        # 同时保存为JSON格式供快速读取
        json_filepath = filepath.with_suffix('.json')
        df.to_json(json_filepath, orient='records', force_ascii=False)

        logger.info(f"仿真结果已保存: {filepath}")

        # 更新base.RESULTS_FILE引用（如果需要）
        base.RESULTS_FILE = str(filepath)

        return str(filepath)

    def save_agent_data(self, agent_data: Dict[str, Any], scenario_id: str) -> str:
        """
        保存主体数据

        Args:
            agent_data: 主体数据字典
            scenario_id: 情景ID

        Returns:
            str: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_data_{scenario_id}_{timestamp}.json"
        filepath = self.output_dir / "agent_data" / filename

        # 保存为JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(agent_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"主体数据已保存: {filepath}")

        # 更新base.AGENTS_FILE引用（如果需要）
        base.AGENTS_FILE = str(filepath)

        return str(filepath)

    def save_policy_evolution(self, policy_history: Dict[str, Any], scenario_id: str) -> str:
        """
        保存政策演化数据

        Args:
            policy_history: 政策演化历史数据
            scenario_id: 情景ID

        Returns:
            str: 保存的文件路径
        """
        # 提取政策演化时间序列
        if 'policy_tools' not in policy_history:
            logger.error("政策历史数据中缺少'policy_tools'键")
            return ""

        policy_tools = policy_history['policy_tools']

        # 转换为DataFrame
        df_list = []

        for year, policies in policy_tools.items():
            row = {'year': year}

            for policy_key, policy_value in policies.items():
                if isinstance(policy_value, dict) and 'value' in policy_value:
                    row[policy_key] = policy_value['value']
                else:
                    row[policy_key] = policy_value

            df_list.append(row)

        df = pd.DataFrame(df_list)

        # 确保年份列在第一列
        cols = ['year'] + [col for col in df.columns if col != 'year']
        df = df[cols]

        # 保存到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"policy_evolution_{scenario_id}_{timestamp}.csv"
        filepath = self.output_dir / "policy_evolution" / filename

        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        logger.info(f"政策演化数据已保存: {filepath}")

        # 更新base.POLICY_FILE引用（如果需要）
        base.POLICY_FILE = str(filepath)

        return str(filepath)

    def save_market_data(self, market_data: List[Dict[str, Any]], scenario_id: str) -> str:
        """
        保存市场出清数据

        Args:
            market_data: 市场数据列表
            scenario_id: 情景ID

        Returns:
            str: 保存的文件路径
        """
        if not market_data:
            logger.warning("市场数据为空，跳过保存")
            return ""

        # 转换为DataFrame
        df = pd.DataFrame(market_data)

        # 保存到CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_data_{scenario_id}_{timestamp}.csv"
        filepath = self.output_dir / "market_data" / filename

        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        logger.info(f"市场数据已保存: {filepath}")

        # 更新base.MARKET_FILE引用（如果需要）
        base.MARKET_FILE = str(filepath)

        return str(filepath)

    def save_network_data(self, network_data: Dict[str, Any], year: int, scenario_id: str) -> str:
        """
        保存网络数据（供R绘制弦图）

        Args:
            network_data: 网络数据字典
            year: 年份
            scenario_id: 情景ID

        Returns:
            str: 保存的文件路径
        """
        # 创建目录
        network_dir = self.output_dir / "network_data" / scenario_id
        network_dir.mkdir(parents=True, exist_ok=True)

        # 保存为JSON
        filename = f"network_year_{year}.json"
        filepath = network_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(network_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"网络数据已保存: {filepath} (年份: {year})")

        # 更新网络数据文件引用
        if year == 0:  # 初始年份
            base.NETWORK_DATA_FILE = str(filepath)

        return str(filepath)

    def save_evaluation_metrics(self, metrics: Dict[str, Any], scenario_id: str) -> str:
        """
        保存评估指标数据

        Args:
            metrics: 评估指标字典
            scenario_id: 情景ID

        Returns:
            str: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_metrics_{scenario_id}_{timestamp}.json"
        filepath = self.output_dir / "reports" / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"评估指标已保存: {filepath}")
        return str(filepath)

    def save_simulation_summary(self, summary: Dict[str, Any], scenario_id: str) -> str:
        """
        保存仿真摘要

        Args:
            summary: 仿真摘要字典
            scenario_id: 情景ID

        Returns:
            str: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_summary_{scenario_id}_{timestamp}.json"
        filepath = self.output_dir / "reports" / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"仿真摘要已保存: {filepath}")
        return str(filepath)

    def save_to_cache(self, data: Any, key: str) -> str:
        """
        保存数据到缓存

        Args:
            data: 要缓存的数据
            key: 缓存键

        Returns:
            str: 缓存文件路径
        """
        cache_dir = self.output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        # 使用MD5哈希作为文件名
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        filepath = cache_dir / f"{hashed_key}.pkl"

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        return str(filepath)

    def load_from_cache(self, key: str) -> Optional[Any]:
        """
        从缓存加载数据

        Args:
            key: 缓存键

        Returns:
            Optional[Any]: 缓存的数据，如果不存在则返回None
        """
        cache_dir = self.output_dir / "cache"

        # 使用MD5哈希作为文件名
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        filepath = cache_dir / f"{hashed_key}.pkl"

        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                return data
            except Exception as e:
                logger.error(f"从缓存加载数据失败 {key}: {str(e)}")

        return None

    def clear_cache(self, older_than_days: int = 7) -> int:
        """
        清理缓存文件

        Args:
            older_than_days: 清理多少天前的缓存文件

        Returns:
            int: 清理的文件数量
        """
        cache_dir = self.output_dir / "cache"

        if not cache_dir.exists():
            return 0

        count = 0
        cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 3600)

        for filepath in cache_dir.glob("*.pkl"):
            if filepath.stat().st_mtime < cutoff_time:
                try:
                    filepath.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"删除缓存文件失败 {filepath}: {str(e)}")

        logger.info(f"已清理 {count} 个缓存文件")
        return count


# ============================================================================
# DataValidator 类 - 数据验证器
# ============================================================================

class DataValidator:
    """数据验证器"""

    def __init__(self):
        """初始化数据验证器"""
        self.validation_rules = self._load_default_validation_rules()

    def _load_default_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        加载默认验证规则

        Returns:
            Dict: 验证规则字典
        """
        return {
            'green_energy': {
                'required_columns': ['firm_id', 'asset_total', 'installed_capacity'],
                'numeric_columns': ['asset_total', 'revenue', 'net_profit', 'installed_capacity'],
                'range_checks': {
                    'asset_total': (0, 1e6),  # 0到100万亿元
                    'installed_capacity': (0, 10000),  # 0到10000万千瓦
                    'unit_cost': (0.1, 0.5)  # 0.1到0.5元/千瓦时
                },
                'non_negative': ['asset_total', 'installed_capacity', 'revenue']
            },
            'consumer': {
                'required_columns': ['consumer_id', 'annual_consumption', 'industry'],
                'numeric_columns': ['annual_consumption', 'annual_revenue', 'operating_cost'],
                'range_checks': {
                    'annual_consumption': (0, 1e6),  # 0到100亿千瓦时
                    'green_preference': (0, 1)  # 0到1
                },
                'non_negative': ['annual_consumption', 'annual_revenue']
            }
        }

    def validate_input_data(self, data: pd.DataFrame, data_type: str) -> ValidationResult:
        """
        验证输入数据质量

        Args:
            data: 输入数据框
            data_type: 数据类型（'green_energy'或'consumer'）

        Returns:
            ValidationResult: 验证结果
        """
        issues = []
        statistics = {}

        if data_type not in self.validation_rules:
            return ValidationResult(
                is_valid=False,
                message=f"未知的数据类型: {data_type}",
                issues=[{'type': 'config_error', 'message': f"未知的数据类型: {data_type}"}],
                statistics={}
            )

        rules = self.validation_rules[data_type]

        # 1. 检查必要列
        missing_columns = [col for col in rules.get('required_columns', [])
                           if col not in data.columns]

        if missing_columns:
            issues.append({
                'type': 'missing_columns',
                'message': f"缺少必要的列: {missing_columns}",
                'details': {'missing_columns': missing_columns}
            })

        # 2. 检查缺失值
        total_records = len(data)
        missing_counts = data.isnull().sum()

        for column, count in missing_counts.items():
            if count > 0:
                percentage = (count / total_records) * 100
                issues.append({
                    'type': 'missing_values',
                    'message': f"列 '{column}' 有 {count} 个缺失值 ({percentage:.1f}%)",
                    'details': {
                        'column': column,
                        'count': int(count),
                        'percentage': float(percentage)
                    }
                })

        statistics['total_records'] = total_records
        statistics['missing_counts'] = missing_counts.to_dict()

        # 3. 检查重复记录
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            issues.append({
                'type': 'duplicate_records',
                'message': f"发现 {duplicate_count} 条重复记录",
                'details': {'count': int(duplicate_count)}
            })

        statistics['duplicate_count'] = duplicate_count

        # 4. 检查数值列的异常值
        numeric_columns = rules.get('numeric_columns', [])
        numeric_columns = [col for col in numeric_columns if col in data.columns]

        for column in numeric_columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                # 检查范围
                if 'range_checks' in rules and column in rules['range_checks']:
                    min_val, max_val = rules['range_checks'][column]
                    out_of_range = ((data[column] < min_val) | (data[column] > max_val)).sum()

                    if out_of_range > 0:
                        issues.append({
                            'type': 'out_of_range',
                            'message': f"列 '{column}' 有 {out_of_range} 个值超出范围 [{min_val}, {max_val}]",
                            'details': {
                                'column': column,
                                'count': int(out_of_range),
                                'min_allowed': min_val,
                                'max_allowed': max_val
                            }
                        })

                # 检查非负性
                if 'non_negative' in rules and column in rules['non_negative']:
                    negative_count = (data[column] < 0).sum()

                    if negative_count > 0:
                        issues.append({
                            'type': 'negative_values',
                            'message': f"列 '{column}' 有 {negative_count} 个负值",
                            'details': {'column': column, 'count': int(negative_count)}
                        })

        # 5. 数据统计
        statistics['data_types'] = data.dtypes.astype(str).to_dict()

        descriptive_stats = {}
        for column in numeric_columns:
            if column in data.columns and pd.api.types.is_numeric_dtype(data[column]):
                col_data = data[column].dropna()
                if len(col_data) > 0:
                    descriptive_stats[column] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median()),
                        'q1': float(col_data.quantile(0.25)),
                        'q3': float(col_data.quantile(0.75))
                    }

        statistics['descriptive_stats'] = descriptive_stats

        # 判断数据是否有效
        critical_issues = [issue for issue in issues
                           if issue['type'] in ['missing_columns', 'severe_outliers']]
        is_valid = len(critical_issues) == 0

        message = f"数据验证{'通过' if is_valid else '失败'}, 发现 {len(issues)} 个问题"

        return ValidationResult(
            is_valid=is_valid,
            message=message,
            issues=issues,
            statistics=statistics
        )

    def check_distribution(self, data: pd.Series, distribution_type: str = "normal") -> DistributionTest:
        """
        检查数据分布

        Args:
            data: 数据序列
            distribution_type: 分布类型（"normal"、"lognormal"等）

        Returns:
            DistributionTest: 分布检验结果
        """
        # 移除缺失值
        clean_data = data.dropna()

        if len(clean_data) < 3:
            return DistributionTest(
                distribution_type=distribution_type,
                test_statistic=0.0,
                p_value=1.0,
                is_normal=False,
                skewness=0.0,
                kurtosis=0.0
            )

        # 计算偏度和峰度
        skewness = float(stats.skew(clean_data))
        kurtosis = float(stats.kurtosis(clean_data))

        # 正态性检验
        if distribution_type == "normal":
            # Shapiro-Wilk检验（适用于小样本）
            if len(clean_data) < 5000:
                stat, p_value = stats.shapiro(clean_data)
            else:
                # Kolmogorov-Smirnov检验（适用于大样本）
                stat, p_value = stats.kstest(clean_data, 'norm',
                                             args=(clean_data.mean(), clean_data.std()))

            is_normal = p_value > 0.05

            return DistributionTest(
                distribution_type=distribution_type,
                test_statistic=float(stat),
                p_value=float(p_value),
                is_normal=is_normal,
                skewness=skewness,
                kurtosis=kurtosis
            )
        else:
            # 对于其他分布，暂时只计算偏度和峰度
            return DistributionTest(
                distribution_type=distribution_type,
                test_statistic=0.0,
                p_value=1.0,
                is_normal=False,
                skewness=skewness,
                kurtosis=kurtosis
            )

    def detect_outliers(self, data: pd.Series, method: str = "iqr", threshold: float = 1.5) -> OutlierReport:
        """
        异常值检测

        Args:
            data: 数据序列
            method: 检测方法（"iqr"、"zscore"、"percentile"）
            threshold: 阈值

        Returns:
            OutlierReport: 异常值检测报告
        """
        # 移除缺失值
        clean_data = data.dropna()

        if len(clean_data) < 3:
            return OutlierReport(
                outlier_indices=[],
                outlier_values=[],
                outlier_percentage=0.0,
                detection_method=method,
                threshold_lower=0.0,
                threshold_upper=0.0
            )

        outlier_indices = []
        outlier_values = []

        if method == "iqr":
            # IQR方法
            q1 = clean_data.quantile(0.25)
            q3 = clean_data.quantile(0.75)
            iqr = q3 - q1

            threshold_lower = q1 - threshold * iqr
            threshold_upper = q3 + threshold * iqr

            outliers = (clean_data < threshold_lower) | (clean_data > threshold_upper)
            outlier_indices = clean_data[outliers].index.tolist()
            outlier_values = clean_data[outliers].tolist()

        elif method == "zscore":
            # Z-score方法
            mean = clean_data.mean()
            std = clean_data.std()

            if std > 0:
                z_scores = np.abs((clean_data - mean) / std)
                outliers = z_scores > threshold

                outlier_indices = clean_data[outliers].index.tolist()
                outlier_values = clean_data[outliers].tolist()

                threshold_lower = mean - threshold * std
                threshold_upper = mean + threshold * std
            else:
                threshold_lower = mean
                threshold_upper = mean

        elif method == "percentile":
            # 百分位数方法
            threshold_lower = clean_data.quantile(threshold / 100)
            threshold_upper = clean_data.quantile(1 - threshold / 100)

            outliers = (clean_data < threshold_lower) | (clean_data > threshold_upper)
            outlier_indices = clean_data[outliers].index.tolist()
            outlier_values = clean_data[outliers].tolist()

        else:
            raise ValueError(f"未知的异常值检测方法: {method}")

        outlier_percentage = (len(outlier_indices) / len(clean_data)) * 100

        return OutlierReport(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            outlier_percentage=outlier_percentage,
            detection_method=method,
            threshold_lower=float(threshold_lower),
            threshold_upper=float(threshold_upper)
        )

    def impute_missing_values(self, data: pd.DataFrame, method: str = "median",
                              group_column: Optional[str] = None) -> pd.DataFrame:
        """
        缺失值填充

        Args:
            data: 原始数据框
            method: 填充方法（"newton"、"median"、"mean"、"linear"）
            group_column: 分组列名，用于按组填充

        Returns:
            pd.DataFrame: 填充后的数据框
        """
        df_imputed = data.copy()

        # 识别数值列
        numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if df_imputed[column].isnull().any():
                missing_count = df_imputed[column].isnull().sum()

                if method == "newton":
                    # 牛顿插值法
                    try:
                        # 获取非缺失值的索引和值
                        known_indices = df_imputed[column].dropna().index
                        known_values = df_imputed.loc[known_indices, column].values

                        # 对每个缺失值进行插值
                        for idx in df_imputed[df_imputed[column].isnull()].index:
                            # 使用牛顿插值法
                            interpolated = fun.newton_interpolation(
                                known_indices, known_values, idx
                            )
                            df_imputed.loc[idx, column] = interpolated

                    except Exception as e:
                        logger.warning(f"牛顿插值失败，列 '{column}'，使用中位数填充: {str(e)}")
                        method = "median"  # 降级到中位数方法

                if method == "median":
                    if group_column and group_column in df_imputed.columns:
                        # 按组填充中位数
                        df_imputed[column] = df_imputed.groupby(group_column)[column] \
                            .transform(lambda x: x.fillna(x.median()))
                    else:
                        # 整体中位数
                        df_imputed[column] = df_imputed[column].fillna(df_imputed[column].median())

                elif method == "mean":
                    if group_column and group_column in df_imputed.columns:
                        # 按组填充均值
                        df_imputed[column] = df_imputed.groupby(group_column)[column] \
                            .transform(lambda x: x.fillna(x.mean()))
                    else:
                        # 整体均值
                        df_imputed[column] = df_imputed[column].fillna(df_imputed[column].mean())

                elif method == "linear":
                    # 线性插值
                    df_imputed[column] = df_imputed[column].interpolate(method='linear')

                    # 对于首尾的缺失值，使用前向/后向填充
                    df_imputed[column] = df_imputed[column].fillna(method='ffill').fillna(method='bfill')

                logger.info(f"列 '{column}' 的 {missing_count} 个缺失值已使用 '{method}' 方法填充")

        return df_imputed

    def generate_data_quality_report(self, data: pd.DataFrame, data_type: str) -> DataQualityReport:
        """
        生成数据质量报告

        Args:
            data: 数据框
            data_type: 数据类型

        Returns:
            DataQualityReport: 数据质量报告
        """
        validation_result = self.validate_input_data(data, data_type)

        # 计算异常值
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        for column in numeric_columns:
            outlier_report = self.detect_outliers(data[column], method="iqr")
            outlier_counts[column] = len(outlier_report.outlier_indices)

        total_outliers = sum(outlier_counts.values())

        # 数据分布检验
        distribution_tests = {}
        for column in numeric_columns[:5]:  # 只检查前5个数值列
            if column in data.columns:
                distribution_tests[column] = self.check_distribution(data[column], "normal").to_dict()

        # 描述性统计
        descriptive_stats = {}
        for column in numeric_columns:
            if column in data.columns:
                col_data = data[column].dropna()
                if len(col_data) > 0:
                    descriptive_stats[column] = {
                        'count': int(len(col_data)),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'missing': int(data[column].isnull().sum())
                    }

        return DataQualityReport(
            total_records=len(data),
            missing_records=int(data.isnull().sum().sum()),
            duplicate_records=int(data.duplicated().sum()),
            outlier_records=total_outliers,
            data_types=data.dtypes.astype(str).to_dict(),
            descriptive_stats=descriptive_stats,
            validation_time=datetime.now()
        )


# ============================================================================
# BatchProcessor 类 - 批量处理器
# ============================================================================

class BatchProcessor:
    """批量处理器"""

    def __init__(self, scenarios: Optional[List[str]] = None,
                 output_dir: str = "./output"):
        """
        初始化批量处理器

        Args:
            scenarios: 情景列表，如果为None则使用base模块中的所有情景
            output_dir: 输出目录
        """
        self.scenarios = scenarios or list(base.SCENARIO_TARGETS.keys())
        self.output_dir = Path(output_dir)
        self.result_saver = ResultSaver(output_dir)

        # 创建批量运行目录
        self.batch_dir = self.output_dir / "batch_results"
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"批量处理器初始化完成，包含 {len(self.scenarios)} 个情景")

    def run_batch_simulations(self, years: int = 50, parallel: bool = True,
                              max_workers: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """
        批量运行所有情景仿真

        Args:
            years: 仿真年数
            parallel: 是否并行运行
            max_workers: 最大工作进程数

        Returns:
            Dict[str, Dict]: 各情景仿真结果
        """
        logger.info(f"开始批量仿真，情景数: {len(self.scenarios)}, 年数: {years}")

        results = {}
        start_time = datetime.now()

        if parallel and len(self.scenarios) > 1:
            # 并行运行
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            max_workers = max_workers or min(mp.cpu_count(), len(self.scenarios))

            logger.info(f"使用并行模式，工作进程数: {max_workers}")

            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # 准备参数
                    params = [(scenario_id, years) for scenario_id in self.scenarios]

                    # 提交任务
                    future_to_scenario = {
                        executor.submit(self._run_single_scenario_wrapper, p[0], p[1]): p[0]
                        for p in params
                    }

                    # 收集结果
                    for future in future_to_scenario:
                        scenario_id = future_to_scenario[future]
                        try:
                            result = future.result(timeout=3600)  # 1小时超时
                            results[scenario_id] = result
                            logger.info(f"情景 {scenario_id} 仿真完成")
                        except Exception as e:
                            logger.error(f"情景 {scenario_id} 仿真失败: {str(e)}")
                            results[scenario_id] = {'error': str(e)}

            except Exception as e:
                logger.error(f"并行处理失败: {str(e)}")
                # 降级为顺序运行
                logger.info("降级为顺序运行")
                results = self._run_sequential(years)
        else:
            # 顺序运行
            results = self._run_sequential(years)

        # 保存批量结果
        self._save_batch_results(results, years)

        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"批量仿真完成，总耗时: {elapsed_time:.2f}秒")

        return results

    def _run_single_scenario_wrapper(self, scenario_id: str, years: int) -> Dict[str, Any]:
        """
        单个情景仿真的包装函数（用于并行处理）

        Args:
            scenario_id: 情景ID
            years: 仿真年数

        Returns:
            Dict: 仿真结果
        """
        try:
            # 设置随机种子以确保可重复性
            random_seed = hash(scenario_id) % 10000
            np.random.seed(random_seed)

            # 运行仿真
            result = math_go.run_scenario_simulation(
                scenario_id=scenario_id,
                years=years,
                random_seed=random_seed
            )

            return result
        except Exception as e:
            logger.error(f"情景 {scenario_id} 仿真包装器异常: {str(e)}")
            raise

    def _run_sequential(self, years: int) -> Dict[str, Dict[str, Any]]:
        """
        顺序运行所有情景

        Args:
            years: 仿真年数

        Returns:
            Dict[str, Dict]: 各情景仿真结果
        """
        results = {}

        for i, scenario_id in enumerate(self.scenarios, 1):
            logger.info(f"运行情景 {scenario_id} ({i}/{len(self.scenarios)})")

            try:
                # 设置随机种子
                random_seed = hash(scenario_id) % 10000
                np.random.seed(random_seed)

                # 运行仿真
                result = math_go.run_scenario_simulation(
                    scenario_id=scenario_id,
                    years=years,
                    random_seed=random_seed
                )

                results[scenario_id] = result
                logger.info(f"情景 {scenario_id} 仿真完成")

            except Exception as e:
                logger.error(f"情景 {scenario_id} 仿真失败: {str(e)}")
                results[scenario_id] = {'error': str(e)}

        return results

    def _save_batch_results(self, results: Dict[str, Dict[str, Any]], years: int) -> None:
        """
        保存批量运行结果

        Args:
            results: 各情景结果
            years: 仿真年数
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = f"batch_{years}years_{timestamp}"

        # 1. 保存汇总结果
        summary = {
            'batch_id': batch_id,
            'timestamp': timestamp,
            'years': years,
            'scenarios': list(results.keys()),
            'successful_scenarios': [sid for sid, res in results.items() if 'error' not in res],
            'failed_scenarios': [sid for sid, res in results.items() if 'error' in res],
            'execution_time': datetime.now().isoformat()
        }

        summary_file = self.batch_dir / f"{batch_id}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 2. 保存详细结果
        for scenario_id, result in results.items():
            if 'error' not in result:
                # 保存各情景结果
                scenario_file = self.batch_dir / f"{batch_id}_{scenario_id}.json"
                with open(scenario_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        # 3. 生成比较报告
        comparison_results = self._generate_comparison_report(results)
        comparison_file = self.batch_dir / f"{batch_id}_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)

        logger.info(f"批量结果已保存: {summary_file}")

    def generate_comparison_report(self, results: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        生成批量比较报告

        Args:
            results: 仿真结果，如果为None则加载最新结果

        Returns:
            Dict: 比较报告
        """
        if results is None:
            # 尝试加载最新结果
            results = self._load_latest_batch_results()

        if not results:
            return {'error': '没有可用的结果数据'}

        comparison_results = self._generate_comparison_report(results)

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / "reports" / f"comparison_report_{timestamp}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)

        logger.info(f"比较报告已生成: {report_file}")

        return comparison_results

    def _generate_comparison_report(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成比较报告（内部实现）

        Args:
            results: 各情景仿真结果

        Returns:
            Dict: 比较报告
        """
        comparison = {
            'scenarios': {},
            'key_metrics': {},
            'ranking': {},
            'created_at': datetime.now().isoformat()
        }

        # 提取各情景的关键指标
        key_metrics = [
            'total_carbon_reduction',  # 累计碳减排
            'avg_green_price',  # 平均绿电价格
            'avg_penetration_rate',  # 平均渗透率
            'total_policy_cost',  # 总政策成本
            'final_social_welfare',  # 最终社会福利
            'avg_capacity_utilization'  # 平均产能利用率
        ]

        for scenario_id, result in results.items():
            if 'error' in result:
                comparison['scenarios'][scenario_id] = {'error': result['error']}
                continue

            scenario_summary = {
                'name': result.get('scenario_name', scenario_id),
                'final_year': result.get('final_year', 0),
                'metrics': {}
            }

            # 提取指标
            for metric in key_metrics:
                if metric in result.get('summary', {}):
                    scenario_summary['metrics'][metric] = result['summary'][metric]
                elif 'evaluation_metrics' in result and metric in result['evaluation_metrics']:
                    scenario_summary['metrics'][metric] = result['evaluation_metrics'][metric]

            comparison['scenarios'][scenario_id] = scenario_summary

        # 计算关键指标的比较
        for metric in key_metrics:
            metric_values = {}

            for scenario_id, scenario_data in comparison['scenarios'].items():
                if 'metrics' in scenario_data and metric in scenario_data['metrics']:
                    metric_values[scenario_id] = scenario_data['metrics'][metric]

            if metric_values:
                # 计算统计量
                values_list = list(metric_values.values())
                comparison['key_metrics'][metric] = {
                    'values': metric_values,
                    'mean': float(np.mean(values_list)),
                    'std': float(np.std(values_list)),
                    'min': float(np.min(values_list)),
                    'max': float(np.max(values_list)),
                    'range': float(np.max(values_list) - np.min(values_list))
                }

        # 生成排名
        ranking_metrics = {
            'total_carbon_reduction': 'desc',  # 降序（越大越好）
            'avg_green_price': 'asc',  # 升序（越小越好）
            'avg_penetration_rate': 'desc',  # 降序（越大越好）
            'total_policy_cost': 'asc',  # 升序（越小越好）
            'final_social_welfare': 'desc'  # 降序（越大越好）
        }

        ranking = {}

        for metric, order in ranking_metrics.items():
            if metric in comparison['key_metrics']:
                values = comparison['key_metrics'][metric]['values']

                # 排序
                if order == 'desc':
                    sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=True)
                else:
                    sorted_items = sorted(values.items(), key=lambda x: x[1])

                ranking[metric] = {
                    'order': order,
                    'ranking': [item[0] for item in sorted_items],
                    'values': {item[0]: item[1] for item in sorted_items}
                }

        comparison['ranking'] = ranking

        return comparison

    def _load_latest_batch_results(self) -> Dict[str, Dict[str, Any]]:
        """
        加载最新的批量结果

        Returns:
            Dict[str, Dict]: 批量结果
        """
        if not self.batch_dir.exists():
            return {}

        # 查找最新的批次摘要文件
        summary_files = list(self.batch_dir.glob("*_summary.json"))

        if not summary_files:
            return {}

        # 按修改时间排序
        summary_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_summary = summary_files[0]

        try:
            with open(latest_summary, 'r', encoding='utf-8') as f:
                summary = json.load(f)

            batch_id = summary.get('batch_id', '')
            scenarios = summary.get('scenarios', [])

            # 加载各情景结果
            results = {}
            for scenario_id in scenarios:
                result_file = self.batch_dir / f"{batch_id}_{scenario_id}.json"

                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        results[scenario_id] = json.load(f)

            return results

        except Exception as e:
            logger.error(f"加载批量结果失败: {str(e)}")
            return {}

    def export_all_results(self, results: Dict[str, Dict[str, Any]],
                           export_format: str = "excel") -> List[str]:
        """
        导出所有结果到指定格式

        Args:
            results: 仿真结果
            export_format: 导出格式（"excel"、"csv"、"json"）

        Returns:
            List[str]: 导出的文件路径列表
        """
        exported_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format == "excel":
            # 导出到Excel工作簿，每个情景一个工作表
            excel_file = self.batch_dir / f"all_results_{timestamp}.xlsx"

            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for scenario_id, result in results.items():
                    if 'time_series' in result:
                        # 提取时间序列数据
                        time_series = result['time_series']
                        df_list = []

                        for year, data in time_series.items():
                            row = {'year': year}
                            row.update(data)
                            df_list.append(row)

                        df = pd.DataFrame(df_list)

                        # 写入工作表
                        sheet_name = scenario_id[:31]  # Excel工作表名限制31字符
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

            exported_files.append(str(excel_file))

        elif export_format == "csv":
            # 每个情景导出为一个CSV文件
            for scenario_id, result in results.items():
                if 'time_series' in result:
                    time_series = result['time_series']
                    df_list = []

                    for year, data in time_series.items():
                        row = {'year': year}
                        row.update(data)
                        df_list.append(row)

                    df = pd.DataFrame(df_list)

                    csv_file = self.batch_dir / f"{scenario_id}_results_{timestamp}.csv"
                    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
                    exported_files.append(str(csv_file))

        elif export_format == "json":
            # 导出为单个JSON文件
            json_file = self.batch_dir / f"all_results_{timestamp}.json"

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            exported_files.append(str(json_file))

        else:
            raise ValueError(f"不支持的导出格式: {export_format}")

        logger.info(f"已导出 {len(exported_files)} 个文件，格式: {export_format}")

        return exported_files


# ============================================================================
# 工具函数
# ============================================================================

def create_default_data_files() -> Dict[str, str]:
    """
    创建默认数据文件（用于测试和演示）

    Returns:
        Dict[str, str]: 创建的文件路径字典
    """
    import tempfile

    data_dir = Path(tempfile.mkdtemp())
    created_files = {}

    try:
        # 创建绿电企业数据
        green_energy_data = pd.DataFrame({
            'firm_id': range(1, 16),
            'asset_total': np.random.lognormal(6, 1, 15),  # 对数正态分布
            'revenue': np.random.lognormal(5, 0.8, 15),
            'operating_cost': np.random.lognormal(4.5, 0.7, 15),
            'installed_capacity': np.random.lognormal(3, 0.5, 15) * 100,
            'group_id': np.random.choice([0, 1, 2], 15, p=[0.5, 0.3, 0.2]),
            'region': np.random.choice(['华东', '华北', '华南', '华中', '西北'], 15)
        })

        green_energy_file = data_dir / "green_energy_data.csv"
        green_energy_data.to_csv(green_energy_file, index=False, encoding='utf-8-sig')
        created_files['green_energy'] = str(green_energy_file)

        # 创建用电企业数据
        consumer_data = pd.DataFrame({
            'consumer_id': range(1, 31),
            'industry': np.random.choice(['高科技', '制造业', '服务业', '其他'], 30),
            'annual_consumption': np.random.lognormal(3, 0.6, 30) * 10,
            'annual_revenue': np.random.lognormal(5, 0.8, 30),
            'operating_cost': np.random.lognormal(4.5, 0.7, 30),
            'green_preference': np.random.beta(2, 2, 30),  # Beta分布，集中在0.5附近
            'region': np.random.choice(['华东', '华北', '华南', '华中', '西北'], 30)
        })

        consumer_file = data_dir / "consumer_data.csv"
        consumer_data.to_csv(consumer_file, index=False, encoding='utf-8-sig')
        created_files['consumer'] = str(consumer_file)

        # 创建验证数据
        validation_data = {
            2024: {
                'avg_green_price': 0.42,
                'green_power_volume': 4500,
                'penetration_rate': 0.15,
                'carbon_reduction': 28000
            },
            2025: {
                'avg_green_price': 0.41,
                'green_power_volume': 5200,
                'penetration_rate': 0.18,
                'carbon_reduction': 32000
            }
        }

        validation_file = data_dir / "validation_data.json"
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, ensure_ascii=False, indent=2)
        created_files['validation'] = str(validation_file)

        logger.info(f"默认数据文件已创建: {data_dir}")

    except Exception as e:
        logger.error(f"创建默认数据文件失败: {str(e)}")

    return created_files


# ============================================================================
# 主函数（测试用）
# ============================================================================

def main():
    """主函数（用于测试）"""
    print("SG-ABM 输入输出与数据管理模块测试")
    print("=" * 50)

    # 创建测试数据文件
    print("1. 创建测试数据文件...")
    created_files = create_default_data_files()

    if not created_files:
        print("创建测试数据文件失败")
        return

    # 测试DataLoader
    print("\n2. 测试DataLoader...")
    data_loader = DataLoader(data_dir=Path(created_files['green_energy']).parent)

    try:
        green_energy_data = data_loader.load_green_energy_data(created_files['green_energy'])
        print(f"绿电企业数据加载成功，记录数: {len(green_energy_data)}")
        print(f"列名: {list(green_energy_data.columns)}")
    except Exception as e:
        print(f"绿电企业数据加载失败: {str(e)}")

    try:
        consumer_data = data_loader.load_consumer_data(created_files['consumer'])
        print(f"用电企业数据加载成功，记录数: {len(consumer_data)}")
        print(f"列名: {list(consumer_data.columns)}")
    except Exception as e:
        print(f"用电企业数据加载失败: {str(e)}")

    # 测试DataValidator
    print("\n3. 测试DataValidator...")
    data_validator = DataValidator()

    validation_result = data_validator.validate_input_data(green_energy_data, 'green_energy')
    print(f"数据验证结果: {validation_result.message}")
    print(f"发现问题数: {len(validation_result.issues)}")

    # 测试ResultSaver
    print("\n4. 测试ResultSaver...")
    result_saver = ResultSaver(output_dir="./test_output")

    # 创建测试数据
    test_results = {
        'scenario_id': 'S0',
        'time_series': {
            2025: {
                'avg_green_price': 0.42,
                'green_power_volume': 4500,
                'penetration_rate': 0.15,
                'carbon_reduction': 28000
            },
            2026: {
                'avg_green_price': 0.41,
                'green_power_volume': 5200,
                'penetration_rate': 0.18,
                'carbon_reduction': 32000
            }
        }
    }

    saved_file = result_saver.save_simulation_results(test_results, 'S0')
    print(f"结果保存到: {saved_file}")

    # 测试BatchProcessor
    print("\n5. 测试BatchProcessor...")
    batch_processor = BatchProcessor(scenarios=['S0', 'S1', 'S2'],
                                     output_dir="./test_output")

    print(f"批量处理器初始化完成，包含 {len(batch_processor.scenarios)} 个情景")

    print("\n测试完成！")

    # 清理测试文件
    import shutil
    test_output_dir = Path("./test_output")
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)
        print("测试输出目录已清理")


if __name__ == "__main__":
    main()