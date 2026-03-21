"""
math_go.py - SG-ABM模型数学计算和优化求解模块

该文件包含SG-ABM模型中所有的数学计算、优化求解、演化博弈逻辑，
以及多主体决策过程中的数值计算。

作者：Liang
日期：2026年
版本：1.0
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import optimize
import pulp
from dataclasses import dataclass

from core import base, fun


# ============================================================================
# 1. 数据结构定义
# ============================================================================

@dataclass
class MarketClearingResult:
    """市场出清结果"""
    cleared_prices: List[float]
    cleared_quantities: List[float]
    total_cleared_quantity: float
    clearing_price: float
    green_cert_price: float
    surplus: float
    shortage: float
    supply_curves: List[Tuple[float, float]]  # 供给曲线 [(价格, 数量)]
    demand_curves: List[Tuple[float, float]]  # 需求曲线 [(价格, 数量)]


@dataclass
class StackelbergEquilibrium:
    """Stackelberg博弈均衡结果"""
    p_g0: float  # 第一阶段价格
    q_d_pred: float  # 预测需求量
    d_g_actual: float  # 实际需求量
    p_g1: float  # 第三阶段价格
    q_g_supply: float  # 供给量
    green_energy_profit: float  # 绿电企业利润
    consumer_profit: float  # 用电企业利润


@dataclass
class EWALearningState:
    """EWA学习算法状态"""
    attractions: Dict[float, float]  # 各策略吸引力 {策略值: 吸引力}
    experience_weight: float  # 经验权重N
    policy_value: float  # 当前政策值
    selection_probabilities: Dict[float, float]  # 各策略选择概率


# ============================================================================
# 2. Stackelberg博弈求解器
# ============================================================================

class StackelbergSolver:
    """Stackelberg博弈求解器"""

    def __init__(self, k: float, alpha: float, beta: float,
                 investment_coef: float, tax_rate: float,
                 x2: float, x3: float):
        """
        初始化求解器

        参数:
            k: 单位生产成本
            alpha: 成本锚定权重
            beta: 初始报价延续性权重
            investment_coef: 投资系数
            tax_rate: 所得税税率
            x2: 设备投资补贴比例
            x3: 增值税即征即退比例
        """
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.investment_coef = investment_coef
        self.tax_rate = tax_rate
        self.x2 = x2
        self.x3 = x3

    def stage1_profit_function(self, p_g0: float, q_t_1: float,
                               previous_investments: List[float]) -> float:
        """
        第一阶段利润函数

        参数:
            p_g0: 初步价格
            q_t_1: 上期绿电出售数量
            previous_investments: 过去投资额列表

        返回:
            预期利润
        """
        # 预测需求量
        q_pred = fun.stage1_demand_prediction(p_g0)

        # 计算投资额
        m = fun.investment_decision(q_pred, q_t_1, previous_investments,
                                    self.investment_coef, base.EQUIPMENT_LIFESPAN)

        # 计算预期利润
        profit = fun.green_energy_profit(p_g0, q_pred, self.k, m,
                                         self.x2, self.x3, self.tax_rate)

        return -profit  # 返回负值用于最小化

    def solve_stage1(self, q_t_1: float, previous_investments: List[float],
                     price_bounds: Tuple[float, float] = (0.1, 1.0),
                     method: str = 'bounded') -> Tuple[float, float]:
        """
        求解第一阶段最优价格和预期需求量

        参数:
            q_t_1: 上期绿电出售数量
            previous_investments: 过去投资额列表
            price_bounds: 价格搜索范围
            method: 优化方法

        返回:
            (最优价格, 预期需求量)
        """
        if method == 'bounded':
            # 有界优化方法
            result = optimize.minimize_scalar(
                lambda p: self.stage1_profit_function(p, q_t_1, previous_investments),
                bounds=price_bounds,
                method='bounded'
            )

            if result.success:
                p_optimal = result.x
                q_pred = fun.stage1_demand_prediction(p_optimal)
                return p_optimal, q_pred
            else:
                # 如果优化失败，使用搜索法
                return self._solve_stage1_by_search(q_t_1, previous_investments, price_bounds)

        elif method == 'search':
            return self._solve_stage1_by_search(q_t_1, previous_investments, price_bounds)
        else:
            raise ValueError(f"未知的优化方法: {method}")

    def _solve_stage1_by_search(self, q_t_1: float, previous_investments: List[float],
                                price_bounds: Tuple[float, float]) -> Tuple[float, float]:
        """通过网格搜索求解第一阶段"""
        n_points = 100
        prices = np.linspace(price_bounds[0], price_bounds[1], n_points)

        best_profit = -float('inf')
        best_price = price_bounds[0]

        for p in prices:
            profit = -self.stage1_profit_function(p, q_t_1, previous_investments)
            if profit > best_profit:
                best_profit = profit
                best_price = p

        q_pred = fun.stage1_demand_prediction(best_price)
        return best_price, q_pred

    def solve_stage3(self, c_g: float, p_g0: float, d_g_actual: float) -> float:
        """
        求解第三阶段最终价格

        参数:
            c_g: 单位边际生产成本
            p_g0: 第一阶段价格
            d_g_actual: 实际市场需求量

        返回:
            最终价格
        """
        # 计算市场边际价格
        p_s = fun.calculate_market_price(d_g_actual)

        # 计算最终价格
        p_g1 = fun.stage3_final_price(c_g, p_g0, p_s, self.alpha, self.beta)

        return p_g1


class ConsumerOptimizer:
    """用电企业优化求解器"""

    def __init__(self, p_b: float, x4: float, x5: float,
                 c_o_coef: float, r_c: float):
        """
        初始化优化器

        参数:
            p_b: 传统电力价格
            x4: 强制消纳配额比例
            x5: 消费端补贴比例
            c_o_coef: 运营成本系数
            r_c: 售电收入
        """
        self.p_b = p_b
        self.x4 = x4
        self.x5 = x5
        self.c_o_coef = c_o_coef
        self.r_c = r_c

    def solve_optimal_purchase(self, p_e: float, d_j: float) -> Tuple[float, float, float]:
        """
        求解最优绿电采购量

        参数:
            p_e: 绿电价格
            d_j: 总电力需求

        返回:
            (绿电采购量, 传统电力采购量, 最大利润)
        """
        # 使用解析解 (KKT条件)
        effective_p_e = (1 - self.x5) * p_e

        if effective_p_e <= self.p_b:
            # 情况1: 全部购买绿电
            q_g_optimal = d_j
            e_c_optimal = 0.0
        else:
            # 情况2: 仅满足配额要求
            q_g_optimal = self.x4 * d_j
            e_c_optimal = d_j - q_g_optimal

        # 计算运营成本
        c_o = fun.calculate_operating_cost(d_j, self.c_o_coef)

        # 计算利润
        profit = fun.electricity_consumer_profit(
            self.r_c, p_e, q_g_optimal, self.p_b, e_c_optimal, c_o, d_j
        )

        return q_g_optimal, e_c_optimal, profit

    def solve_with_linear_programming(self, p_e: float, d_j: float) -> Tuple[float, float, float]:
        """
        使用线性规划求解最优采购量

        参数:
            p_e: 绿电价格
            d_j: 总电力需求

        返回:
            (绿电采购量, 传统电力采购量, 最大利润)
        """
        # 创建线性规划问题
        prob = pulp.LpProblem("Consumer_Optimization", pulp.LpMaximize)

        # 决策变量
        q_g = pulp.LpVariable('q_g', lowBound=0, upBound=d_j)  # 绿电采购量
        e_c = pulp.LpVariable('e_c', lowBound=0, upBound=d_j)  # 传统电力采购量

        # 目标函数
        c_o = self.c_o_coef * d_j  # 运营成本
        effective_p_e = (1 - self.x5) * p_e

        # 将价格转换为万元/万千瓦时
        effective_p_e_per_10k = effective_p_e * 10000
        p_b_per_10k = self.p_b * 10000

        # 最大化利润 = 收入 - 成本
        prob += (self.r_c - effective_p_e_per_10k * q_g - p_b_per_10k * e_c - c_o)

        # 约束条件
        # 1. 电力需求约束
        prob += (q_g + e_c == d_j)

        # 2. 配额约束
        prob += (q_g >= self.x4 * (q_g + e_c))

        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] == 'Optimal':
            q_g_optimal = pulp.value(q_g)
            e_c_optimal = pulp.value(e_c)
            profit = pulp.value(prob.objective)
            return q_g_optimal, e_c_optimal, profit
        else:
            # 如果线性规划失败，使用解析解
            return self.solve_optimal_purchase(p_e, d_j)


# ============================================================================
# 3. EWA学习算法实现
# ============================================================================

class EWALearner:
    """EWA学习算法实现"""

    def __init__(self, policy_name: str, initial_value: float,
                 value_range: Tuple[float, float], num_strategies: int = 10,
                 phi: float = base.GOV_PHI, delta: float = base.GOV_DELTA,
                 lambda_a: float = base.GOV_LAMBDA_A, kappa: float = base.GOV_KAPPA):
        """
        初始化EWA学习器

        参数:
            policy_name: 政策名称
            initial_value: 初始政策值
            value_range: 政策值范围
            num_strategies: 策略离散化数量
            phi: 折旧率
            delta: 想象力参数
            lambda_a: 反应灵敏度参数
            kappa: 经验权重调整参数
        """
        self.policy_name = policy_name
        self.value_range = value_range
        self.phi = phi
        self.delta = delta
        self.lambda_a = lambda_a
        self.kappa = kappa

        # 离散化策略空间
        self.strategies = np.linspace(value_range[0], value_range[1], num_strategies)

        # 初始化EWA状态
        self.state = EWALearningState(
            attractions={s: base.GOV_EWA_INITIAL['A0'] for s in self.strategies},
            experience_weight=base.GOV_EWA_INITIAL['N0'],
            policy_value=initial_value,
            selection_probabilities={s: 1.0 / num_strategies for s in self.strategies}
        )

        # 记录历史
        self.history = {
            'policy_values': [initial_value],
            'attractions': [self.state.attractions.copy()],
            'probabilities': [self.state.selection_probabilities.copy()]
        }

    def update(self, welfare_change: float, chosen_value: Optional[float] = None) -> float:
        """
        更新EWA学习器

        参数:
            welfare_change: 社会福利变化ΔJ
            chosen_value: 实际选择的政策值（如果为None，使用当前值）

        返回:
            更新后的政策值
        """
        if chosen_value is None:
            chosen_value = self.state.policy_value

        # 更新吸引力
        new_attractions = {}
        for s in self.strategies:
            is_chosen = (abs(s - chosen_value) < 1e-6)

            # 修正：根据fun.ewa_attraction_update函数的新签名调用
            # 注意：fun.ewa_attraction_update现在只返回吸引力，不返回经验权重
            new_attraction = fun.ewa_attraction_update(
                previous_attractions=self.state.attractions[s],
                previous_choice=1 if is_chosen else 0,  # 1表示选中，0表示未选中
                payoff=welfare_change,
                phi=self.phi,
                delta=self.delta,
                kappa=self.kappa,
                lambda_a=self.lambda_a
            )

            # 如果返回的是数组（修正后的函数可能返回数组），取第一个元素
            if hasattr(new_attraction, '__len__') and len(new_attraction) > 0:
                new_attractions[s] = new_attraction[0] if isinstance(new_attraction,
                                                                     (list, np.ndarray)) else new_attraction
            else:
                new_attractions[s] = new_attraction

        # 更新经验权重（使用学习算法逻辑）
        self.state.experience_weight = self.phi * self.state.experience_weight * self.kappa + 1

        self.state.attractions = new_attractions

        # 更新选择概率
        attractions_array = np.array(list(new_attractions.values()))
        probabilities = fun.policy_selection_probability(attractions_array, self.lambda_a)

        # 转换为字典
        self.state.selection_probabilities = {
            s: p for s, p in zip(self.strategies, probabilities)
        }

        # 选择新策略
        new_policy_value = self._select_strategy()
        self.state.policy_value = new_policy_value

        # 记录历史
        self.history['policy_values'].append(new_policy_value)
        self.history['attractions'].append(self.state.attractions.copy())
        self.history['probabilities'].append(self.state.selection_probabilities.copy())

        return new_policy_value

    def _select_strategy(self) -> float:
        """根据选择概率选择策略"""
        strategies = list(self.state.selection_probabilities.keys())
        probabilities = list(self.state.selection_probabilities.values())

        # 确保概率和为1
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()

        # 随机选择
        chosen_index = np.random.choice(len(strategies), p=probabilities)
        return strategies[chosen_index]

    def get_history(self) -> Dict[str, List]:
        """获取历史记录"""
        return self.history

    def get_current_state(self) -> EWALearningState:
        """获取当前状态"""
        return self.state


class GovernmentPolicyOptimizer:
    """政府政策优化器"""

    def __init__(self, scenario_id: str = 'S3'):
        """
        初始化政府政策优化器

        参数:
            scenario_id: 情景ID
        """
        self.scenario_id = scenario_id
        self.scenario_target = base.get_scenario_target(scenario_id)

        # 为每个政策工具创建EWA学习器
        self.ewa_learners = {}

        for policy_name in ['x1', 'x2', 'x3', 'x4', 'x5']:
            initial_value = base.GOV_POLICY_INITIAL[policy_name]
            value_range = base.GOV_POLICY_RANGES[policy_name]

            self.ewa_learners[policy_name] = EWALearner(
                policy_name=policy_name,
                initial_value=initial_value,
                value_range=value_range,
                num_strategies=10
            )

    def update_policies(self, welfare_changes: Dict[str, float]) -> Dict[str, float]:
        """
        更新所有政策工具

        参数:
            welfare_changes: 各政策工具带来的社会福利变化

        返回:
            更新后的政策工具值
        """
        new_policies = {}

        for policy_name, learner in self.ewa_learners.items():
            welfare_change = welfare_changes.get(policy_name, 0.0)
            new_value = learner.update(welfare_change)
            new_policies[policy_name] = new_value

        return new_policies

    def calculate_welfare_changes(self, carbon_reduction: float,
                                  green_power_volume: float,
                                  policy_costs: Dict[str, float]) -> Dict[str, float]:
        """
        计算各政策工具带来的社会福利变化

        参数:
            carbon_reduction: 碳减排量
            green_power_volume: 绿电交易量
            policy_costs: 各政策工具的成本

        返回:
            各政策工具的社会福利变化
        """
        welfare_changes = {}

        for policy_name in self.ewa_learners.keys():
            # 计算该政策工具的社会福利
            welfare = fun.government_welfare(
                carbon_reduction, green_power_volume, policy_costs.get(policy_name, 0.0)
            )

            # 计算社会福利变化（这里简化为直接使用社会福利值）
            # 实际应用中可能需要更复杂的计算
            welfare_changes[policy_name] = welfare

        return welfare_changes

    def get_current_policies(self) -> Dict[str, float]:
        """获取当前政策工具值"""
        return {name: learner.state.policy_value for name, learner in self.ewa_learners.items()}

    def get_policy_history(self) -> Dict[str, List[float]]:
        """获取所有政策工具的历史值"""
        history = {}
        for policy_name, learner in self.ewa_learners.items():
            history[policy_name] = learner.history['policy_values']
        return history


# ============================================================================
# 4. 多主体优化协调器
# ============================================================================

class MultiAgentCoordinator:
    """多主体优化协调器"""

    def __init__(self, num_green_energy_firms: Tuple[int, int] = (10, 5),
                 num_consumers: int = 10):
        """
        初始化协调器

        参数:
            num_green_energy_firms: (低规模组数量, 中规模组数量)
            num_consumers: 用电企业数量
        """
        self.num_green_energy_firms = num_green_energy_firms
        self.num_consumers = num_consumers

        # 初始化GDP数据
        self.gdp_history = [base.GDP_INITIAL * 0.95, base.GDP_INITIAL]  # 上上期和上期GDP

        # 初始化政策优化器
        self.policy_optimizer = GovernmentPolicyOptimizer()

        # 初始化绿电企业
        self.green_energy_firms = self._initialize_green_energy_firms()

        # 初始化用电企业
        self.consumers = self._initialize_consumers()

        # 记录历史
        self.history = {
            'year': [],
            'policies': [],
            'market_results': [],
            'welfare': [],
            'carbon_reduction': [],
            'green_power_volume': []
        }

    def _initialize_green_energy_firms(self) -> List[Dict]:
        """初始化绿电企业"""
        firms = []
        firm_id = 0

        # 低规模组
        for _ in range(self.num_green_energy_firms[0]):
            group_params = base.get_initial_green_energy_params(0)
            firms.append({
                'id': firm_id,
                'group': 0,
                **group_params,
                'previous_investments': [group_params['fixed_asset_investment']] * base.EQUIPMENT_LIFESPAN,
                'q_t_1': group_params['power_generation'] * 10000,  # 亿千瓦时 -> 万千瓦时
                'solver': StackelbergSolver(
                    k=base.GREEN_ENERGY_BEHAVIOR['K'],
                    alpha=base.GREEN_ENERGY_BEHAVIOR['alpha'],
                    beta=base.GREEN_ENERGY_BEHAVIOR['beta'],
                    investment_coef=base.INVESTMENT_COEFFICIENT,
                    tax_rate=fun.income_tax_rate(0, base.GOV_POLICY_INITIAL['x1']),
                    x2=base.GOV_POLICY_INITIAL['x2'],
                    x3=base.GOV_POLICY_INITIAL['x3']
                )
            })
            firm_id += 1

        # 中规模组
        for _ in range(self.num_green_energy_firms[1]):
            group_params = base.get_initial_green_energy_params(1)
            firms.append({
                'id': firm_id,
                'group': 1,
                **group_params,
                'previous_investments': [group_params['fixed_asset_investment']] * base.EQUIPMENT_LIFESPAN,
                'q_t_1': group_params['power_generation'] * 10000,  # 亿千瓦时 -> 万千瓦时
                'solver': StackelbergSolver(
                    k=base.GREEN_ENERGY_BEHAVIOR['K'],
                    alpha=base.GREEN_ENERGY_BEHAVIOR['alpha'],
                    beta=base.GREEN_ENERGY_BEHAVIOR['beta'],
                    investment_coef=base.INVESTMENT_COEFFICIENT,
                    tax_rate=fun.income_tax_rate(0, base.GOV_POLICY_INITIAL['x1']),
                    x2=base.GOV_POLICY_INITIAL['x2'],
                    x3=base.GOV_POLICY_INITIAL['x3']
                )
            })
            firm_id += 1

        return firms

    def _initialize_consumers(self) -> List[Dict]:
        """初始化用电企业"""
        consumers = []

        # 使用分组数据中的售电量作为初始需求参考
        total_power_sales = sum(firm['power_sales'] for firm in self.green_energy_firms)

        for consumer_id in range(self.num_consumers):
            # 将兆瓦时转换为万千瓦时
            avg_sales = total_power_sales / self.num_consumers / 10  # 兆瓦时 -> 万千瓦时

            # 初始需求基于平均售电量，加上随机变化
            initial_demand = avg_sales * np.random.uniform(0.8, 1.2)

            consumers.append({
                'id': consumer_id,
                'd_j': initial_demand,
                'd_j_history': [initial_demand],
                'optimizer': ConsumerOptimizer(
                    p_b=base.TRADITIONAL_POWER_PRICE,
                    x4=base.GOV_POLICY_INITIAL['x4'],
                    x5=base.GOV_POLICY_INITIAL['x5'],
                    c_o_coef=base.OPERATING_COST_COEFFICIENT,
                    r_c=initial_demand * base.MARKET_POWER_PRICE * 10000  # 售电收入
                )
            })

        return consumers

    def run_one_period(self, year: int) -> Dict[str, Any]:
        """
        运行一个周期（一年）

        参数:
            year: 当前年份

        返回:
            该周期的结果
        """
        # 获取当前政策
        current_policies = self.policy_optimizer.get_current_policies()

        # 第一阶段：绿电企业初步报价
        stage1_results = []
        for firm in self.green_energy_firms:
            # 更新税率
            tax_rate = fun.income_tax_rate(year, current_policies['x1'])
            firm['solver'].tax_rate = tax_rate
            firm['solver'].x2 = current_policies['x2']
            firm['solver'].x3 = current_policies['x3']

            # 求解第一阶段
            p_g0, q_pred = firm['solver'].solve_stage1(
                firm['q_t_1'], firm['previous_investments']
            )

            stage1_results.append({
                'firm_id': firm['id'],
                'p_g0': p_g0,
                'q_pred': q_pred,
                'investment': fun.investment_decision(
                    q_pred, firm['q_t_1'], firm['previous_investments'],
                    base.INVESTMENT_COEFFICIENT, base.EQUIPMENT_LIFESPAN
                )
            })

        # 计算平均初步价格
        avg_p_g0 = np.mean([r['p_g0'] for r in stage1_results])

        # 第二阶段：用电企业决策
        stage2_results = []
        total_green_demand = 0

        # 预测GDP
        if len(self.gdp_history) >= 2:
            gdp_forecast = fun.gdp_forecast(
                self.gdp_history[-1], self.gdp_history[-2]
            )
        else:
            gdp_forecast = base.GDP_INITIAL * 1.05  # 假设增长5%

        # 更新GDP历史
        self.gdp_history.append(gdp_forecast)

        for consumer in self.consumers:
            # 预测电力需求
            if len(consumer['d_j_history']) >= 2:
                d_j_pred = fun.electricity_demand_forecast(
                    consumer['d_j_history'][-1],
                    self.gdp_history[-1],
                    self.gdp_history[-2]
                )
            else:
                d_j_pred = consumer['d_j'] * np.random.uniform(0.95, 1.05)

            consumer['d_j'] = d_j_pred
            consumer['d_j_history'].append(d_j_pred)

            # 使用fun.py中的函数直接计算
            q_g_optimal, e_c_optimal, profit = fun.optimal_green_power_purchase(
                p_e=avg_p_g0,
                p_b=base.TRADITIONAL_POWER_PRICE,
                d_j=d_j_pred,
                x4=current_policies['x4'],
                x5=current_policies['x5'],
                c_o=fun.calculate_operating_cost(d_j_pred),
                r_c=d_j_pred * base.MARKET_POWER_PRICE * 10000
            )

            stage2_results.append({
                'consumer_id': consumer['id'],
                'q_g': q_g_optimal,
                'e_c': e_c_optimal,
                'profit': profit,
                'd_j': d_j_pred
            })

            total_green_demand += q_g_optimal
        # 第三阶段：绿电企业最终定价
        stage3_results = []
        supply_curves = []

        for idx, firm in enumerate(self.green_energy_firms):
            stage1_result = stage1_results[idx]

            # 计算单位边际生产成本（简化为K）
            c_g = base.GREEN_ENERGY_BEHAVIOR['K']

            # 求解第三阶段
            p_g1 = firm['solver'].solve_stage3(
                c_g, stage1_result['p_g0'], total_green_demand
            )

            # 计算最优供给量
            capacity_constraint = stage1_result['investment'] / base.INVESTMENT_COEFFICIENT
            q_supply = fun.optimal_supply_quantity(
                p_g1, base.GREEN_ENERGY_BEHAVIOR['K'],
                stage1_result['investment'], capacity_constraint
            )

            # 计算利润
            profit = fun.green_energy_profit(
                p_g1, q_supply, base.GREEN_ENERGY_BEHAVIOR['K'],
                stage1_result['investment'], current_policies['x2'],
                current_policies['x3'], firm['solver'].tax_rate
            )

            stage3_results.append({
                'firm_id': firm['id'],
                'p_g1': p_g1,
                'q_supply': q_supply,
                'profit': profit,
                'investment': stage1_result['investment']
            })

            # 更新企业状态
            firm['q_t_1'] = q_supply
            firm['previous_investments'].append(stage1_result['investment'])
            if len(firm['previous_investments']) > base.EQUIPMENT_LIFESPAN + 5:
                firm['previous_investments'] = firm['previous_investments'][-base.EQUIPMENT_LIFESPAN - 5:]

            # 添加到供给曲线
            supply_curves.append((p_g1, q_supply))

        # 市场出清
        market_result = fun.market_clearing(supply_curves, total_green_demand)

        # 计算社会福利指标
        green_power_volume = market_result['total_cleared_quantity']
        carbon_reduction = fun.calculate_carbon_reduction(green_power_volume)

        # 计算政策成本（简化）
        policy_costs = {
            'x1': 0.0,  # 税收减免成本
            'x2': sum(r['investment'] for r in stage1_results) * current_policies['x2'],
            'x3': green_power_volume * avg_p_g0 * 10000 * base.VAT_RATE * current_policies['x3'],
            'x4': 0.0,  # 配额约束成本（间接）
            'x5': total_green_demand * avg_p_g0 * 10000 * current_policies['x5']
        }

        total_policy_cost = sum(policy_costs.values())

        # 计算社会福利
        welfare = fun.government_welfare(
            carbon_reduction, green_power_volume, total_policy_cost
        )

        # 更新政策
        welfare_changes = self.policy_optimizer.calculate_welfare_changes(
            carbon_reduction, green_power_volume, policy_costs
        )
        new_policies = self.policy_optimizer.update_policies(welfare_changes)

        # 记录结果
        result = {
            'year': year,
            'policies': current_policies,
            'market_result': market_result,
            'green_power_volume': green_power_volume,
            'carbon_reduction': carbon_reduction,
            'policy_costs': policy_costs,
            'total_policy_cost': total_policy_cost,
            'welfare': welfare,
            'stage1_results': stage1_results,
            'stage2_results': stage2_results,
            'stage3_results': stage3_results,
            'avg_p_g0': avg_p_g0,
            'total_green_demand': total_green_demand,
            'gdp': self.gdp_history[-1]
        }

        # 更新历史记录
        self.history['year'].append(year)
        self.history['policies'].append(current_policies)
        self.history['market_results'].append(market_result)
        self.history['welfare'].append(welfare)
        self.history['carbon_reduction'].append(carbon_reduction)
        self.history['green_power_volume'].append(green_power_volume)

        return result

    def run_simulation(self, years: int = base.SIMULATION_YEARS) -> Dict[str, Any]:
        """
        运行完整仿真

        参数:
            years: 仿真年数

        返回:
            仿真结果
        """
        results = []

        for year in range(1, years + 1):
            print(f"运行第 {year} 年...")
            result = self.run_one_period(year)
            results.append(result)

        # 计算综合评估指标
        evaluation = self._calculate_evaluation_metrics(results)

        return {
            'results': results,
            'evaluation': evaluation,
            'history': self.history,
            'policy_history': self.policy_optimizer.get_policy_history()
        }

    def _calculate_evaluation_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """计算综合评估指标"""
        if not results:
            return {}

        # 累计值
        total_carbon_reduction = sum(r['carbon_reduction'] for r in results)
        total_green_power_volume = sum(r['green_power_volume'] for r in results)
        total_policy_cost = sum(r['total_policy_cost'] for r in results)

        # 平均价格
        avg_clearing_prices = [r['market_result']['clearing_price'] for r in results]
        avg_clearing_price = np.mean(avg_clearing_prices) if avg_clearing_prices else 0

        # 绿电渗透率
        # 需要总用电量数据，这里简化为假设
        total_electricity_consumption = sum(
            sum(consumer['d_j'] for consumer in r['stage2_results']) for r in results
        )
        penetration_rate = fun.calculate_penetration_rate(
            total_green_power_volume, total_electricity_consumption
        )

        # 产能利用率（简化）
        total_generation = total_green_power_volume
        total_capacity = sum(firm['installed_capacity'] for firm in self.green_energy_firms)
        capacity_utilization = fun.calculate_capacity_utilization(
            total_generation, total_capacity
        )

        # 单位碳减排政策成本
        cost_per_carbon = fun.calculate_cost_per_carbon_reduction(
            total_policy_cost, total_carbon_reduction
        )

        # 单位绿电激发成本
        cost_per_green_power = fun.calculate_cost_per_green_power(
            total_policy_cost, total_green_power_volume
        )

        return {
            'total_carbon_reduction': total_carbon_reduction,
            'total_green_power_volume': total_green_power_volume,
            'total_policy_cost': total_policy_cost,
            'avg_clearing_price': avg_clearing_price,
            'penetration_rate': penetration_rate,
            'capacity_utilization': capacity_utilization,
            'cost_per_carbon': cost_per_carbon,
            'cost_per_green_power': cost_per_green_power
        }


# ============================================================================
# 5. 数值优化工具
# ============================================================================

class NumericalOptimizer:
    """数值优化工具类"""

    @staticmethod
    def solve_kkt_conditions(gradient_func, constraint_funcs,
                             initial_guess, bounds=None):
        """
        求解KKT条件

        参数:
            gradient_func: 梯度函数
            constraint_funcs: 约束条件函数列表
            initial_guess: 初始猜测
            bounds: 变量边界

        返回:
            优化结果
        """

        # 定义KKT条件函数
        def kkt_conditions(x_lambda):
            n_vars = len(initial_guess)
            n_constraints = len(constraint_funcs)

            x = x_lambda[:n_vars]
            lambda_vars = x_lambda[n_vars:n_vars + n_constraints]

            # 梯度条件
            grad_conditions = gradient_func(x)
            for constr_idx, constr_func in enumerate(constraint_funcs):
                grad_conditions += lambda_vars[constr_idx] * constr_func(x)

            # 互补松弛条件
            compl_conditions = []
            for constr_idx, constr_func in enumerate(constraint_funcs):
                compl_conditions.append(lambda_vars[constr_idx] * constr_func(x))

            return np.concatenate([grad_conditions, compl_conditions])

        # 初始猜测（包括拉格朗日乘子）
        initial_guess_full = np.concatenate([initial_guess, np.zeros(len(constraint_funcs))])

        # 求解KKT条件
        result = optimize.root(kkt_conditions, initial_guess_full)

        if result.success:
            n_vars = len(initial_guess)
            solution = result.x[:n_vars]
            multipliers = result.x[n_vars:]
            return solution, multipliers
        else:
            raise ValueError("KKT条件求解失败")

    @staticmethod
    def solve_linear_program(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
                             bounds=None, method='highs'):
        """
        求解线性规划问题

        参数:
            c: 目标函数系数
            A_ub: 不等式约束矩阵
            b_ub: 不等式约束右侧
            A_eq: 等式约束矩阵
            b_eq: 等式约束右侧
            bounds: 变量边界
            method: 求解方法

        返回:
            优化结果
        """
        if bounds is None:
            bounds = [(0, None)] * len(c)

        result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                  bounds=bounds, method=method)

        if result.success:
            return result.x, result.fun
        else:
            raise ValueError(f"线性规划求解失败: {result.message}")

    @staticmethod
    def solve_quadratic_program(P, q, G=None, h=None, A=None, b=None):
        """
        求解二次规划问题

        参数:
            P: 二次项系数矩阵
            q: 一次项系数向量
            G: 不等式约束矩阵
            h: 不等式约束右侧
            A: 等式约束矩阵
            b: 等式约束右侧

        返回:
            优化结果
        """
        try:
            from cvxopt import matrix, solvers
            solvers.options['show_progress'] = False

            # 转换为cvxopt格式
            P_mat = matrix(P)
            q_mat = matrix(q)

            if G is not None and h is not None:
                G_mat = matrix(G)
                h_mat = matrix(h)
            else:
                G_mat = None
                h_mat = None

            if A is not None and b is not None:
                A_mat = matrix(A)
                b_mat = matrix(b)
            else:
                A_mat = None
                b_mat = None

            # 求解
            solution = solvers.qp(P_mat, q_mat, G_mat, h_mat, A_mat, b_mat)

            if solution['status'] == 'optimal':
                return np.array(solution['x']).flatten(), solution['primal objective']
            else:
                raise ValueError(f"二次规划求解失败: {solution['status']}")
        except ImportError:
            # 如果cvxopt不可用，使用scipy
            from scipy.optimize import minimize

            n = len(q)

            # 目标函数
            def objective(x):
                return 0.5 * x.T @ P @ x + q.T @ x

            # 约束条件
            constraints = []

            if G is not None and h is not None:
                # 创建不等式约束
                for row_idx in range(G.shape[0]):
                    # 捕获当前行的索引和值
                    G_row = G[row_idx].copy()
                    h_val = h[row_idx]
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, G_row=G_row, h_val=h_val: h_val - G_row @ x
                    })

            if A is not None and b is not None:
                # 创建等式约束
                for row_idx in range(A.shape[0]):
                    # 捕获当前行的索引和值
                    A_row = A[row_idx].copy()
                    b_val = b[row_idx]
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda x, A_row=A_row, b_val=b_val: A_row @ x - b_val
                    })

            # 初始猜测
            x0 = np.zeros(n)

            # 求解
            result = minimize(objective, x0, constraints=constraints)

            if result.success:
                return result.x, result.fun
            else:
                raise ValueError(f"二次规划求解失败: {result.message}")


# ============================================================================
# 6. 模型验证和敏感性分析
# ============================================================================

class ModelValidator:
    """模型验证器"""

    @staticmethod
    def validate_with_actual_data(simulated_results: List[Dict],
                                  actual_data: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """
        用实际数据验证模型

        参数:
            simulated_results: 模拟结果
            actual_data: 实际数据 {年份: {指标: 值}}

        返回:
            验证结果
        """
        validation_results = {}

        for year_data in simulated_results:
            year = year_data['year']
            if year in actual_data:
                actual = actual_data[year]
                simulated = {
                    'avg_price': year_data['market_result']['clearing_price'],
                    'green_power_volume': year_data['green_power_volume'],
                    'penetration_rate': fun.calculate_penetration_rate(
                        year_data['green_power_volume'],
                        sum(consumer['d_j'] for consumer in year_data['stage2_results'])
                    )
                }

                # 计算误差
                errors = fun.validate_model_output(simulated, actual)

                # 检查是否在容忍范围内
                within_tolerance = fun.check_error_within_tolerance(
                    errors, base.ACCEPTABLE_ERROR
                )

                validation_results[year] = {
                    'simulated': simulated,
                    'actual': actual,
                    'errors': errors,
                    'within_tolerance': within_tolerance
                }

        return validation_results

    @staticmethod
    def sensitivity_analysis(base_params: Dict, param_ranges: Dict,
                             n_samples: int = 10) -> Dict[str, List[float]]:
        """
        敏感性分析

        参数:
            base_params: 基准参数
            param_ranges: 参数范围 {参数名: (最小值, 最大值)}
            n_samples: 每个参数的采样数

        返回:
            敏感性分析结果
        """
        results = {}

        for param_name, (min_val, max_val) in param_ranges.items():
            param_values = np.linspace(min_val, max_val, n_samples)
            outcome_values = []

            for val in param_values:
                # 创建参数副本并修改
                test_params = base_params.copy()
                test_params[param_name] = val

                # 运行测试（这里需要根据实际情况实现）
                # outcome = run_test_with_params(test_params)
                # outcome_values.append(outcome)
                pass

            results[param_name] = outcome_values

        return results


# ============================================================================
# 7. 主计算函数
# ============================================================================

def run_scenario_simulation(scenario_id: str = 'S3', years: int = base.SIMULATION_YEARS,
                            random_seed: int = base.RANDOM_SEED) -> Dict[str, Any]:
    """
    运行指定情景的仿真

    参数:
        scenario_id: 情景ID
        years: 仿真年数
        random_seed: 随机种子

    返回:
        仿真结果
    """
    # 设置随机种子
    np.random.seed(random_seed)

    # 创建协调器
    coordinator = MultiAgentCoordinator()

    # 运行仿真
    results = coordinator.run_simulation(years)

    # 添加情景信息
    results['scenario_id'] = scenario_id
    results['scenario_name'] = base.get_scenario_name(scenario_id)

    return results


def compare_scenarios(scenario_ids: List[str] = None,
                      years: int = base.SIMULATION_YEARS) -> Dict[str, Any]:
    """
    比较多个情景

    参数:
        scenario_ids: 情景ID列表
        years: 仿真年数

    返回:
        情景比较结果
    """
    if scenario_ids is None:
        scenario_ids = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']

    scenario_results = {}

    for scenario_id in scenario_ids:
        print(f"\n运行情景 {scenario_id}: {base.get_scenario_name(scenario_id)}")
        results = run_scenario_simulation(scenario_id, years)
        scenario_results[scenario_id] = results

    # 比较评估指标
    comparison = {}
    for scenario_id, results in scenario_results.items():
        comparison[scenario_id] = results['evaluation']

    return {
        'scenario_results': scenario_results,
        'comparison': comparison
    }


if __name__ == "__main__":
    """测试模块"""
    print("SG-ABM模型数学计算模块测试:")

    # 测试Stackelberg求解器
    solver = StackelbergSolver(
        k=0.2187, alpha=0.2317, beta=0.3904,
        investment_coef=4.55, tax_rate=0.15,
        x2=0.1, x3=0.5
    )

    p_optimal, q_pred = solver.solve_stage1(
        q_t_1=1000,
        previous_investments=[100, 150, 200, 180, 160]
    )
    print(f"第一阶段最优价格: {p_optimal:.4f}, 预期需求量: {q_pred:.2f}")

    # 测试EWA学习器
    ewa_learner = EWALearner(
        policy_name='x4',
        initial_value=0.125,
        value_range=(0.1, 0.3)
    )

    new_value = ewa_learner.update(welfare_change=100.0)
    print(f"EWA学习器更新后的政策值: {new_value:.4f}")

    # 测试协调器
    print("\n运行一个周期的仿真测试...")
    coordinator = MultiAgentCoordinator(num_green_energy_firms=(2, 1), num_consumers=3)
    result = coordinator.run_one_period(year=1)
    print(f"第一年绿电交易量: {result['green_power_volume']:.2f} 万千瓦时")
    print(f"第一年碳减排量: {result['carbon_reduction']:.2f} 吨CO2")
    print(f"第一年社会福利: {result['welfare']:.2f}")

    print("\n所有测试完成!")