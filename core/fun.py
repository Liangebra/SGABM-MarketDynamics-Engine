"""
fun.py - SG-ABM模型数学函数定义文件

该文件包含SG-ABM模型中所有的数学逻辑、函数和计算过程，
包括政府、绿电企业、用电企业等主体的决策函数和市场机制。

作者：Liang
日期：2026年
版本：1.0
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union

''

from core import base



# ============================================================================
# 1. 通用数学函数
# ============================================================================

def newton_interpolation(x: np.ndarray, y: np.ndarray, x_target: float) -> float:
    """
    牛顿插值法

    参数:
        x: 已知时间点数组
        y: 已知数值数组
        x_target: 目标时间点

    返回:
        插值结果
    """
    n = len(x)
    # 计算差商
    f = np.zeros((n, n))
    f[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            f[i, j] = (f[i + 1, j - 1] - f[i, j - 1]) / (x[i + j] - x[i])

    # 计算插值
    result = f[0, 0]
    product = 1.0

    for j in range(1, n):
        product *= (x_target - x[j - 1])
        result += f[0, j] * product

    return result


# ============================================================================
# 2. 政府主体函数 (G)
# ============================================================================

def government_welfare(carbon_reduction: float, green_power_volume: float,
                       policy_cost: float) -> float:
    """
    政府社会福利函数

    J^G = αE + βT - γC

    参数:
        carbon_reduction: 碳减排量E (吨CO2)
        green_power_volume: 绿电交易量T (万千瓦时)
        policy_cost: 政策成本C (万元)

    返回:
        社会福利值
    """
    return (base.GOV_ALPHA * carbon_reduction +
            base.GOV_BETA * green_power_volume -
            base.GOV_GAMMA * policy_cost)


def calculate_carbon_reduction(green_power_volume: float) -> float:
    """
    计算碳减排量

    E = T × EF_grid

    参数:
        green_power_volume: 绿电上网电量T (万千瓦时)

    返回:
        碳减排量 (吨CO2)
    """
    # 将万千瓦时转换为兆瓦时
    volume_mwh = green_power_volume * base.UNIT_CONVERSION['万千瓦时_to_兆瓦时']
    return volume_mwh * base.GRID_EMISSION_FACTOR


def calculate_policy_cost(subsidies: float, tax_reduction: float,
                          consumer_subsidies: float) -> float:
    """
    计算政策成本

    C = Σ(S_subsidy + S_tax) + ΣS_consumer

    参数:
        subsidies: 补贴总额 (万元)
        tax_reduction: 税收减免总额 (万元)
        consumer_subsidies: 消费补贴总额 (万元)

    返回:
        政策成本 (万元)
    """
    return subsidies + tax_reduction + consumer_subsidies


def ewa_attraction_update(previous_attractions, previous_choice, payoff,
                          phi, delta, kappa, lambda_a):
    """EWA吸引力更新 - 数值稳定版本"""

    # 修复：确保previous_attractions是数组
    if isinstance(previous_attractions, (int, float)):
        # 如果是单个数值，转换为包含一个元素的数组
        previous_attractions = np.array([previous_attractions])
    elif not isinstance(previous_attractions, np.ndarray):
        # 如果是列表，转换为数组
        previous_attractions = np.array(previous_attractions)

    # 初始化新吸引力数组
    n_strategies = len(previous_attractions)
    new_attractions = np.zeros(n_strategies)

    # 确保previous_choice是整数且在有效范围内
    if previous_choice >= n_strategies:
        previous_choice = 0  # 默认为第一个策略

    # 确保payoff不是太大
    if abs(payoff) > 1e6:
        payoff = np.sign(payoff) * 1e6

    # 对于每个策略
    for i in range(n_strategies):
        if i == previous_choice:
            # 更新逻辑的第一部分
            new_attractions[i] = (phi * previous_attractions[i] * kappa +
                                  payoff) / (phi * kappa + 1)
        else:
            # 更新逻辑的第二部分
            new_attractions[i] = (phi * previous_attractions[i] * kappa +
                                  delta * payoff) / (phi * kappa + 1)

    # 防止吸引力值过大
    max_attraction = np.max(np.abs(new_attractions))
    if max_attraction > 100:
        # 缩放吸引力值
        scaling_factor = 100 / max_attraction
        new_attractions = new_attractions * scaling_factor

    return new_attractions

def policy_selection_probability(attractions, lambda_a):
    """计算政策选择概率（logit响应函数）- 数值稳定版本"""
    # 数值稳定的softmax实现
    if lambda_a == 0:
        # 如果lambda_a为0，返回均匀分布
        return np.ones(len(attractions)) / len(attractions)

    # 缩放吸引力值
    scaled = lambda_a * attractions

    # 减去最大值以防止溢出
    scaled_shifted = scaled - np.max(scaled)

    # 计算指数
    exp_values = np.exp(scaled_shifted)

    # 计算总和
    sum_exp = np.sum(exp_values)

    # 避免除零
    if sum_exp == 0 or np.isnan(sum_exp):
        # 返回均匀分布
        return np.ones(len(attractions)) / len(attractions)

    # 计算概率
    probabilities = exp_values / sum_exp

    # 确保概率有效（非负且和为1）
    probabilities = np.maximum(probabilities, 0)
    probabilities = probabilities / np.sum(probabilities)

    return probabilities

def income_tax_rate(operation_years: int, x1: float, base_tax_rate: float = base.BASE_INCOME_TAX_RATE) -> float:
    """
    绿电企业所得税实际税率计算

    考虑"三免三减半"及地区优惠

    参数:
        operation_years: 企业运营年限
        x1: 所得税优惠强度
        base_tax_rate: 基准税率 (高新技术企业适用税率)

    返回:
        实际税率
    """
    if operation_years <= 3:
        effective_rate = 0.0  # 三免
    elif operation_years <= 6:
        effective_rate = 0.5 * base_tax_rate  # 三减半
    else:
        effective_rate = base_tax_rate

    # 应用优惠强度调整
    adjusted_rate = effective_rate * (1 - x1)

    # 确保税率在合理范围内
    return max(0.0, min(adjusted_rate, 1.0))


# ============================================================================
# 3. 绿电企业函数 (GE)
# ============================================================================

def green_energy_profit(p_e: float, q_d: float, k: float, m: float,
                        x2: float, x3: float, tax_rate: float) -> float:
    """
    绿电企业利润函数

    π_t^g = { [1 - v_e·(1 - x3)]·p_e·Q_d - K·Q_d - (1 - x2)·M } · (1 - τ_g)

    参数:
        p_e: 绿电出售价格 (元/千瓦时)
        q_d: 绿电出售数量 (万千瓦时)
        k: 单位生产成本 (元/千瓦时)
        m: 固定资产投资 (万元)
        x2: 设备投资补贴比例
        x3: 增值税即征即退比例
        tax_rate: 所得税实际税率τ_g

    返回:
        净利润 (万元)
    """
    # 将价格和成本转换为万元/万千瓦时 (1万千瓦时 = 10000千瓦时)
    p_e_per_10k = p_e * 10000  # 元/千瓦时 -> 元/万千瓦时
    k_per_10k = k * 10000  # 元/千瓦时 -> 元/万千瓦时

    # 计算收入 (考虑增值税即征即退)
    revenue = (1 - base.VAT_RATE * (1 - x3)) * p_e_per_10k * q_d

    # 计算成本
    cost = k_per_10k * q_d + (1 - x2) * m

    # 税前利润
    profit_before_tax = revenue - cost

    # 税后利润
    profit_after_tax = profit_before_tax * (1 - tax_rate)

    return profit_after_tax


def investment_decision(q_t: float, q_t_1: float, previous_investments: List[float],
                        investment_coef: float, equipment_lifespan: int) -> float:
    """
    绿电企业固定资产投资决策

    参数:
        q_t: 当期绿电出售数量 (万千瓦时)
        q_t_1: 上期绿电出售数量 (万千瓦时)
        previous_investments: 过去几年的投资额列表 (万元)
        investment_coef: 投资系数ϱ (万元/万千瓦时)
        equipment_lifespan: 设备寿命 (年)

    返回:
        当期固定资产投资额 (万元)
    """
    # 确保previous_investments长度足够
    if len(previous_investments) < equipment_lifespan + 1:
        # 如果历史数据不足，用0填充
        padded_investments = previous_investments + [0] * (equipment_lifespan + 1 - len(previous_investments))
    else:
        padded_investments = previous_investments

    # 计算过去equipment_lifespan年的总投资
    sum_past_investments = sum(padded_investments[-equipment_lifespan:])

    if q_t > q_t_1:
        # 需求增长情况
        m_t = (q_t * investment_coef - sum_past_investments +
               padded_investments[-(equipment_lifespan + 1)])
    else:
        # 需求下降或不变情况
        excess_capacity = sum_past_investments - q_t * investment_coef
        reset_investment = padded_investments[-(equipment_lifespan + 1)] - excess_capacity
        m_t = max(0, reset_investment)

    return m_t


def stage1_demand_prediction(p_g0: float, intercept: float = base.DEMAND_FUNCTION['intercept'],
                             slope: float = base.DEMAND_FUNCTION['slope']) -> float:
    """
    第一阶段绿电企业反需求函数

    ln(Q_d') = intercept + slope·ln(p_g^0)

    参数:
        p_g0: 初步绿证报价 (元/千瓦时)
        intercept: 截距项
        slope: 斜率

    返回:
        预期需求量 (万千瓦时)
    """
    # 避免对0或负数取对数
    if p_g0 <= 0:
        p_g0 = 0.001

    ln_q = intercept + slope * math.log(p_g0)
    return math.exp(ln_q)


def stage1_optimal_price(k: float, x2: float, x3: float, tax_rate: float,
                         investment_coef: float, previous_investments: List[float],
                         q_t_1: float, price_range: Tuple[float, float] = (0.1, 1.0),
                         num_points: int = 100) -> float:
    """
    第一阶段绿电企业最优价格求解

    通过搜索法找到使预期利润最大化的价格

    参数:
        k: 单位生产成本 (元/千瓦时)
        x2: 设备投资补贴比例
        x3: 增值税即征即退比例
        tax_rate: 所得税实际税率
        investment_coef: 投资系数 (万元/万千瓦时)
        previous_investments: 过去投资额列表
        q_t_1: 上期绿电出售数量
        price_range: 价格搜索范围
        num_points: 搜索点数

    返回:
        最优初步价格 (元/千瓦时)
    """
    best_price = price_range[0]
    best_profit = -float('inf')

    # 在价格范围内搜索
    prices = np.linspace(price_range[0], price_range[1], num_points)

    for p in prices:
        # 预测需求量
        q_pred = stage1_demand_prediction(p)

        # 计算投资额
        m = investment_decision(q_pred, q_t_1, previous_investments,
                                investment_coef, base.EQUIPMENT_LIFESPAN)

        # 计算预期利润
        profit = green_energy_profit(p, q_pred, k, m, x2, x3, tax_rate)

        if profit > best_profit:
            best_profit = profit
            best_price = p

    return best_price


def stage3_final_price(c_g: float, p_g0: float, p_s: float, alpha: float,
                       beta: float, epsilon_std: float = base.GREEN_ENERGY_BEHAVIOR['epsilon_std']) -> float:
    """
    第三阶段绿电企业最终定价

    p_g,i^1 = α_i·c_g,i + β_i·p_g^0 + (1 - α_i - β_i)·p_s + ε_i

    参数:
        c_g: 单位边际生产成本 (元/千瓦时)
        p_g0: 第一阶段初步价格 (元/千瓦时)
        p_s: 市场愿意支付的边际价格 (元/千瓦时)
        alpha: 成本锚定权重
        beta: 初始报价延续性权重
        epsilon_std: 随机扰动项标准差

    返回:
        最终绿证报价 (元/千瓦时)
    """
    # 计算确定性部分
    deterministic_part = (alpha * c_g + beta * p_g0 +
                          (1 - alpha - beta) * p_s)

    # 添加随机扰动
    epsilon = np.random.normal(0, epsilon_std)

    # 确保价格为正值
    final_price = max(0.001, deterministic_part + epsilon)

    return final_price


def calculate_market_price(demand_volume: float, intercept: float = base.DEMAND_FUNCTION['intercept'],
                           slope: float = base.DEMAND_FUNCTION['slope']) -> float:
    """
    基于市场需求计算边际价格

    ln(p_s) = (intercept - ln(D_g*)) / slope

    参数:
        demand_volume: 市场需求总量 (万千瓦时)
        intercept: 反需求函数截距
        slope: 反需求函数斜率

    返回:
        市场边际价格 (元/千瓦时)
    """
    if demand_volume <= 0:
        return 0.001

    ln_p = (intercept - math.log(demand_volume)) / slope
    return math.exp(ln_p)


def optimal_supply_quantity(p_g1: float, k: float, m: float,
                            capacity_constraint: float) -> float:
    """
    绿电企业最优供给量求解

    在给定价格下最大化利润的线性规划问题

    参数:
        p_g1: 最终绿证报价 (元/千瓦时)
        k: 单位生产成本 (元/千瓦时)
        m: 固定资产投资 (万元)
        capacity_constraint: 产能约束 (万千瓦时)

    返回:
        最优供给量 (万千瓦时)
    """
    # 将价格和成本转换为万元/万千瓦时
    p_per_10k = p_g1 * 10000
    k_per_10k = k * 10000

    # 单位利润
    unit_profit = p_per_10k - k_per_10k

    if unit_profit <= 0:
        # 如果单位利润为负，不生产
        return 0.0

    # 在产能约束下最大化产量
    return min(capacity_constraint, m / base.INVESTMENT_COEFFICIENT)


# ============================================================================
# 4. 用电企业函数 (CE)
# ============================================================================

def electricity_consumer_profit(r_c: float, p_e: float, t_c: float,
                                p_b: float, e_c: float, c_o: float,
                                d_j: float) -> float:
    """
    用电企业利润函数

    π^c = R_c - (1 - x5)·p_e·T_c - p_b·E_c - C_o

    参数:
        r_c: 售电收入 (万元)
        p_e: 绿电价格 (元/千瓦时)
        t_c: 绿电采购量 (万千瓦时)
        p_b: 传统电力价格 (元/千瓦时)
        e_c: 传统电力采购量 (万千瓦时)
        c_o: 其他生产成本 (万元)
        d_j: 总电力需求 (万千瓦时)

    返回:
        利润 (万元)
    """
    # 将价格转换为万元/万千瓦时
    p_e_per_10k = p_e * 10000
    p_b_per_10k = p_b * 10000

    # 计算总成本
    total_cost = (1 - base.GOV_POLICY_INITIAL['x5']) * p_e_per_10k * t_c + \
                 p_b_per_10k * e_c + c_o

    return r_c - total_cost


def optimal_green_power_purchase(p_e: float, p_b: float, d_j: float,
                                 x4: float, x5: float, c_o: float,
                                 r_c: float) -> Tuple[float, float, float]:
    """
    用电企业最优绿电采购量求解

    在满足电力需求和配额约束下最大化利润

    参数:
        p_e: 绿电价格 (元/千瓦时)
        p_b: 传统电力价格 (元/千瓦时)
        d_j: 总电力需求 (万千瓦时)
        x4: 强制消纳配额比例
        x5: 消费端补贴比例
        c_o: 其他生产成本 (万元)
        r_c: 售电收入 (万元)

    返回:
        (最优绿电采购量, 最优传统电力采购量, 最大利润)
    """
    # 补贴后的绿电实际价格
    effective_p_e = (1 - x5) * p_e

    if effective_p_e <= p_b:
        # 如果补贴后绿电价格不高于传统电力，全部购买绿电
        q_g_optimal = d_j
        e_c_optimal = 0.0
    else:
        # 否则仅购买满足配额要求的绿电量
        q_g_optimal = x4 * d_j
        e_c_optimal = d_j - q_g_optimal

    # 计算利润
    profit = electricity_consumer_profit(r_c, p_e, q_g_optimal,
                                         p_b, e_c_optimal, c_o, d_j)

    return q_g_optimal, e_c_optimal, profit


def calculate_operating_cost(d_j: float, c_o_coef: float = base.OPERATING_COST_COEFFICIENT) -> float:
    """
    计算用电企业其他生产成本

    C_o = c_o * D_j

    参数:
        d_j: 总电力需求 (万千瓦时)
        c_o_coef: 运营成本系数 (万元/万千瓦时)

    返回:
        其他生产成本 (万元)
    """
    return c_o_coef * d_j


def electricity_demand_forecast(d_t_1: float, gdp_t_1: float, gdp_t_2: float,
                                u_range: Tuple[float, float] = base.DEMAND_SHARE_RANGE) -> float:
    """
    用电需求预测

    D_t = (D_{t-1} + E_g·ΔGDP_j) * U

    参数:
        d_t_1: 上期电力需求量 (万千瓦时)
        gdp_t_1: 上期GDP (万亿元)
        gdp_t_2: 上上期GDP (万亿元)
        u_range: 分摊比U的取值范围

    返回:
        当期电力需求量预测 (万千瓦时)
    """
    # 计算GDP变化
    delta_gdp = gdp_t_1 - gdp_t_2

    # 计算电力需求变化
    delta_demand = base.GDP_ELASTICITY * delta_gdp

    # 基础需求预测
    base_demand = d_t_1 + delta_demand

    # 随机分摊比
    u = np.random.uniform(u_range[0], u_range[1])

    return base_demand * u


def gdp_forecast(gdp_t: float, gdp_t_1: float, epsilon_t: float = 0.0) -> float:
    """
    GDP预测模型 (ARIMA模型)

    基于ARIMA模型的预测逻辑

    参数:
        gdp_t: 当期GDP (万亿元)
        gdp_t_1: 上期GDP (万亿元)
        epsilon_t: 模型残差项

    返回:
        下期GDP预测 (万亿元)
    """
    # 避免对0或负数取对数
    if gdp_t <= 0 or gdp_t_1 <= 0:
        return gdp_t

    log_gdp_t = math.log(gdp_t)
    log_gdp_t_1 = math.log(gdp_t_1)

    # 预测计算
    log_gdp_forecast = (log_gdp_t + base.GDP_ARIMA_PARAMS['constant'] +
                        base.GDP_ARIMA_PARAMS['ar_coef'] * (log_gdp_t - log_gdp_t_1) +
                        base.GDP_ARIMA_PARAMS['ma_coef'] * epsilon_t)

    return math.exp(log_gdp_forecast)


# ============================================================================
# 5. 市场出清函数
# ============================================================================

def market_clearing(supplies: List[Tuple[float, float]],
                    total_demand: float) -> Dict[str, Union[float, List[float]]]:
    """
    绿电市场出清机制

    根据总供给量和总需求量进行市场出清

    参数:
        supplies: 绿电企业供给列表 [(价格, 数量), ...]
        total_demand: 总需求量 (万千瓦时)

    返回:
        包含出清结果的字典
    """
    # 按价格从低到高排序
    sorted_supplies = sorted(supplies, key=lambda x: x[0])

    total_supply = sum(q for _, q in sorted_supplies)

    if total_supply <= total_demand:
        # 供不应求情况
        cleared_prices = [p for p, _ in sorted_supplies]
        cleared_quantities = [q for _, q in sorted_supplies]
        clearing_price = sum(p * q for p, q in sorted_supplies) / total_supply if total_supply > 0 else 0
        green_cert_price = clearing_price  # 绿证价格等于出清价格
        surplus = total_demand - total_supply
    else:
        # 供过于求情况
        cumulative_supply = 0
        cleared_prices = []
        cleared_quantities = []

        for price, quantity in sorted_supplies:
            if cumulative_supply + quantity <= total_demand:
                cleared_prices.append(price)
                cleared_quantities.append(quantity)
                cumulative_supply += quantity
            else:
                # 最后一家部分出清
                remaining = total_demand - cumulative_supply
                if remaining > 0:
                    cleared_prices.append(price)
                    cleared_quantities.append(remaining)
                    cumulative_supply += remaining
                break

        clearing_price = sum(
            p * q for p, q in zip(cleared_prices, cleared_quantities)) / total_demand if total_demand > 0 else 0
        green_cert_price = min(p for p, _ in sorted_supplies)  # 绿证价格为最低报价
        surplus = total_supply - total_demand

    return {
        'cleared_prices': cleared_prices,
        'cleared_quantities': cleared_quantities,
        'total_cleared_quantity': sum(cleared_quantities),
        'clearing_price': clearing_price,
        'green_cert_price': green_cert_price,
        'surplus': surplus,
        'shortage': max(0, total_demand - sum(cleared_quantities))
    }


# ============================================================================
# 6. 税收优惠计算函数
# ============================================================================

def calculate_income_tax_benefit(net_profit: float, income_tax_expense: float,
                                 government_subsidy: float, revenue: float,
                                 x1: float) -> Dict[str, float]:
    """
    计算所得税优惠比例 (第2.4.2节)

    参数:
        net_profit: 净利润 (万元)
        income_tax_expense: 所得税费用 (万元)
        government_subsidy: 政府补助 (万元)
        revenue: 营业收入 (万元)
        x1: 所得税优惠强度

    返回:
        包含所得税优惠相关指标的字典
    """
    # 1. 计算会计利润总额
    accounting_profit = net_profit + income_tax_expense

    # 2. 估算研发费用
    rnd_estimate = revenue * base.RND_EXPENSE_RATIO

    # 3. 估算纳税调整
    tax_adjustment = -government_subsidy * 0.5 - rnd_estimate * 0.75

    # 4. 估算应纳税所得额
    taxable_income_estimate = accounting_profit + tax_adjustment
    taxable_income_estimate = max(0, taxable_income_estimate)  # 确保非负

    # 5. 计算理论所得税
    theoretical_income_tax = taxable_income_estimate * base.BASE_INCOME_TAX_RATE

    # 6. 计算所得税优惠值
    income_tax_benefit_value = theoretical_income_tax - income_tax_expense

    # 7. 计算所得税优惠比例
    if theoretical_income_tax > 0:
        income_tax_benefit_ratio = income_tax_benefit_value / theoretical_income_tax
        # 限制在-50%到200%之间
        income_tax_benefit_ratio = max(-0.5, min(2.0, income_tax_benefit_ratio))
    else:
        income_tax_benefit_ratio = 0.0

    # 8. 考虑优惠强度调整
    adjusted_benefit_ratio = income_tax_benefit_ratio * x1

    return {
        'accounting_profit': accounting_profit,
        'taxable_income_estimate': taxable_income_estimate,
        'theoretical_income_tax': theoretical_income_tax,
        'income_tax_benefit_value': income_tax_benefit_value,
        'income_tax_benefit_ratio': income_tax_benefit_ratio,
        'adjusted_benefit_ratio': adjusted_benefit_ratio
    }


def calculate_vat_refund_ratio(tax_payable: float, income_tax_expense: float,
                               tax_refund_received: float) -> float:
    """
    计算增值税即征即退比例 (第2.4.2节)

    参数:
        tax_payable: 应交税费 (万元)
        income_tax_expense: 所得税费用 (万元)
        tax_refund_received: 收到的税费返还 (万元)

    返回:
        增值税即征即退比例
    """
    # 计算非所得税税费
    non_income_tax = tax_payable - income_tax_expense

    if non_income_tax > 0:
        vat_refund_ratio = tax_refund_received / non_income_tax
        # 限制在-20%到120%之间
        vat_refund_ratio = max(-0.2, min(1.2, vat_refund_ratio))
    else:
        vat_refund_ratio = 0.0

    return vat_refund_ratio


# ============================================================================
# 7. 评估指标计算函数
# ============================================================================

def calculate_penetration_rate(green_power_volume: float, total_electricity_consumption: float) -> float:
    """
    计算绿电渗透率

    绿电渗透率 = (年度绿电消费量 / 社会总用电量) × 100%

    参数:
        green_power_volume: 绿电消费量 (万千瓦时)
        total_electricity_consumption: 社会总用电量 (万千瓦时)

    返回:
        绿电渗透率 (%)
    """
    if total_electricity_consumption > 0:
        return (green_power_volume / total_electricity_consumption) * 100
    return 0.0


def calculate_capacity_utilization(total_generation: float, total_capacity: float) -> float:
    """
    计算产能利用率

    产能利用率 = (实际发电量 / 装机容量) × 100%

    参数:
        total_generation: 总发电量 (万千瓦时)
        total_capacity: 总装机容量 (万千瓦)

    返回:
        产能利用率 (%)
    """
    # 将装机容量从兆瓦转换为万千瓦时/年 (假设年利用小时数为8760)
    if total_capacity > 0:
        # 装机容量 (万千瓦) * 8760小时 = 潜在年发电量 (万千瓦时)
        potential_generation = total_capacity * 8760 / 10000  # 转换为万千瓦时
        return (total_generation / potential_generation) * 100
    return 0.0


def calculate_roe(net_profit: float, equity: float) -> float:
    """
    计算净资产收益率 (ROE)

    ROE = (净利润 / 净资产) × 100%

    参数:
        net_profit: 净利润 (万元)
        equity: 净资产 (万元)

    返回:
        净资产收益率 (%)
    """
    if equity > 0:
        return (net_profit / equity) * 100
    return 0.0


def calculate_cost_per_carbon_reduction(policy_cost: float, carbon_reduction: float) -> float:
    """
    计算单位碳减排政策成本

    单位碳减排政策成本 = 累计政策总成本 / 累计碳减排量

    参数:
        policy_cost: 政策成本 (万元)
        carbon_reduction: 碳减排量 (吨CO2)

    返回:
        单位碳减排政策成本 (元/吨CO2)
    """
    if carbon_reduction > 0:
        # 将万元转换为元
        return (policy_cost * 10000) / carbon_reduction
    return 0.0


def calculate_cost_per_green_power(policy_cost: float, green_power_volume: float) -> float:
    """
    计算单位绿电激发成本

    单位绿电激发成本 = 累计政策总成本 / 累计绿电交易量

    参数:
        policy_cost: 政策成本 (万元)
        green_power_volume: 绿电交易量 (万千瓦时)

    返回:
        单位绿电激发成本 (元/千瓦时)
    """
    if green_power_volume > 0:
        # 将万元转换为元，万千瓦时转换为千瓦时
        return (policy_cost * 10000) / (green_power_volume * 10000)
    return 0.0


# ============================================================================
# 8. 验证函数
# ============================================================================

def validate_model_output(simulated_values: Dict[str, float],
                          actual_values: Dict[str, float]) -> Dict[str, float]:
    """
    验证模型输出

    计算模拟值与实际值之间的相对误差

    参数:
        simulated_values: 模拟值字典
        actual_values: 实际值字典

    返回:
        相对误差字典
    """
    errors = {}

    for key in simulated_values:
        if key in actual_values and actual_values[key] != 0:
            relative_error = abs(simulated_values[key] - actual_values[key]) / abs(actual_values[key])
            errors[key] = relative_error
        else:
            errors[key] = None

    return errors


def check_error_within_tolerance(errors: Dict[str, float],
                                 tolerance: Dict[str, float]) -> bool:
    """
    检查误差是否在容忍范围内

    参数:
        errors: 相对误差字典
        tolerance: 容忍误差字典

    返回:
        所有误差是否都在容忍范围内
    """
    for key, error in errors.items():
        if error is not None and key in tolerance:
            if error > tolerance[key]:
                return False
    return True


# ============================================================================
# 9. 辅助函数
# ============================================================================

def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    单位转换函数

    参数:
        value: 原始值
        from_unit: 原始单位
        to_unit: 目标单位

    返回:
        转换后的值
    """
    # 电量单位转换
    if from_unit == '万千瓦时' and to_unit == '兆瓦时':
        return value * base.UNIT_CONVERSION['万千瓦时_to_兆瓦时']
    elif from_unit == '亿千瓦时' and to_unit == '万千瓦时':
        return value * base.UNIT_CONVERSION['亿千瓦时_to_万千瓦时']
    elif from_unit == '兆瓦时' and to_unit == '千瓦时':
        return value * base.UNIT_CONVERSION['兆瓦时_to_千瓦时']
    elif from_unit == '万千瓦时' and to_unit == '千瓦时':
        return value * base.UNIT_CONVERSION['万千瓦时_to_千瓦时']

    # 货币单位转换
    elif from_unit == '亿元' and to_unit == '万元':
        return value * base.CURRENCY_CONVERSION['亿元_to_万元']
    elif from_unit == '万元' and to_unit == '元':
        return value * base.CURRENCY_CONVERSION['万元_to_元']

    # 如果单位相同或无法转换，返回原值
    return value


def calculate_group_statistics(values: List[float], method: str = 'median') -> float:
    """
    计算分组统计量

    参数:
        values: 数值列表
        method: 统计方法 ('median', 'mean', 'min', 'max')

    返回:
        统计量值
    """
    if not values:
        return 0.0

    if method == 'median':
        return np.median(values)
    elif method == 'mean':
        return np.mean(values)
    elif method == 'min':
        return np.min(values)
    elif method == 'max':
        return np.max(values)
    else:
        return np.mean(values)



def json_serializable(obj):
    """
    将对象转换为JSON可序列化的格式

    参数:
        obj: 任意对象

    返回:
        JSON可序列化的对象
    """
    if isinstance(obj, (demand, np.datetime64)):
        return str(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

if __name__ == "__main__":
    """测试函数"""
    print("SG-ABM模型函数测试:")

    # 测试碳减排计算
    carbon_reduction = calculate_carbon_reduction(1000)  # 1000万千瓦时
    print(f"碳减排量: {carbon_reduction:.2f} 吨CO2")

    # 测试政府社会福利
    welfare = government_welfare(carbon_reduction, 1000, 500)
    print(f"社会福利: {welfare:.2f}")

    # 测试绿电企业利润计算
    profit = green_energy_profit(0.5, 1000, 0.2, 1000, 0.1, 0.5, 0.15)
    print(f"绿电企业利润: {profit:.2f} 万元")

    # 测试市场需求预测
    demand = stage1_demand_prediction(0.4)
    print(f"预期需求量: {demand:.2f} 万千瓦时")

    print("所有函数测试完成!")

