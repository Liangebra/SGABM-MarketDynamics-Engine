#!/usr/bin/env python3
"""
run_simulation.py - SG-ABM批量仿真运行主脚本

该脚本用于配置并启动SG-ABM模型的多情景批量仿真运行。
它通过调用 core 模块中的逻辑，模拟不同政策环境下市场的动态演化过程，
并生成详细的仿真结果报告、数据可视化图表。

作者：Liang
日期：2026年
版本：1.0
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from datetime import datetime

# 导入必要的模块
from core import base, math_go
from core.math_go import MultiAgentCoordinator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def run_single_scenario(scenario_id: str, years: int = 50,
                        num_firms: tuple = (10, 5), num_consumers: int = 10) -> Dict[str, Any]:
    """
    运行单个情景仿真

    参数:
        scenario_id: 情景ID (S0-S6)
        years: 仿真年数
        num_firms: (低规模组企业数, 中规模组企业数)
        num_consumers: 用电企业数

    返回:
        仿真结果字典
    """
    print(f"开始运行情景 {scenario_id} ({base.get_scenario_name(scenario_id)})...")

    # 设置随机种子以确保可重复性
    random_seed = hash(scenario_id) % 10000
    np.random.seed(random_seed)

    try:
        # 创建协调器
        coordinator = MultiAgentCoordinator(
            num_green_energy_firms=num_firms,
            num_consumers=num_consumers
        )

        # 运行仿真
        results = coordinator.run_simulation(years=years)

        # 提取关键数据
        scenario_data = {
            'scenario_id': scenario_id,
            'scenario_name': base.get_scenario_name(scenario_id),
            'results': results['results'],  # 每年详细结果
            'evaluation': results['evaluation'],  # 评估指标
            'policy_history': results['policy_history'],  # 政策演化历史
            'random_seed': random_seed,
            'status': 'success',
            'run_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print(f"情景 {scenario_id} 运行完成")
        return scenario_data

    except Exception as e:
        print(f"情景 {scenario_id} 运行失败: {str(e)}")
        return {
            'scenario_id': scenario_id,
            'scenario_name': base.get_scenario_name(scenario_id),
            'error': str(e),
            'status': 'error',
            'run_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def extract_time_series_data(scenario_results: Dict[str, Any]) -> pd.DataFrame:
    """
    从仿真结果中提取时间序列数据

    参数:
        scenario_results: 单个情景的仿真结果

    返回:
        时间序列DataFrame
    """
    if 'results' not in scenario_results or not scenario_results['results']:
        return pd.DataFrame()

    data_list = []

    for result in scenario_results['results']:
        # 基础指标
        row = {
            'scenario_id': scenario_results['scenario_id'],
            'scenario_name': scenario_results['scenario_name'],
            'year': result['year'],
            'avg_p_g0': result.get('avg_p_g0', 0),
            'green_power_volume': result.get('green_power_volume', 0),
            'carbon_reduction': result.get('carbon_reduction', 0),
            'welfare': result.get('welfare', 0),
            'total_green_demand': result.get('total_green_demand', 0),
            'total_policy_cost': result.get('total_policy_cost', 0),
            'gdp': result.get('gdp', 0)
        }

        # 市场出清结果
        if 'market_result' in result:
            mr = result['market_result']
            row['clearing_price'] = mr.get('clearing_price', 0)
            row['green_cert_price'] = mr.get('green_cert_price', 0)
            row['total_cleared_quantity'] = mr.get('total_cleared_quantity', 0)
            row['surplus'] = mr.get('surplus', 0)
            row['shortage'] = mr.get('shortage', 0)

        # 政策指标
        if 'policies' in result:
            policies = result['policies']
            for key, value in policies.items():
                row[f'policy_{key}'] = value

        # 政策成本
        if 'policy_costs' in result:
            policy_costs = result['policy_costs']
            for key, value in policy_costs.items():
                row[f'policy_cost_{key}'] = value

        data_list.append(row)

    return pd.DataFrame(data_list)


def extract_evaluation_metrics(scenario_results: Dict[str, Any]) -> pd.DataFrame:
    """
    提取评估指标

    参数:
        scenario_results: 单个情景的仿真结果

    返回:
        评估指标DataFrame
    """
    if 'evaluation' not in scenario_results:
        return pd.DataFrame()

    evaluation = scenario_results['evaluation']
    metrics = {
        'scenario_id': scenario_results['scenario_id'],
        'scenario_name': scenario_results['scenario_name'],
        'total_carbon_reduction': evaluation.get('total_carbon_reduction', 0),
        'total_green_power_volume': evaluation.get('total_green_power_volume', 0),
        'total_policy_cost': evaluation.get('total_policy_cost', 0),
        'avg_clearing_price': evaluation.get('avg_clearing_price', 0),
        'penetration_rate': evaluation.get('penetration_rate', 0),
        'capacity_utilization': evaluation.get('capacity_utilization', 0),
        'cost_per_carbon': evaluation.get('cost_per_carbon', 0),
        'cost_per_green_power': evaluation.get('cost_per_green_power', 0),
        'run_status': scenario_results.get('status', 'unknown')
    }

    return pd.DataFrame([metrics])


def extract_policy_history(scenario_results: Dict[str, Any]) -> pd.DataFrame:
    """
    提取政策演化历史

    参数:
        scenario_results: 单个情景的仿真结果

    返回:
        政策演化DataFrame
    """
    if 'policy_history' not in scenario_results:
        return pd.DataFrame()

    policy_history = scenario_results['policy_history']
    data_list = []

    for policy_name, values in policy_history.items():
        for year_idx, value in enumerate(values):
            data_list.append({
                'scenario_id': scenario_results['scenario_id'],
                'scenario_name': scenario_results['scenario_name'],
                'policy_name': policy_name,
                'year': year_idx + 1,  # 从第1年开始
                'value': value
            })

    return pd.DataFrame(data_list)


def create_excel_report(all_results: Dict[str, Dict[str, Any]], output_dir: Path) -> str:
    """
    创建Excel报告

    参数:
        all_results: 所有情景的结果字典
        output_dir: 输出目录

    返回:
        Excel文件路径
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = output_dir / f"sgabm_all_scenarios_{timestamp}.xlsx"

    # 创建Excel写入器
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # 1. 汇总表：所有情景的评估指标
        summary_dfs = []
        for scenario_id, results in all_results.items():
            df = extract_evaluation_metrics(results)
            if not df.empty:
                summary_dfs.append(df)

        if summary_dfs:
            summary_df = pd.concat(summary_dfs, ignore_index=True)
            summary_df.to_excel(writer, sheet_name='情景汇总', index=False)
            print(f" [OK] 创建'情景汇总'工作表: {len(summary_df)} 行")

        # 2. 时间序列数据表
        time_series_dfs = []
        for scenario_id, results in all_results.items():
            if results.get('status') == 'success':
                df = extract_time_series_data(results)
                if not df.empty:
                    time_series_dfs.append(df)

        if time_series_dfs:
            time_series_df = pd.concat(time_series_dfs, ignore_index=True)
            time_series_df.to_excel(writer, sheet_name='时间序列数据', index=False)
            print(f" [OK] 创建'时间序列数据'工作表: {len(time_series_df)} 行")

        # 3. 政策演化表
        policy_history_dfs = []
        for scenario_id, results in all_results.items():
            if results.get('status') == 'success':
                df = extract_policy_history(results)
                if not df.empty:
                    policy_history_dfs.append(df)

        if policy_history_dfs:
            policy_history_df = pd.concat(policy_history_dfs, ignore_index=True)
            policy_history_df.to_excel(writer, sheet_name='政策演化', index=False)
            print(f" [OK] 创建'政策演化'工作表: {len(policy_history_df)} 行")

        # 4. 情景描述表
        scenario_info = []
        for scenario_id in base.SCENARIO_TARGETS.keys():
            scenario_info.append({
                '情景ID': scenario_id,
                '情景名称': base.get_scenario_name(scenario_id),
                '所得税优惠(x1)': base.SCENARIO_TARGETS[scenario_id].get('x1', 0),
                '设备投资补贴(x2)': base.SCENARIO_TARGETS[scenario_id].get('x2', 0),
                '增值税即征即退(x3)': base.SCENARIO_TARGETS[scenario_id].get('x3', 0),
                '强制消纳配额(x4)': base.SCENARIO_TARGETS[scenario_id].get('x4', 0),
                '消费端补贴(x5)': base.SCENARIO_TARGETS[scenario_id].get('x5', 0)
            })

        scenario_df = pd.DataFrame(scenario_info)
        scenario_df.to_excel(writer, sheet_name='情景配置', index=False)
        print(f" [OK] 创建'情景配置'工作表: {len(scenario_df)} 行")

        # 5. 运行统计表
        run_stats = []
        for scenario_id, results in all_results.items():
            run_stats.append({
                '情景ID': scenario_id,
                '情景名称': results.get('scenario_name', '未知'),
                '运行状态': results.get('status', 'unknown'),
                '随机种子': results.get('random_seed', 0),
                '运行时间': results.get('run_time', '未知'),
                '错误信息': results.get('error', '无')
            })

        stats_df = pd.DataFrame(run_stats)
        stats_df.to_excel(writer, sheet_name='运行统计', index=False)
        print(f" [OK] 创建'运行统计'工作表: {len(stats_df)} 行")

    print(f"\nExcel 汇总报告已保存至: {excel_file}")
    return str(excel_file)


def create_summary_report(all_results: Dict[str, Dict[str, Any]], output_dir: Path) -> str:
    """
    创建摘要报告

    参数:
        all_results: 所有情景的结果字典
        output_dir: 输出目录

    返回:
        摘要报告文件路径
    """
    # 提取所有成功的评估指标
    successful_scenarios = []
    for scenario_id, results in all_results.items():
        if results.get('status') == 'success' and 'evaluation' in results:
            metrics = results['evaluation']
            metrics['scenario_id'] = scenario_id
            metrics['scenario_name'] = results.get('scenario_name', '')
            successful_scenarios.append(metrics)

    if not successful_scenarios:
        print(" [ERROR] 没有成功的仿真结果")
        return ""

    # 创建DataFrame
    summary_df = pd.DataFrame(successful_scenarios)

    # 计算排名
    # 碳减排排名（越高越好）
    summary_df['carbon_reduction_rank'] = summary_df['total_carbon_reduction'].rank(ascending=False, method='min')

    # 政策成本排名（越低越好）
    summary_df['policy_cost_rank'] = summary_df['total_policy_cost'].rank(ascending=True, method='min')

    # 单位碳减排成本排名（越低越好）
    summary_df['cost_per_carbon_rank'] = summary_df['cost_per_carbon'].rank(ascending=True, method='min')

    # 渗透率排名（越高越好）
    summary_df['penetration_rank'] = summary_df['penetration_rate'].rank(ascending=False, method='min')

    # 计算综合得分（排名平均值，越低越好）
    summary_df['综合得分'] = summary_df[['carbon_reduction_rank', 'policy_cost_rank',
                                         'cost_per_carbon_rank', 'penetration_rank']].mean(axis=1)
    summary_df['综合排名'] = summary_df['综合得分'].rank(ascending=True, method='min')

    # 按综合排名排序
    summary_df = summary_df.sort_values('综合排名')

    # 保存摘要报告
    report_file = output_dir / "summary_report.json"

    # 构建报告数据
    report_data = {
        'generated_at': datetime.now().isoformat(),
        'total_scenarios': len(all_results),
        'successful_scenarios': len(successful_scenarios),
        'failed_scenarios': len(all_results) - len(successful_scenarios),
        'top_scenario': {
            'id': summary_df.iloc[0]['scenario_id'],
            'name': summary_df.iloc[0]['scenario_name'],
            'total_carbon_reduction': summary_df.iloc[0]['total_carbon_reduction'],
            'total_policy_cost': summary_df.iloc[0]['total_policy_cost'],
            'penetration_rate': summary_df.iloc[0]['penetration_rate'],
            'composite_score': summary_df.iloc[0]['综合得分']
        },
        'scenario_ranking': summary_df[['scenario_id', 'scenario_name', '综合排名',
                                        'total_carbon_reduction', 'total_policy_cost',
                                        'penetration_rate']].to_dict('records'),
        'recommendations': []
    }

    # 添加建议
    best_scenario = report_data['top_scenario']
    report_data['recommendations'].append(
        f"推荐情景：{best_scenario['name']} (ID: {best_scenario['id']})"
    )
    report_data['recommendations'].append(
        f"该情景在碳减排、政策成本和渗透率方面表现最佳"
    )
    report_data['recommendations'].append(
        f"累计碳减排：{best_scenario['total_carbon_reduction']:,.0f} 吨CO₂"
    )
    report_data['recommendations'].append(
        f"绿电渗透率：{best_scenario['penetration_rate']:.1%}"
    )

    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print(f" [REPORT] 摘要报告已保存至: {report_file}")
    return str(report_file)


def create_visualizations(all_results: Dict[str, Dict[str, Any]], output_dir: Path):
    """
    创建基本可视化图表

    参数:
        all_results: 所有情景的结果字典
        output_dir: 输出目录
    """
    # 创建可视化目录
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 碳减排对比图
        carbon_data = []
        for scenario_id, results in all_results.items():
            if results.get('status') == 'success' and 'evaluation' in results:
                carbon_data.append({
                    'scenario': results.get('scenario_name', scenario_id),
                    'carbon_reduction': results['evaluation'].get('total_carbon_reduction', 0)
                })

        if carbon_data:
            carbon_df = pd.DataFrame(carbon_data)
            plt.figure(figsize=(10, 6))
            bars = plt.bar(carbon_df['scenario'], carbon_df['carbon_reduction'])
            plt.title('各情景累计碳减排量对比')
            plt.xlabel('情景')
            plt.ylabel('碳减排量 (吨CO₂)')
            plt.xticks(rotation=45, ha='right')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:,.0f}',
                         ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(viz_dir / "carbon_reduction_comparison.png", dpi=300)
            plt.close()
            print(" [INFO] 已生成碳减排对比图")

        # 2. 政策成本对比图
        cost_data = []
        for scenario_id, results in all_results.items():
            if results.get('status') == 'success' and 'evaluation' in results:
                cost_data.append({
                    'scenario': results.get('scenario_name', scenario_id),
                    'policy_cost': results['evaluation'].get('total_policy_cost', 0)
                })

        if cost_data:
            cost_df = pd.DataFrame(cost_data)
            plt.figure(figsize=(10, 6))
            bars = plt.bar(cost_df['scenario'], cost_df['policy_cost'])
            plt.title('各情景累计政策成本对比')
            plt.xlabel('情景')
            plt.ylabel('政策成本 (万元)')
            plt.xticks(rotation=45, ha='right')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:,.0f}',
                         ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(viz_dir / "policy_cost_comparison.png", dpi=300)
            plt.close()
            print(" [INFO] 已生成政策成本对比图")

        # 3. 渗透率对比图
        penetration_data = []
        for scenario_id, results in all_results.items():
            if results.get('status') == 'success' and 'evaluation' in results:
                penetration_data.append({
                    'scenario': results.get('scenario_name', scenario_id),
                    'penetration_rate': results['evaluation'].get('penetration_rate', 0) * 100
                })

        if penetration_data:
            penetration_df = pd.DataFrame(penetration_data)
            plt.figure(figsize=(10, 6))
            bars = plt.bar(penetration_df['scenario'], penetration_df['penetration_rate'])
            plt.title('各情景绿电渗透率对比')
            plt.xlabel('情景')
            plt.ylabel('渗透率 (%)')
            plt.xticks(rotation=45, ha='right')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.1f}%',
                         ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(viz_dir / "penetration_rate_comparison.png", dpi=300)
            plt.close()
            print(" [INFO] 已生成渗透率对比图")

    except Exception as e:
        print(f" [WARNING] 创建可视化图表时出错: {str(e)}")


def main():
    """主函数"""
    print("=" * 60)
    print("SG-ABM 绿电市场仿真模型 - 多情景批量并行仿真系统")
    print("=" * 60)

    # 配置参数
    YEARS = 50  # 仿真年数
    NUM_FIRMS = (10, 5)  # (低规模组企业数, 中规模组企业数)
    NUM_CONSUMERS = 10  # 用电企业数

    # 获取所有情景
    all_scenarios = list(base.SCENARIO_TARGETS.keys())
    print(f"情景数量: {len(all_scenarios)}")
    print(f"仿真年数: {YEARS} 年")
    print(f"绿电企业: {NUM_FIRMS[0]}+{NUM_FIRMS[1]} 家")
    print(f"用电企业: {NUM_CONSUMERS} 家")
    print("\n开始批量仿真...\n")

    # 运行所有情景
    all_results = {}
    start_time = datetime.now()

    for scenario_id in all_scenarios:
        result = run_single_scenario(
            scenario_id=scenario_id,
            years=YEARS,
            num_firms=NUM_FIRMS,
            num_consumers=NUM_CONSUMERS
        )
        all_results[scenario_id] = result

    # 计算总时间
    elapsed_time = (datetime.now() - start_time).total_seconds()

    # 统计结果
    successful = sum(1 for r in all_results.values() if r.get('status') == 'success')
    failed = len(all_results) - successful

    print(f"\n{'=' * 60}")
    print("批量仿真完成!")
    print(f"{'=' * 60}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"总情景数: {len(all_results)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")

    if failed > 0:
        print("\n失败的情景:")
        for scenario_id, result in all_results.items():
            if result.get('status') != 'success':
                print(f"  {scenario_id}: {result.get('error', '未知错误')}")

    # 创建输出目录
    output_dir = Path("./batch_results") / datetime.now().strftime("%Y%m%d_%H%M%S")

    # 生成Excel报告
    if successful > 0:
        print("\n" + "=" * 60)
        print("生成输出文件...")
        print("=" * 60)

        try:
            # 1. 创建Excel报告
            excel_file = create_excel_report(all_results, output_dir)

            # 2. 创建摘要报告
            summary_file = create_summary_report(all_results, output_dir)

            # 3. 创建可视化图表
            create_visualizations(all_results, output_dir)

            print(f"\n{'=' * 60}")
            print("所有仿真输出文件已成功生成。")
            print(f"{'=' * 60}")
            print(f" [DIR] 输出目录: {output_dir}")
            print(f" [XLS] Excel报告: {excel_file}")
            print(f" [TXT] 摘要报告: {summary_file}")
            print(f" [IMG] 可视化图表: {output_dir}/visualizations/")
            print(f"\n项目建议:")
            print("1. 打开Excel文件查看各情景详细数据")
            print("2. 查看摘要报告获取情景排名和推荐")
            print("3. 查看可视化图表进行直观对比")

        except Exception as e:
            print(f" [ERROR] 生成输出文件时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print("\n [ERROR] 没有成功的仿真结果，无法生成报告")

    return all_results



if __name__ == "__main__":
    try:
        results = main()
        print("\n[SUCCESS] 批量仿真实验执行完毕。")
        print("数据分析结果已存入指定目录，请查阅 Excel 汇总报告。")
    except KeyboardInterrupt:
        print("\n\n[WARNING] 用户中断执行")
    except Exception as e:
        print(f"\n[FAILURE] 程序运行异常: {str(e)}")
        import traceback

        traceback.print_exc()