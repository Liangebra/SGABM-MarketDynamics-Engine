# SGABM-MarketDynamics-Engine

**High-Performance Agent-Based Modeling (ABM) Framework for Green Power Market Evolution**
**高性能绿电市场演化 ABM 仿真引擎**

---

## 1. 概览 / Overview

**SGABM-MarketDynamics-Engine** is a sophisticated simulation platform designed to model the long-term  co-evolution of policy interventions, power generation strategies, and consumer behaviors in green energy markets. 

**SGABM-MarketDynamics-Engine** 是一个先进的仿真平台，旨在模拟绿电市场中政策干预、发电策略与消费者行为在长期尺度下的协同演化过程。

It transitions from a data-dependent research script to a **Zero-Dependency, Parameter-Driven** engine, allowing researchers to explore market equilibrium directly through parameter tuning.

该引擎已实现从“数据依赖型脚本”向“**零依赖、参数驱动型**”引擎的转变，研究人员可以直接通过参数调优探索市场均衡。

---

## 2. 核心原理 / Core Principles

The engine integrates three major mathematical frameworks to simulate market complexity:
本引擎集成了三大数学框架来模拟市场复杂性：

1.  **Stackelberg Game (Gov vs. Firms)**:
    - **English**: Simulates the leader-follower interaction where the government sets tax/subsidy policies to maximize social welfare, and firms respond to maximize profits.
    - **中文 (政府与企业博弈)**：模拟领导者-跟随者交互，政府通过设定税收/补贴政策最大化社会福利，企业据此响应以最大化利润。

2.  **EWA Learning (Adaptive Strategy)**:
    - **English**: Implements Experience-Weighted Attraction learning. Agents (Firms/Gov) learn from historical payoffs and adjust their strategy attractions dynamically.
    - **中文 (EWA 学习算法)**：实现经验加权吸引力学习逻辑。主体（企业/政府）根据历史收益学习，并动态调整其策略吸引力。

3.  **Evolutionary Game (Consumer Choice)**:
    - **English**: Models the dynamic shift in clean energy preference among consumers based on price signals, policy incentives, and environmental awareness.
    - **中文 (演化博弈)**：基于价格信号、政策激励和环保意识，模拟用电企业对清洁能源偏好的动态转移。

---

## 3. 快速开始 / Quick Start

### 3.1 环境准备 / Prerequisites
```bash
pip install -r requirements.txt
```

### 3.2 运行仿真 / Run Simulation
You can run the batch simulation for all default scenarios (S0-S6).
您可以直接运行覆盖所有默认情景 (S0-S6) 的批量仿真。

```bash
# Default (50 years) / 默认运行（50年）
python run_simulation.py

# Custom duration (e.g., 20 years) / 自定义时长（如 20 年）
python run_simulation.py --years 20
```

### 3.3 数据可视化 / Visualization
Generate "Origami-style" professional charts:
生成专业“折纸风格”图表：
```bash
python visualize.py
```

---

## 4. 项目结构 / Project Structure

```text
SGABM-MarketDynamics-Engine/
├── core/                   # 仿真核心逻辑包 / Core Simulation Logic
│   ├── __init__.py         # 包初始化 / Package Initialization
│   ├── base.py             # 基础参数与情景定义 / Base Parameters & Scenarios
│   ├── fun.py              # 核心公式与收益计算 / Mathematical Formulas
│   └── math_go.py          # 博弈求解与学习算法 / Game Solvers & Learning
├── utils/                  # 工具函数包 / Utility Modules
│   ├── __init__.py         # 包初始化 / Package Initialization
│   ├── input.py            # 数据加载与初始化 / Data Loading & Initialization
│   └── view.py             # 结果处理与底层绘图 / Data Processing & Plotting
├── run_simulation.py       # 仿真运行主入口 / Simulation Entry Point
├── visualize.py            # 可视化增强脚本 / Visualization Script
├── requirements.txt        # 依赖清单 / Dependencies
└── README.md               # 项目说明文档 / Documentation
```

---

## 5. 参数修改指南 / Parameter Modification

The project is **Zero-Dependency**; all initial states are defined in `core/base.py`.
本项目已实现“零依赖”，所有初始状态均在 `core/base.py` 中定义。

### 修改位置 / Location: `core/base.py`

| Parameter Group / 参数组 | Key Constants / 关键常量 | Description / 说明 |
| :--- | :--- | :--- |
| **Market Scale / 市场规模** | `MARKET_SCALING_FACTOR` | Controls the demand-supply base volume. (控制供需基础规模。) |
| **Gov Preference / 政府偏好** | `GOV_ALPHA`, `GOV_BETA` | Weights for Environment vs Economy. (环境与经济的决策权重。) |
| **Firm Config / 企业配置** | `GREEN_ENERGY_GROUPS` | Initial assets, costs, and counts of generation firms. (发电企业的资产、成本及数量。) |
| **Consumer Config / 用户配置** | `CONSUMER_GROUPS` | Industry-specific energy consumption and green preferences. (分行业的用电量与绿电偏好。) |
| **Scenarios / 实验情景** | `SCENARIO_TARGETS` | Policy tool values for S1-S6 (Tax, Subsidy, Quota). (各情景下的政策工具值。) |

---

## 5. 输出解析 / Output Interpretation

Simulation results are stored in the `batch_results/[TIMESTAMP]/` directory:
仿真结果存储在 `batch_results/[时间戳]/` 目录下：

*   **`comprehensive_results.xlsx`**: Detailed time-series data for all scenarios. (所有情景的详细时间序列数据。)
*   **`summary_report.txt`**: Ranking and recommendation of policy scenarios. (政策情景的排名与推荐建议。)
*   **`visualizations/`**: Includes Line, Radar, and Origami charts. (包含折线图、雷达图和专项分析图。)

---

## 6. 应用范围 / Use Cases

*   **Policy Sensitivity Test**: Analyze how carbon quotas affect market clearing prices. (政策灵敏度测试：分析碳配额如何影响市场出清电价。)
*   **Strategic Planning**: Simulate long-term ROI for green energy investment groups. (战略规划：模拟绿电投资集团的长期投资回报率。)
*   **Decarbonization Pathways**: Evaluate the time required to reach net-zero under different subsidies. (脱碳路径：评估不同补贴强度下达到净零排放所需的时间。)

---
© 2026 SGABM Development Team. Lead Developer: Liang.
