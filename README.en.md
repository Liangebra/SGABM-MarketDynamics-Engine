<div align="center">

# SGABM-MarketDynamics-Engine

**High-Performance Green Electricity Market Evolution and Policy Synergy Simulation Engine**

---
</div>

<div align="center">


  [中文](README..md) | English | [日本語](README.ja.md)


</div>

## 1. Overview

The SGABM-MarketDynamics-Engine is an advanced Agent-Based Modeling (ABM) platform designed to simulate the synergistic co-evolution of policy interventions (supply-side fiscal incentives and demand-side green electricity trading policies), power generation enterprise strategies, and electricity consumption enterprise behaviors in the green electricity market over the long term, under the "Dual Carbon" goals.

The engine has transitioned from a data-dependent script to a **dependency-free, parameter-driven** architecture. Researchers can now directly explore policy synergy effects, multi-agent interactions, and market equilibrium through parameter tuning.

---

## 2. Core Principles

The simulation logic of this engine is built upon two mutually supporting mathematical frameworks. Through the dual-layer dynamic interaction of "policy learning-market game", it jointly simulates the complex evolution of the green electricity market and the policy synergy mechanisms.

### 1. **Three-Stage Stackelberg Sequential Game**:

   This engine employs a three-stage Stackelberg sequential game to construct the micro-core of market operation. In this framework, green electricity enterprises act as the "leaders" with first-mover advantage, while electricity-consuming enterprises act as "followers" who react to the leaders' signals. The government acts as an external regulator that sets the rules (policies) and learns from them. This accurately depicts the decision-making sequence and power structure in reality: policies are set first, followed by generation enterprise investments, and then consumption enterprise decisions.

#### Stage 1 (Initial Decision by Green Electricity Enterprises):
   In this stage, as market leaders, green electricity enterprises, based on their awareness of the current government fiscal and tax incentives (e.g., income tax reductions, investment subsidies) and forward-looking expectations of market demand, aim to maximize long-term profits. They simultaneously formulate preliminary green electricity market prices and corresponding capacity investment decisions. This decision-making is a preliminary judgment under uncertainty, aiming to seize market opportunities and lock in capacity layout.

#### Stage 2 (Response by Electricity-Consuming Enterprises):
   Observing the preliminary price signals from green electricity enterprises, the follower electricity-consuming enterprises begin their decision-making. Under the dual constraints of an exogenously given total electricity demand and a mandatory government green electricity consumption quota (RPS), and aiming to minimize their total electricity procurement cost, they calculate the optimal procurement mix of green and conventional electricity. This process is essentially a constrained linear optimization problem, the solution of which forms the actual market demand curve for green electricity.

#### Stage 3 (Final Green Electricity Pricing and Market Clearing):
   Green electricity enterprises, observing the aggregated actual market demand from Stage 2, initiate a price adjustment mechanism. Based on a Bayesian updating concept, considering their own production costs, initial quotes, and the market's feedback on willingness to pay, they revise their preliminary quotes to form the final listed price. Subsequently, the trading center aggregates the final supply curves from all green electricity enterprises and the demand curves from consuming enterprises. Following the principle of "price priority, supply-demand matching", a unified clearing is conducted to determine the final green electricity trading volume, clearing price, and settlement method, completing the closed loop of a full market cycle.

### 2. **EWA (Experience-Weighted Attraction) Learning Algorithm**:

   To break through the strong assumption of "complete rationality" of the government in traditional game theory, this engine introduces the Experience-Weighted Attraction (EWA) learning algorithm from behavioral game theory. This is used to simulate the bounded rationality, adaptive learning, and path-dependent characteristics of policy evolution seen in real-world governments.
   Within this algorithm framework, each possible intensity value for a government policy tool (e.g., income tax incentive strength, consumption quota proportion) is considered a "strategy". Each strategy has a dynamically updated "attraction" value. At the end of each simulation period, the government calculates the social welfare (covering carbon emission reduction benefits, green electricity trading volume growth, and policy fiscal costs) brought by the current policy combination. The core of the EWA algorithm is that it not only updates the attraction of the implemented strategy based on the actual social welfare generated, but also, through an "imagination" parameter, performs weighted updates on the virtual benefits that unchosen strategies might have brought, simulating the policymakers' "counterfactual thinking".
   After the attraction values are updated following specific rules (involving depreciation rate, learning intensity, and other parameters), the probability of the government choosing a certain policy intensity in the next period is determined by an exponential function of its attraction, forming a stochastic response mode where "dominant strategies are chosen with higher probability, but exploration remains possible." This process makes the government's policy adjustment appear as a gradual, historically experienced feedback-based, and somewhat randomly exploratory adaptive learning process, thereby endogenously evolving the policy adjustment path and differentiated policy combinations under different scenarios.

---

## 3. Quick Start

### 3.1 Environment Preparation
```bash
pip install -r requirements.txt
```

### 3.2 Running the Simulation
You can directly run a batch simulation covering all default scenarios (S0-S6).

```bash
# Default run (50 years)
python run_simulation.py

# Custom duration (e.g., 20 years)
python run_simulation.py --years 20
```

### 3.3 Data Visualization
Generate multi-dimensional analysis charts:

```bash
python visualize.py
```

## 4. Project Structure

```Plaintext
SGABM-MarketDynamics-Engine/
├── core/                   # Core simulation logic package
│   ├── __init__.py         # Package initialization
│   ├── base.py             # Base parameters and scenario definitions
│   ├── fun.py              # Core formulas and profit calculation
│   └── math_go.py          # Game solving and learning algorithms
├── utils/                  # Utility functions package
│   ├── __init__.py         # Package initialization
│   ├── input.py            # Data loading and initialization
│   └── view.py             # Result processing and underlying plotting
├── run_simulation.py       # Main simulation entry point
├── visualize.py            # Enhanced visualization script
├── requirements.txt        # Dependencies list
└── README.md               # Project documentation
```

## 5. Parameter Modification Guide
### Core Notes
- Project Feature: Achieved "**zero dependency**", all initial states are defined in `core/base.py`.
- Modification Location: The only configuration file that needs adjustment is `core/base.py`.

### Parameter Group Details
Parameter groups categorized by function and key descriptions are as follows:

1. **Government Preference Parameters**
   - Key Constants: **GOV_ALPHA**, **GOV_BETA**, **GOV_GAMMA**
   - Description: Decision weights for **environmental, economic, and policy costs** in the social welfare function (fitted values: **4.44, 18.98, 5.2**).

2. **Enterprise Configuration Parameters**
   - Key Constants: **GREEN_ENERGY_GROUPS**
   - Description: Initial asset, cost, and capacity settings for the **low-scale group** (SMEs) and **medium-scale group** (leading enterprises) based on clustering analysis.

3. **User Configuration Parameters**
   - Key Constants: **CONSUMER_GROUPS**
   - Description: **Regional electricity-consuming enterprises** defined based on natural monopoly attributes, with exogenously set electricity demand combining **GDP elasticity coefficients** and **ARIMA models**.

4. **Experimental Scenario Parameters**
   - Key Constants: **SCENARIO_TARGETS**
   - Description: Covers **seven major scenarios S0-S6** (baseline, strong supply-side incentives, strong demand-side pull, symmetric synergy, regionally differentiated synergy, etc.).

5. **Learning Algorithm Parameters**
   - Key Constants: **EWA_PARAMETERS**
   - Description: Core parameters for government adaptive learning (including **depreciation rate, imagination parameter**, etc.).


## 6. Output Interpretation
Simulation results are uniformly stored in the `batch_results/[timestamp]/` directory, with core outputs divided into three categories:

### 1. Comprehensive Results Table (**comprehensive_results.xlsx**)
Contains detailed time-series data for multi-dimensional evaluation metrics, including **carbon emission reduction, green electricity penetration rate, cumulative green electricity trading volume, clearing price, capacity utilization rate, cumulative policy costs**, and more.

### 2. Summary Report (**summary_report.txt**)
- **Comprehensive performance evaluation** and **ranking** for each policy scenario;
- Recommended suggestions for the **optimal policy synergy path**.

### 3. Visualization Folder (**visualizations/**)
Contains four types of key charts:
- **Long-term evolution trend line charts** for key indicators;
- **Performance comparison radar charts** for multiple policy scenarios;
- **Network chord diagrams** for agent interaction;
- **Dynamic policy tool invocation heatmaps**.

## 7. Application Scope
The project focuses on four core scenarios, supporting green electricity market policy design and evolution analysis:

### 1. Policy Synergy Effect Evaluation
The core application of this simulation engine is the systematic evaluation of the synergy effects of different policy combinations. By pre-setting seven scenarios from S0 (baseline) to S6 (regionally differentiated), the engine can quantitatively compare the long-term effects of policy combinations such as "strong supply-side incentives", "strong demand-side pull", and "symmetric synergy". The analysis focuses on how these combinations affect the equilibrium state of the green electricity market, including the formation of clearing prices and trading volumes; how they impact economic operation, reflected in changes to green electricity enterprise profits, capacity utilization, and electricity-consuming enterprise costs; and ultimately how they achieve carbon emission reduction in the environmental dimension. The simulation results can intuitively reveal why the "symmetric synergy" scenario (S3) can achieve the "double dividend" of economic efficiency improvement and environmental emission reduction better than single-sided policies, providing policymakers with quantitative evidence beyond qualitative judgments.

### 2. Market Entity Strategy Deduction
The engine meticulously portrays the heterogeneity of market entities and their strategic interactions. On the supply side, green electricity enterprises are divided into a "medium-scale group" (leading enterprises) and a "low-scale group" (SMEs), with different initial assets, cost structures, and risk preferences, leading to differentiated investment and pricing decisions in response to the same policy signals. On the demand side, electricity-consuming enterprises are constrained by both regional GDP growth and mandatory consumption quotas (RPS). Through the three-stage Stackelberg game framework, this tool can simulate how these heterogeneous agents, under policy pressure, quota targets, and electricity cost fluctuations, evolve from short-term cost-minimization responses to long-term strategic investment adjustments, thereby deducing how micro-agent behaviors aggregate and shape macro-market dynamics.

### 3. Long-Term Evolution Path Exploration
The model is specifically designed to observe the long-term dynamics of complex systems. Over a scale of 50 simulation periods (simulated years), the engine can track the nonlinear trajectories of key indicators. For example, it can observe how the green electricity clearing price gradually converges from an initial divergent state to a long-term equilibrium point determined by generation costs and substitution value; how the green electricity penetration rate continuously exhibits aperiodic fluctuations due to investment lags, policy learning, and demand uncertainty; and how the cumulative policy cost forms a specific growth path under self-reinforcing positive feedback mechanisms. These analyses aim to go beyond static equilibrium analysis, revealing the inherent path dependency, lock-in effects, and evolutionary characteristics of the green electricity market as a complex adaptive system.

### 4. Regionally Differentiated Design
The engine supports testing the effects of two policy modes: "nationwide unified planning" and "adaptation to local conditions". Through the "regionally differentiated synergy" scenario (S6), users can simulate the typical pattern of China's "West-to-East Electricity Transmission": in western regions rich in renewable energy resources, policy parameters can be set to focus on supply-side incentives (e.g., higher equipment investment subsidies) to encourage generation investment; in eastern load center regions, the focus can be on demand-side pull (e.g., higher consumption quotas and consumption subsidies) to ensure green electricity consumption and incentivize local users. By comparing the performance of such differentiated designs with nationwide unified policies (e.g., S3) in terms of overall efficiency, regional equity, and cross-regional trading flows, it provides simulation-based evidence for formulating more precise and adaptive "one region, one policy" strategies.


## 8. Copyright Information
© 2026 SGABM Development Team. Lead Developer: **Liang**.
