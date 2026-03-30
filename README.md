# PSty-Despolymerization
Modeling of Polystyrene Degradation using Kinetic Monte Carlo (KMC).

Access the paper at: https://doi.org/10.1016/j.jaap.2022.105683

# Kinetic Monte Carlo Simulation: Polystyrene Pyrolysis and Depolymerization

This repository contains the Python and Jupyter Notebook source code for a Kinetic Monte Carlo (KMC) simulation focused on describing the thermal degradation, pyrolysis, and depolymerization processes of polystyrene (PSty).

## 🧪 About the Project

The dynamic model simulates the step-by-step kinetic mechanism of plastic degradation. By utilizing computational modeling, it tracks the stochastic nature of polymer chain scissions over time. The algorithm monitors the degradation rate of macroscopic polymer chains, shifts in molecular weight distribution, and the specific yield of volatile degradation products (such as styrene monomers, dimers, and trimers).

## 🗂 File Structure

This repository is organized into core simulation modules and interactive notebooks for executing specific case studies:

**Execution & Case Studies (Jupyter Notebooks):**
* **`Fig7_PS_500.ipynb`**: Simulation setup and execution for Polystyrene degradation at 500°C.
* **`Fig8_PS_600.ipynb`**: Simulation setup and execution for Polystyrene degradation at 600°C.
* **`Fig9_PS_700.ipynb`**: Simulation setup and execution for Polystyrene degradation at 700°C.

**Core Modules (Python Scripts):**
* **`McPackage.py` / `kMC_v2024.py` / `McPackage_ThermalInitiation.py`**: The core Kinetic Monte Carlo engine handling the stochastic simulation loop.
* **`KineticConstantsPS.py`**: Contains the kinetic rate constants, activation energies, and temperature-dependent mathematical models from literature.
* **`InitialDistribution.py`**: Establishes the initial conditions of the polymer chains (molecular weight, chain length distribution).
* **`Fle.py` & `Sigmoide.py`**: Auxiliary scripts for specific mathematical functions and data handling.

## ⚙️ How the Algorithm Works

The mathematical logic bypasses continuous deterministic equations and relies on a stochastic Monte Carlo approach to model individual reaction events. In practice, the program works as follows:

1. **Rate Calculation:** It evaluates the reaction rates for all possible degradation events (e.g., random scission, end-chain scission) at any given moment, based on the current temperature and kinetic parameters.
2. **Time Advancement:** It uses random numbers to determine the exact time until the next degradation reaction occurs (time step tau).
3. **Reaction Selection:** It randomly selects which specific chemical bond will break or which volatile product will be formed, based on probabilities proportional to the rates calculated in step 1.
4. **Update:** It updates the system matrices (e.g., splitting a polymer chain, recording a monomer yield) and repeats the cycle until the polymer is fully degraded or the established final time is reached.

## 🚀 How to Run

**Prerequisites:**
Ensure you have Python 3 installed in your environment. Since the core engine relies on Just-In-Time (JIT) compilation for performance, you must install `numba` along with the standard scientific libraries and Jupyter:

```bash
pip install numpy matplotlib numba jupyter
```

Step-by-step:

Clone the repository to your local machine:

```bash
git clone [https://github.com/gaappucrio/PSty-Despolymerization.git](https://github.com/gaappucrio/PSty-Despolymerization.git)
```

Navigate to the project's root folder:

```bash
cd PSty-Despolymerization
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Run a Simulation:
Once the Jupyter interface opens in your browser, click on one of the execution files (e.g., Fig7_PS_500.ipynb). Inside the notebook, click "Run All" (or run the cells sequentially) to start the Monte Carlo simulation.

⏳ Note: The computational time may vary significantly depending on your machine's hardware, the chosen temperature, and the initial polymer chain lengths. The numba library will compile the functions on the first run, which may take a few extra seconds.

📊 Understanding the Outputs
The outputs are directly plotted and visualized within the Jupyter Notebooks. The execution cells will generate graphs and arrays representing:

The time progression and overall fractional mass loss of the polystyrene.

The specific quantities and mass fractions of generated volatiles (styrene monomers, dimers, trimers, etc.).

The evolution of the remaining polymer's properties (Mn, Mw, and PDI) over time.

High-resolution figures (e.g., .jpeg files) are automatically saved to your directory upon cell completion.
