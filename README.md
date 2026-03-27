# PSty-Despolymerization
Modeling of Polystyrene Degradation using Kinetic Monte Carlo (KMC).

Access the paper at: https://doi.org/10.1016/j.jaap.2022.105683

# Kinetic Monte Carlo Simulation: Polystyrene Pyrolysis and Depolymerization

This repository contains the Python source code for a Kinetic Monte Carlo (KMC) simulation focused on describing the thermal degradation, pyrolysis, and depolymerization processes of polystyrene (PSty).

## 🧪 About the Project

The dynamic model simulates the step-by-step kinetic mechanism of plastic degradation. By utilizing computational modeling, it tracks the stochastic nature of polymer chain scissions over time. The algorithm monitors the degradation rate of macroscopic polymer chains, shifts in molecular weight distribution, and the specific yield of volatile degradation products (such as styrene monomers, dimers, and trimers).

## 🗂 File Structure

*(Note: Update the file names below to match your exact repository files if they differ)*

This repository consists of the primary simulation files:

* **`main.py`**: The central script. It establishes the initial polymer chain conditions, operating temperatures, and kinetic constants from the literature for the degradation events, executing the main stochastic simulation loop.
* **`config.json`** (or equivalent parameters file): An auxiliary file or section containing the specific reaction probabilities, rate constants, and system control volume settings.

## ⚙️ How the Algorithm Works

The mathematical logic bypasses continuous deterministic equations and relies on a stochastic Monte Carlo approach to model individual reaction events. In practice, the program works as follows:

1. **Rate Calculation:** It evaluates the reaction rates for all possible degradation events (e.g., random scission, end-chain scission) at any given moment, based on the current temperature and kinetic parameters.
2. **Time Advancement:** It uses random numbers to determine the exact time until the next degradation reaction occurs (time step tau).
3. **Reaction Selection:** It randomly selects which specific chemical bond will break or which volatile product will be formed, based on probabilities proportional to the rates calculated in step 1.
4. **Update:** It updates the system matrices (e.g., splitting a polymer chain, recording a monomer yield) and repeats the cycle until the polymer is fully degraded or the established final time is reached.

## 🚀 How to Run

**Prerequisites:**
Ensure you have Python 3 installed in your environment, along with standard scientific libraries:
```bash
pip install numpy matplotlib
```
Step-by-step:

Clone this repository to your local machine.

```bash
git clone https://github.com/gaappucrio/PSty-Despolymerization.git
```
Navigate to the project's root folder.

Run the simulator in your terminal:

```bash
python main.py
The computational time may vary depending on your machine's hardware and the initial polymer chain lengths defined in the simulation.
```

📊 Understanding the Outputs
After the routine finishes, the script automatically exports data files (e.g., .txt or .csv) to the same directory containing raw data ready for plotting or statistical analysis. Outputs typically include:

degradation_profile.txt: Main data table. It lists the time progression and the overall fractional mass loss of the polystyrene.

product_yields.txt: Tracks the specific quantities and mass fractions of generated volatiles (styrene, dimers, trimers).

molecular_weight.txt: Records the evolution of the remaining polymer's properties, including the number-average molecular weight (Mn) and the polydispersity index (PDI) over time.
