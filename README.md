# Entanglement Entropy Simulation

## Overview

This project simulates the scaling of entanglement entropy in cosmological and gravitational settings. The code is designed to compute entanglement entropy, analyze its behavior over time, and fit the results to extract key parameters. The main focus lies in simulating scenarios such as the Snyder collapse and other cosmological models.

## Features
- **Cosmological Modeling:** Supports various cosmologies, including EdS, LCDM, and Snyder collapse.
- **Entanglement Entropy Calculation:** Computes the entanglement entropy using time-evolved harmonic oscillator models.
- **Custom Integrators:** Leverages custom numerical integration methods for high precision.
- **Visualization:** Plots results like entropy scaling, slopes, and other derived parameters.
- **Configurable Parameters:** Offers customizable settings for precision, cosmology, and system size.

## File Descriptions

### Main Scripts
- **`main.py`**: Entry point of the simulation. Configures the simulation, runs it, and performs analysis like linear fitting. Generates visualizations of the results.
- **`entanglement_entropy_simulation.py`**: Defines the `EntanglementEntropySimulation` class, which encapsulates simulation logic, linear fitting, and data saving/loading.
- **`compute_entanglement_entropy.py`**: Contains functions to compute entanglement entropy and solve Ermakov-like equations.

### Supporting Modules
- **`cosmology.py`**: Implements functions for various cosmological models, including scaling factors, Hubble function, and horizon calculations.
- **`cosmological_functions.py`**: Defines cosmological evolution functions like matter density, radiation density, and dark energy density evolution.
- **`coupling_matrix.py`**: Handles the generation and manipulation of coupling matrices for spherical harmonic expansions.
- **`simulation_parameters.py`**: Centralized configuration file defining physical constants, cosmology settings, and precision parameters.

### Additional Files
Ensure the following custom libraries are available and correctly configured in your system:
- **`numerical_methods.py`**: Provides custom integration and fitting utilities.
- **`plotting_function.py`**: Utility for generating plots.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Ensure Python 3.8+ is installed along with required dependencies:
   ```bash
   pip install numpy scipy matplotlib
   ```
3. Set up custom libraries by ensuring they are available in the PYTHONPATH or placed in the appropriate directory.
4. Configure parameters in `simulation_parameters.py` to suit your requirements.

## Usage
### Running with Python
1. Run the main script:
   ```bash
   python main.py
   ```
2. Outputs, including plots and data, will be saved in the specified directory configured in `simulation_parameters.py`.

### Running with Spyder
1. Open the project in Spyder.
2. Set `main.py` as the main script in Spyder's Run Configuration.
3. Execute the script by clicking the green play button or pressing `F5`.

## Key Configurations
- **Cosmology Selection:** Set `cosmology` in `simulation_parameters.py` to choose the desired cosmology (e.g., `snyder`, `lcdm`, `eds`).
- **Precision:** Adjust integration tolerances (`ermak_atol`, `ermak_rtol`) for the Ermakov-like equation and other computations.
- **Output Options:** Control saving and loading behavior using `save_plots`, `plot_saved_data`, and related flags in `simulation_parameters.py`.

## Example Outputs
- **Entanglement Entropy Scaling:** Plots the entropy scaling as a function of dimensionless area.
- **Slopes:** Shows best-fit slopes and their errors over time.
- **Schwarzschild Time Transformations:** For the Snyder collapse, provides entropy data in Schwarzschild time.

## Dependencies
- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Custom modules (`numerical_methods`, `plotting_function`, `constants`) available at https://github.com/SebastianoTomasi/custom_libraries

## License
This software is freely available for use, modification, and distribution without restriction.
 Attribution to the author is appreciated but not required. If you use this code,
please cite "Entanglement entropy during black hole collapse"  

## Author
Sebastiano Tomasi

