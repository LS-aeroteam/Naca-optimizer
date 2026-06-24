# NACA Optimization Tool

## 📖 The Project
This open-source repository was born from the merger of two parallel projects created for the analysis and study of airfoils. While chatting about our respective work, we realized we were developing complementary tools in different environments:
1. A **MATLAB** solver based on the Panel Method (sources and vortices) for potential flow analysis, Cp calculation, and streamline visualization.
2. A **Python** tool interfaced with **XFOIL** dedicated to optimization and inverse design.

To avoid overlap and create a more powerful and versatile aerodynamic tool, we decided to join forces under this organization and consolidate the working environment in **Python**. The result is an integrated tool that combines algorithmic optimization with rapid aerodynamic analysis.

## ✨ Key Features
* **NACA Airfoil Generation:** Parametric creation of 4-digit NACA airfoils using a cosine distribution of points, essential for ensuring stability in boundary layer analysis.
* **Inverse Design (SLSQP Optimization):** Automatic search for optimal geometric parameters (maximum camber m, position p, thickness t) to reach a target C_L, respecting a maximum C_D limit set by the user.
* **Integrated XFOIL Wrapper:** Complete automation of XFOIL calls for viscous analysis, with autonomous handling of crashes, angle of attack (alpha) ramp-up, and data extraction from polars.
* **Flow Field Analysis (Legacy Panel Method):** Resolution of potential flow to calculate tangential velocities, pressure coefficient distribution, and graphic generation of streamlines around the airfoil.
* **Export and Logging:** Automatic saving of the final geometry in .dat format (ready for CAD or CFD meshing), high-resolution plots, and optimization history in .csv.

## 🛠️ Prerequisites & Installation

Make sure you have **Python 3.8+** installed on your system (the tool is tested in a Windows environment).

The required libraries are:
```bash
pip install numpy scipy matplotlib
Crucial note on XFOIL: The project requires the XFOIL executable to work. Ensure that the xfoil.exe file is present in the same folder as the main script, otherwise the optimization module will not be able to launch the simulations.

🚀 Usage
Run the main script from the terminal:

Bash
python NACA_profile_optimization_tool.py
The program will guide you through a command-line interface to define the operating conditions:

Fluid selection: Air (Standard SL) or Water.

Design speed and chord: For the automatic calculation of Mach and Reynolds numbers.

Optimization parameters: Angle of attack, transition parameter N_crit, desired C_L, and maximum tolerated C_D.

Once the optimization is complete, the tool will automatically create a folder named with the chosen operating parameters (e.g., Results_Re1500000_Alpha2.0_Cl0.6) containing all the output files.

🤝 Contributing
We are aerospace engineering students and enthusiasts: any pull requests to improve code robustness, convert the remaining MATLAB routines to Python (especially the complete panel method part), or add support for 5-digit NACA airfoils are absolutely welcome.