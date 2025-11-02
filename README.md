# PyBaMM Battery Parameter Updater & Overpotential Visualization

An interactive Streamlit web application for visualizing and analyzing battery overpotentials using PyBaMM (Python Battery Mathematical Modelling).

## Features

- **Parameter Modification**: Update Ecker2015_graphite_halfcell parameters through an intuitive UI
- **Multiple C-Rate Simulation**: Run simulations at various C-rates (0.1C, 0.5C, 1C, 2C, 5C, 10C)
- **Overpotential Visualization**: Interactive Plotly plots showing:
  - Open-circuit voltage (Sauer OCP)
  - Particle concentration overpotential
  - Reaction overpotential
  - Electrolyte concentration overpotential
  - Ohmic electrolyte overpotential
  - Ohmic electrode overpotential
  - Battery voltage (calculated vs PyBaMM)
- **Parameter Function Plots**: Visualize transport and kinetic parameter functions
- **OCP Curve Visualization**: View the Sauer 2018 OCP curve used in the model

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Model_V2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app locally:

```bash
streamlit run half_cell_ecker_sauer_paramupdate_overp.py
```

Or deploy to Streamlit Community Cloud (see deployment section).

## Data Files

The app requires `sauer_2018_graphite_ocv.csv` to be in the same directory as the script.

## License

Open source - feel free to use and modify.

## Author

Developed for battery modeling research using PyBaMM.

