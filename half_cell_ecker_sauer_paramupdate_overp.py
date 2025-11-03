"""
PyBaMM Parameter Updater & Overpotential Visualization Web App

This Streamlit web application allows you to:
1. Update parameters from the Ecker2015_graphite_halfcell dataset via a UI
2. Configure simulation settings (C-rates, number of cycles)
3. Run simulations with updated parameters
4. Visualize voltage component breakdowns (overpotentials) interactively

To run this app:
    streamlit run half_cell_ecker_sauer_paramupdate_overp.py

Requirements:
    - streamlit
    - pybamm
    - plotly
    - numpy
    - scipy
    
Make sure sauer_2018_graphite_ocv.csv is in the same directory as this script.
"""

import os
import inspect
import math
import numpy as np
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import pybamm
from scipy.interpolate import UnivariateSpline
import streamlit as st
from typing import Callable, Dict

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize session state
if 'simulations' not in st.session_state:
    st.session_state.simulations = {}
if 'show_function_plots' not in st.session_state:
    st.session_state.show_function_plots = False
if 'show_ocp_plot' not in st.session_state:
    st.session_state.show_ocp_plot = False

# Helper functions for parameter function visualization
ScalarFunction = Callable[..., pybamm.Symbol]

def _evaluate_scalar(expr, param_values: pybamm.ParameterValues) -> float:
    """Safely reduce a PyBaMM expression (or numeric) to a Python float."""
    evaluated = param_values.evaluate(expr)
    if isinstance(evaluated, (list, tuple)):
        evaluated = np.asarray(evaluated)
    if isinstance(evaluated, np.ndarray):
        if evaluated.size != 1:
            raise ValueError("Expected a scalar value but received an array")
        evaluated = evaluated.item()
    return float(evaluated)

def _build_kwargs(
    fn: ScalarFunction,
    x_value: float,
    resolver: Callable[[str, float], float],
) -> Dict[str, float]:
    """Construct keyword arguments for ``fn`` according to its signature."""
    kwargs = {}
    parameters = inspect.signature(fn).parameters
    for name, param in parameters.items():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        try:
            kwargs[name] = resolver(name, x_value)
        except KeyError as exc:
            raise KeyError(
                f"Unable to resolve argument '{name}' for function '{fn.__name__}'"
            ) from exc
    return kwargs

# Page configuration
st.set_page_config(
    page_title="PyBaMM Parameter Updater & Visualization",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 PyBaMM Battery Parameter Updater & Overpotential Visualization")

with st.expander("ℹ️ How to Use", expanded=False):
    st.markdown("""
    **Step 1:** Configure your simulation settings in the sidebar:
    - Select number of cycles
    - Choose which C-rates to simulate
    
    **Step 2:** Update parameters (optional):
    - Expand parameter sections to modify values
    - Use scale factors to multiply entire parameter functions
    
    **Step 3:** Click "Run Simulations" to execute
    
    **Step 4:** View the interactive plots showing voltage component breakdowns
    
    **Note:** Simulations may take several minutes depending on the number of cycles and C-rates selected.
    """)

st.markdown("---")

# Sidebar for parameter configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Experiment settings
    st.subheader("Experiment Settings")
    num_cycles = st.number_input("Number of Cycles", min_value=1, max_value=10, value=1, step=1)
    
    # C-rate selection
    st.subheader("C-Rate Selection")
    c_rate_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    selected_c_rates = st.multiselect(
        "Select C-rates to simulate",
        options=c_rate_options,
        default=[0.1, 1.0, 2.0, 5.0, 10.0]
    )
    
    # Function visualization toggle
    st.subheader("📈 Function Visualization")
    show_function_plots = st.checkbox(
        "Show Parameter Function Plots",
        value=st.session_state.show_function_plots,
        help="Visualize the parameter functions (exchange current, diffusivity, conductivity, etc.)"
    )
    st.session_state.show_function_plots = show_function_plots
    
    show_ocp_plot = st.checkbox(
        "Show OCP vs SOC Plot",
        value=st.session_state.show_ocp_plot,
        help="Visualize the Sauer OCP curve (Open Circuit Potential vs State of Charge)"
    )
    st.session_state.show_ocp_plot = show_ocp_plot
    
    # Parameter update section
    st.subheader("📝 Parameter Updates")
    st.markdown("Update parameters from Ecker2015_graphite_halfcell dataset")
    
    # Load default parameters to show current values
    @st.cache_data
    def load_default_params():
        return pybamm.ParameterValues("Ecker2015_graphite_halfcell")
    
    default_params = load_default_params()
    
    # Helper function to safely get parameter value
    def get_param_value(param_name, default_val):
        try:
            val = default_params.get(param_name)
            if callable(val):
                return default_val  # Function parameters return default
            return float(val) if val is not None else default_val
        except:
            return default_val
    
    # Parameter updates dictionary
    param_updates = {}
    scale_factors = {}
    
    # 1. Capacity & Current Settings
    with st.expander("🔋 Capacity & Current Settings", expanded=False):
        nominal_capacity = st.number_input(
            "Nominal cell capacity [A.h]",
            value=0.193,  # Default value
            step=0.01,
            format="%.3f",
            key="nominal_capacity"
        )
        param_updates["Nominal cell capacity [A.h]"] = nominal_capacity
        
        current_function = st.number_input(
            "Current function [A]",
            value=0.193,  # Default value
            step=0.001,
            format="%.5f",
            key="current_function"
        )
        param_updates["Current function [A]"] = current_function
    
    # 2. Geometry & Dimensions
    with st.expander("📐 Geometry & Dimensions", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            pos_thickness = st.number_input(
                "Positive electrode thickness [m]",
                value=get_param_value("Positive electrode thickness [m]", 7.4e-05),
                step=1e-06,
                format="%.2e",
                key="pos_thickness"
            )
            param_updates["Positive electrode thickness [m]"] = pos_thickness
            
            neg_thickness = st.number_input(
                "Negative electrode thickness [m]",
                value=get_param_value("Negative electrode thickness [m]", 0.0007),
                step=1e-05,
                format="%.5f",
                key="neg_thickness"
            )
            param_updates["Negative electrode thickness [m]"] = neg_thickness
            
            separator_thickness = st.number_input(
                "Separator thickness [m]",
                value=get_param_value("Separator thickness [m]", 2e-05),
                step=1e-06,
                format="%.2e",
                key="separator_thickness"
            )
            param_updates["Separator thickness [m]"] = separator_thickness
        
        with col2:
            pos_cc_thickness = st.number_input(
                "Positive current collector thickness [m]",
                value=get_param_value("Positive current collector thickness [m]", 1.4e-05),
                step=1e-06,
                format="%.2e",
                key="pos_cc_thickness"
            )
            param_updates["Positive current collector thickness [m]"] = pos_cc_thickness
            
            neg_cc_thickness = st.number_input(
                "Negative current collector thickness [m]",
                value=get_param_value("Negative current collector thickness [m]", 1.4e-05),
                step=1e-06,
                format="%.2e",
                key="neg_cc_thickness"
            )
            param_updates["Negative current collector thickness [m]"] = neg_cc_thickness
            
            electrode_height = st.number_input(
                "Electrode height [m]",
                value=get_param_value("Electrode height [m]", 0.101),
                step=0.001,
                format="%.3f",
                key="electrode_height"
            )
            param_updates["Electrode height [m]"] = electrode_height
            
            electrode_width = st.number_input(
                "Electrode width [m]",
                value=get_param_value("Electrode width [m]", 0.085),
                step=0.001,
                format="%.3f",
                key="electrode_width"
            )
            param_updates["Electrode width [m]"] = electrode_width
    
    # 3. Positive Electrode Properties
    with st.expander("➕ Positive Electrode Properties", expanded=False):
        max_concentration = st.number_input(
            "Maximum concentration in positive electrode [mol.m-3]",
            value=get_param_value("Maximum concentration in positive electrode [mol.m-3]", 31920.0),
            step=100.0,
            format="%.1f",
            key="max_concentration"
        )
        param_updates["Maximum concentration in positive electrode [mol.m-3]"] = max_concentration
        
        initial_concentration_frac = st.slider(
            "Initial concentration fraction (positive electrode)",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            key="initial_conc_frac"
        )
        param_updates["Initial concentration in positive electrode [mol.m-3]"] = initial_concentration_frac * max_concentration
        
        pos_porosity = st.number_input(
            "Positive electrode porosity",
            value=get_param_value("Positive electrode porosity", 0.329),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.3f",
            key="pos_porosity"
        )
        param_updates["Positive electrode porosity"] = pos_porosity
        
        pos_active_frac = st.number_input(
            "Positive electrode active material volume fraction",
            value=get_param_value("Positive electrode active material volume fraction", 0.372403),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.6f",
            key="pos_active_frac"
        )
        param_updates["Positive electrode active material volume fraction"] = pos_active_frac
        
        pos_particle_radius = st.number_input(
            "Positive particle radius [m]",
            value=get_param_value("Positive particle radius [m]", 1.37e-05),
            step=1e-06,
            format="%.2e",
            key="pos_particle_radius"
        )
        param_updates["Positive particle radius [m]"] = pos_particle_radius
        
        pos_conductivity = st.number_input(
            "Positive electrode conductivity [S.m-1]",
            value=get_param_value("Positive electrode conductivity [S.m-1]", 14.0),
            step=1.0,
            format="%.1f",
            key="pos_conductivity"
        )
        param_updates["Positive electrode conductivity [S.m-1]"] = pos_conductivity
        
        pos_charge_transfer = st.number_input(
            "Positive electrode charge transfer coefficient",
            value=get_param_value("Positive electrode charge transfer coefficient", 0.5),
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            format="%.2f",
            key="pos_charge_transfer"
        )
        param_updates["Positive electrode charge transfer coefficient"] = pos_charge_transfer
        
        pos_double_layer = st.number_input(
            "Positive electrode double-layer capacity [F.m-2]",
            value=get_param_value("Positive electrode double-layer capacity [F.m-2]", 0.2),
            step=0.1,
            format="%.2f",
            key="pos_double_layer"
        )
        param_updates["Positive electrode double-layer capacity [F.m-2]"] = pos_double_layer
    
    # 4. Negative Electrode Properties
    with st.expander("➖ Negative Electrode Properties", expanded=False):
        neg_conductivity = st.number_input(
            "Negative electrode conductivity [S.m-1]",
            value=get_param_value("Negative electrode conductivity [S.m-1]", 10776000.0),
            step=100000.0,
            format="%.1f",
            key="neg_conductivity"
        )
        param_updates["Negative electrode conductivity [S.m-1]"] = neg_conductivity
        
        neg_charge_transfer = st.number_input(
            "Negative electrode charge transfer coefficient",
            value=get_param_value("Negative electrode charge transfer coefficient", 0.5),
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            format="%.2f",
            key="neg_charge_transfer"
        )
        param_updates["Negative electrode charge transfer coefficient"] = neg_charge_transfer
        
        neg_double_layer = st.number_input(
            "Negative electrode double-layer capacity [F.m-2]",
            value=get_param_value("Negative electrode double-layer capacity [F.m-2]", 0.2),
            step=0.1,
            format="%.2f",
            key="neg_double_layer"
        )
        param_updates["Negative electrode double-layer capacity [F.m-2]"] = neg_double_layer
    
    # 5. Separator Properties
    with st.expander("🔀 Separator Properties", expanded=False):
        separator_porosity = st.number_input(
            "Separator porosity",
            value=get_param_value("Separator porosity", 0.508),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.3f",
            key="separator_porosity"
        )
        param_updates["Separator porosity"] = separator_porosity
    
    # 6. Electrolyte Properties
    with st.expander("💧 Electrolyte Properties", expanded=False):
        initial_electrolyte_conc = st.number_input(
            "Initial concentration in electrolyte [mol.m-3]",
            value=get_param_value("Initial concentration in electrolyte [mol.m-3]", 1000.0),
            step=10.0,
            format="%.1f",
            key="initial_electrolyte_conc"
        )
        param_updates["Initial concentration in electrolyte [mol.m-3]"] = initial_electrolyte_conc
        
        cation_transference = st.number_input(
            "Cation transference number",
            value=get_param_value("Cation transference number", 0.26),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            key="cation_transference"
        )
        param_updates["Cation transference number"] = cation_transference
        
        thermodynamic_factor = st.number_input(
            "Thermodynamic factor",
            value=get_param_value("Thermodynamic factor", 1.0),
            step=0.1,
            format="%.2f",
            key="thermodynamic_factor"
        )
        param_updates["Thermodynamic factor"] = thermodynamic_factor
        
        EC_initial_conc = st.number_input(
            "EC initial concentration in electrolyte [mol.m-3]",
            value=get_param_value("EC initial concentration in electrolyte [mol.m-3]", 4541.0),
            step=10.0,
            format="%.1f",
            key="EC_initial_conc"
        )
        param_updates["EC initial concentration in electrolyte [mol.m-3]"] = EC_initial_conc
        
        bulk_solvent_conc = st.number_input(
            "Bulk solvent concentration [mol.m-3]",
            value=get_param_value("Bulk solvent concentration [mol.m-3]", 2636.0),
            step=10.0,
            format="%.1f",
            key="bulk_solvent_conc"
        )
        param_updates["Bulk solvent concentration [mol.m-3]"] = bulk_solvent_conc
    
    # 7. Transport Parameters (Tortuosity)
    with st.expander("🚚 Transport Parameters (Tortuosity)", expanded=False):
        st.markdown("*Tortuosity values are converted to Bruggeman coefficients for PyBaMM*")
        st.markdown("*Relationship: τ = ε^(1-b), where τ is tortuosity, ε is porosity, and b is Bruggeman coefficient*")
        
        # Get default porosity and Bruggeman values to calculate default tortuosity
        default_pos_porosity = get_param_value("Positive electrode porosity", 0.329)
        default_pos_bruggeman = get_param_value("Positive electrode Bruggeman coefficient (electrolyte)", 1.6372789338386007)
        default_pos_tortuosity = default_pos_porosity ** (1.0 - default_pos_bruggeman)
        
        default_separator_porosity = get_param_value("Separator porosity", 0.508)
        default_separator_bruggeman = get_param_value("Separator Bruggeman coefficient (electrolyte)", 1.9804586773134945)
        default_separator_tortuosity = default_separator_porosity ** (1.0 - default_separator_bruggeman)
        
        # Get current porosity values from param_updates (set earlier in sidebar)
        pos_porosity_val = param_updates.get("Positive electrode porosity")
        if pos_porosity_val is None:
            pos_porosity_val = default_pos_porosity
        
        separator_porosity_val = param_updates.get("Separator porosity")
        if separator_porosity_val is None:
            separator_porosity_val = default_separator_porosity
        
        # Positive electrode tortuosity
        pos_tortuosity = st.number_input(
            "Positive electrode tortuosity (electrolyte)",
            value=float(default_pos_tortuosity),  # Calculated from default Bruggeman coefficient
            min_value=1.0,
            step=0.1,
            format="%.2f",
            key="pos_tortuosity",
            help=f"Tortuosity for electrolyte transport in positive electrode (porosity: {pos_porosity_val:.3f})"
        )
        
        # Calculate Bruggeman coefficient from tortuosity: b = 1 - ln(τ) / ln(ε)
        if pos_porosity_val > 0 and pos_porosity_val < 1 and pos_tortuosity >= 1.0:
            pos_bruggeman_electrolyte = 1.0 - (np.log(pos_tortuosity) / np.log(pos_porosity_val))
            param_updates["Positive electrode Bruggeman coefficient (electrolyte)"] = pos_bruggeman_electrolyte
            st.info(f"Calculated Bruggeman coefficient: {pos_bruggeman_electrolyte:.6f}")
        else:
            st.warning("Porosity must be between 0 and 1, and tortuosity ≥ 1 to calculate Bruggeman coefficient")
            pos_bruggeman_electrolyte = default_pos_bruggeman
            param_updates["Positive electrode Bruggeman coefficient (electrolyte)"] = pos_bruggeman_electrolyte
        
        # Separator tortuosity
        separator_tortuosity = st.number_input(
            "Separator tortuosity (electrolyte)",
            value=float(default_separator_tortuosity),  # Calculated from default Bruggeman coefficient
            min_value=1.0,
            step=0.1,
            format="%.2f",
            key="separator_tortuosity",
            help=f"Tortuosity for electrolyte transport in separator (porosity: {separator_porosity_val:.3f})"
        )
        
        # Calculate Bruggeman coefficient from tortuosity
        if separator_porosity_val > 0 and separator_porosity_val < 1 and separator_tortuosity >= 1.0:
            separator_bruggeman = 1.0 - (np.log(separator_tortuosity) / np.log(separator_porosity_val))
            param_updates["Separator Bruggeman coefficient (electrolyte)"] = separator_bruggeman
            st.info(f"Calculated Bruggeman coefficient: {separator_bruggeman:.6f}")
        else:
            st.warning("Porosity must be between 0 and 1, and tortuosity ≥ 1 to calculate Bruggeman coefficient")
            separator_bruggeman = default_separator_bruggeman
            param_updates["Separator Bruggeman coefficient (electrolyte)"] = separator_bruggeman
        
        # Keep electrode Bruggeman coefficient (typically not used for tortuosity conversion)
        pos_bruggeman_electrode = st.number_input(
            "Positive electrode Bruggeman coefficient (electrode)",
            value=get_param_value("Positive electrode Bruggeman coefficient (electrode)", 0.0),
            step=0.1,
            format="%.2f",
            key="pos_bruggeman_electrode"
        )
        param_updates["Positive electrode Bruggeman coefficient (electrode)"] = pos_bruggeman_electrode
    
    # 8. Reaction Kinetics & Exchange Current (Scale Factors)
    with st.expander("⚡ Reaction Kinetics & Exchange Current (Scale Factors)", expanded=False):
        st.markdown("*Scale factors multiply the original function values*")
        
        exchange_current_scale = st.number_input(
            "Positive electrode exchange-current density scale factor",
            value=1.0,
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="exchange_scale",
            help="Multiplier for exchange current density"
        )
        scale_factors["Positive electrode exchange-current density [A.m-2]"] = exchange_current_scale
        
        li_metal_exchange_scale = st.number_input(
            "Lithium metal exchange-current density scale factor",
            value=1.0,
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="li_metal_exchange_scale"
        )
        scale_factors["Exchange-current density for lithium metal electrode [A.m-2]"] = li_metal_exchange_scale
        
        plating_exchange_scale = st.number_input(
            "Plating exchange-current density scale factor",
            value=1.0,
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="plating_exchange_scale"
        )
        scale_factors["Exchange-current density for plating [A.m-2]"] = plating_exchange_scale
        
        stripping_exchange_scale = st.number_input(
            "Stripping exchange-current density scale factor",
            value=1.0,
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="stripping_exchange_scale"
        )
        scale_factors["Exchange-current density for stripping [A.m-2]"] = stripping_exchange_scale
    
    # 9. Diffusion Parameters (Scale Factors)
    with st.expander("🌊 Diffusion Parameters (Scale Factors)", expanded=False):
        st.markdown("*Scale factors multiply the original function values*")
        
        diffusivity_scale = st.number_input(
            "Positive particle diffusivity scale factor",
            value=1.0,
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="diffusivity_scale",
            help="Multiplier for particle diffusivity"
        )
        scale_factors["Positive particle diffusivity [m2.s-1]"] = diffusivity_scale
        
        electrolyte_diffusivity_scale = st.number_input(
            "Electrolyte diffusivity scale factor",
            value=1.0,
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="electrolyte_diff_scale"
        )
        scale_factors["Electrolyte diffusivity [m2.s-1]"] = electrolyte_diffusivity_scale
        
        EC_diffusivity = st.number_input(
            "EC diffusivity [m2.s-1]",
            value=get_param_value("EC diffusivity [m2.s-1]", 2e-18),
            step=1e-19,
            format="%.2e",
            key="EC_diffusivity"
        )
        param_updates["EC diffusivity [m2.s-1]"] = EC_diffusivity
    
    # 10. Conductivity Parameters (Scale Factors)
    with st.expander("🔌 Conductivity Parameters (Scale Factors)", expanded=False):
        st.markdown("*Scale factors multiply the original function values*")
        
        electrolyte_conductivity_scale = st.number_input(
            "Electrolyte conductivity scale factor",
            value=1.0,
            min_value=0.1,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="electrolyte_cond_scale"
        )
        scale_factors["Electrolyte conductivity [S.m-1]"] = electrolyte_conductivity_scale
        
        pos_cc_conductivity = st.number_input(
            "Positive current collector conductivity [S.m-1]",
            value=get_param_value("Positive current collector conductivity [S.m-1]", 58411000.0),
            step=100000.0,
            format="%.1f",
            key="pos_cc_conductivity"
        )
        param_updates["Positive current collector conductivity [S.m-1]"] = pos_cc_conductivity
    
    # 11. SEI (Solid Electrolyte Interphase) Parameters
    with st.expander("🛡️ SEI Parameters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            SEI_resistivity = st.number_input(
                "SEI resistivity [Ohm.m]",
                value=get_param_value("SEI resistivity [Ohm.m]", 200000.0),
                step=1000.0,
                format="%.1f",
                key="SEI_resistivity"
            )
            param_updates["SEI resistivity [Ohm.m]"] = SEI_resistivity
            
            SEI_ocp = st.number_input(
                "SEI open-circuit potential [V]",
                value=get_param_value("SEI open-circuit potential [V]", 0.4),
                step=0.1,
                format="%.2f",
                key="SEI_ocp"
            )
            param_updates["SEI open-circuit potential [V]"] = SEI_ocp
            
            inner_SEI_ocp = st.number_input(
                "Inner SEI open-circuit potential [V]",
                value=get_param_value("Inner SEI open-circuit potential [V]", 0.1),
                step=0.1,
                format="%.2f",
                key="inner_SEI_ocp"
            )
            param_updates["Inner SEI open-circuit potential [V]"] = inner_SEI_ocp
            
            outer_SEI_ocp = st.number_input(
                "Outer SEI open-circuit potential [V]",
                value=get_param_value("Outer SEI open-circuit potential [V]", 0.8),
                step=0.1,
                format="%.2f",
                key="outer_SEI_ocp"
            )
            param_updates["Outer SEI open-circuit potential [V]"] = outer_SEI_ocp
        
        with col2:
            SEI_reaction_exchange = st.number_input(
                "SEI reaction exchange current density [A.m-2]",
                value=get_param_value("SEI reaction exchange current density [A.m-2]", 1.5e-07),
                step=1e-08,
                format="%.2e",
                key="SEI_reaction_exchange"
            )
            param_updates["SEI reaction exchange current density [A.m-2]"] = SEI_reaction_exchange
            
            initial_inner_SEI = st.number_input(
                "Initial inner SEI thickness [m]",
                value=get_param_value("Initial inner SEI thickness [m]", 2.5e-09),
                step=1e-10,
                format="%.2e",
                key="initial_inner_SEI"
            )
            param_updates["Initial inner SEI thickness [m]"] = initial_inner_SEI
            
            initial_outer_SEI = st.number_input(
                "Initial outer SEI thickness [m]",
                value=get_param_value("Initial outer SEI thickness [m]", 2.5e-09),
                step=1e-10,
                format="%.2e",
                key="initial_outer_SEI"
            )
            param_updates["Initial outer SEI thickness [m]"] = initial_outer_SEI
    
    # 12. Voltage Cutoffs (Used in Experiment)
    with st.expander("⚡ Voltage Cutoffs (Used in Experiment)", expanded=True):
        st.markdown("**These values are used to define discharge and charge termination in the experiment**")
        
        lower_cutoff = st.number_input(
            "Lower voltage cut-off [V] (Discharge termination)",
            value=0.015,  # Default from experiment
            step=0.001,
            format="%.3f",
            key="lower_cutoff",
            help="Voltage at which discharge terminates"
        )
        param_updates["Lower voltage cut-off [V]"] = lower_cutoff
        
        upper_cutoff = st.number_input(
            "Upper voltage cut-off [V] (Charge termination)",
            value=0.75,  # Default from experiment
            step=0.01,
            format="%.2f",
            key="upper_cutoff",
            help="Voltage at which charge terminates"
        )
        param_updates["Upper voltage cut-off [V]"] = upper_cutoff
    
    # 13. Thermal Properties
    with st.expander("🌡️ Thermal Properties", expanded=False):
        ambient_temp = st.number_input(
            "Ambient temperature [K]",
            value=get_param_value("Ambient temperature [K]", 298.15),
            step=1.0,
            format="%.2f",
            key="ambient_temp"
        )
        param_updates["Ambient temperature [K]"] = ambient_temp
        
        initial_temp = st.number_input(
            "Initial temperature [K]",
            value=get_param_value("Initial temperature [K]", 298.15),
            step=1.0,
            format="%.2f",
            key="initial_temp"
        )
        param_updates["Initial temperature [K]"] = initial_temp
        
        ref_temp = st.number_input(
            "Reference temperature [K]",
            value=get_param_value("Reference temperature [K]", 296.15),
            step=1.0,
            format="%.2f",
            key="ref_temp"
        )
        param_updates["Reference temperature [K]"] = ref_temp
    
    # 14. Contact Resistance
    with st.expander("🔗 Contact & Resistance", expanded=False):
        contact_resistance = st.number_input(
            "Contact resistance [Ohm]",
            value=get_param_value("Contact resistance [Ohm]", 0.0),
            step=0.001,
            format="%.3f",
            key="contact_resistance"
        )
        param_updates["Contact resistance [Ohm]"] = contact_resistance
    
    # 15. Lithium Plating Parameters
    with st.expander("⚗️ Lithium Plating Parameters", expanded=False):
        li_plating_rate = st.number_input(
            "Lithium plating kinetic rate constant [m.s-1]",
            value=get_param_value("Lithium plating kinetic rate constant [m.s-1]", 1e-10),
            step=1e-11,
            format="%.2e",
            key="li_plating_rate"
        )
        param_updates["Lithium plating kinetic rate constant [m.s-1]"] = li_plating_rate
        
        li_plating_transfer = st.number_input(
            "Lithium plating transfer coefficient",
            value=get_param_value("Lithium plating transfer coefficient", 0.5),
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            format="%.2f",
            key="li_plating_transfer"
        )
        param_updates["Lithium plating transfer coefficient"] = li_plating_transfer
        
        initial_plated_li = st.number_input(
            "Initial plated lithium concentration [mol.m-3]",
            value=get_param_value("Initial plated lithium concentration [mol.m-3]", 0.0),
            step=100.0,
            format="%.1f",
            key="initial_plated_li"
        )
        param_updates["Initial plated lithium concentration [mol.m-3]"] = initial_plated_li
        
        typical_plated_li = st.number_input(
            "Typical plated lithium concentration [mol.m-3]",
            value=get_param_value("Typical plated lithium concentration [mol.m-3]", 1000.0),
            step=100.0,
            format="%.1f",
            key="typical_plated_li"
        )
        param_updates["Typical plated lithium concentration [mol.m-3]"] = typical_plated_li
    
    # 16. Material Properties (Density, Heat Capacity, etc.)
    with st.expander("🧱 Material Properties", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            pos_density = st.number_input(
                "Positive electrode density [kg.m-3]",
                value=get_param_value("Positive electrode density [kg.m-3]", 1555.0),
                step=10.0,
                format="%.1f",
                key="pos_density"
            )
            param_updates["Positive electrode density [kg.m-3]"] = pos_density
            
            pos_specific_heat = st.number_input(
                "Positive electrode specific heat capacity [J.kg-1.K-1]",
                value=get_param_value("Positive electrode specific heat capacity [J.kg-1.K-1]", 1437.0),
                step=10.0,
                format="%.1f",
                key="pos_specific_heat"
            )
            param_updates["Positive electrode specific heat capacity [J.kg-1.K-1]"] = pos_specific_heat
            
            pos_thermal_cond = st.number_input(
                "Positive electrode thermal conductivity [W.m-1.K-1]",
                value=get_param_value("Positive electrode thermal conductivity [W.m-1.K-1]", 1.58),
                step=0.1,
                format="%.2f",
                key="pos_thermal_cond"
            )
            param_updates["Positive electrode thermal conductivity [W.m-1.K-1]"] = pos_thermal_cond
        
        with col2:
            separator_density = st.number_input(
                "Separator density [kg.m-3]",
                value=get_param_value("Separator density [kg.m-3]", 1017.0),
                step=10.0,
                format="%.1f",
                key="separator_density"
            )
            param_updates["Separator density [kg.m-3]"] = separator_density
            
            separator_specific_heat = st.number_input(
                "Separator specific heat capacity [J.kg-1.K-1]",
                value=get_param_value("Separator specific heat capacity [J.kg-1.K-1]", 1978.0),
                step=10.0,
                format="%.1f",
                key="separator_specific_heat"
            )
            param_updates["Separator specific heat capacity [J.kg-1.K-1]"] = separator_specific_heat
            
            separator_thermal_cond = st.number_input(
                "Separator thermal conductivity [W.m-1.K-1]",
                value=get_param_value("Separator thermal conductivity [W.m-1.K-1]", 0.34),
                step=0.01,
                format="%.2f",
                key="separator_thermal_cond"
            )
            param_updates["Separator thermal conductivity [W.m-1.K-1]"] = separator_thermal_cond
            
            pos_cc_density = st.number_input(
                "Positive current collector density [kg.m-3]",
                value=get_param_value("Positive current collector density [kg.m-3]", 8933.0),
                step=10.0,
                format="%.1f",
                key="pos_cc_density"
            )
            param_updates["Positive current collector density [kg.m-3]"] = pos_cc_density
            
            pos_cc_specific_heat = st.number_input(
                "Positive current collector specific heat capacity [J.kg-1.K-1]",
                value=get_param_value("Positive current collector specific heat capacity [J.kg-1.K-1]", 385.0),
                step=1.0,
                format="%.1f",
                key="pos_cc_specific_heat"
            )
            param_updates["Positive current collector specific heat capacity [J.kg-1.K-1]"] = pos_cc_specific_heat
            
            pos_cc_thermal_cond = st.number_input(
                "Positive current collector thermal conductivity [W.m-1.K-1]",
                value=get_param_value("Positive current collector thermal conductivity [W.m-1.K-1]", 398.0),
                step=1.0,
                format="%.1f",
                key="pos_cc_thermal_cond"
            )
            param_updates["Positive current collector thermal conductivity [W.m-1.K-1]"] = pos_cc_thermal_cond

# Function to create parameter function plots
def create_parameter_function_plots(param_values: pybamm.ParameterValues):
    """Create plots for parameter functions used in the model."""
    parameter_dict = pybamm.parameter_sets["Ecker2015_graphite_halfcell"]
    
    reference_temperature = param_values["Reference temperature [K]"]
    electrolyte_init_conc = param_values["Initial concentration in electrolyte [mol.m-3]"]
    typical_plated_li_conc = param_values["Typical plated lithium concentration [mol.m-3]"]
    c_n_max = param_values["Maximum concentration in positive electrode [mol.m-3]"]
    
    sto_range = np.linspace(1e-4, 0.999, 200)
    electrolyte_conc_range = np.linspace(0.5 * electrolyte_init_conc, 1.5 * electrolyte_init_conc, 200)
    
    # Get parameter functions
    li_metal_exchange_fn: ScalarFunction = parameter_dict[
        "Exchange-current density for lithium metal electrode [A.m-2]"
    ]
    graphite_diffusivity_fn: ScalarFunction = parameter_dict[
        "Positive particle diffusivity [m2.s-1]"
    ]
    graphite_exchange_fn: ScalarFunction = parameter_dict[
        "Positive electrode exchange-current density [A.m-2]"
    ]
    electrolyte_diffusivity_fn: ScalarFunction = parameter_dict[
        "Electrolyte diffusivity [m2.s-1]"
    ]
    electrolyte_conductivity_fn: ScalarFunction = parameter_dict[
        "Electrolyte conductivity [S.m-1]"
    ]
    
    # Resolver functions
    def li_metal_resolver(arg_name: str, x_value: float) -> float:
        key = arg_name.lower()
        if key in {"c_e", "ce", "electrolyte_concentration"}:
            return x_value
        if key in {"c_li", "cli", "c_plated", "c_plated_li", "c_lithium"}:
            return typical_plated_li_conc
        if key in {"t", "temperature", "temp"}:
            return reference_temperature
        raise KeyError(arg_name)
    
    def graphite_diffusivity_resolver(arg_name: str, x_value: float) -> float:
        key = arg_name.lower()
        if key in {"sto", "theta", "x", "stoichiometry"}:
            return x_value
        if key in {"t", "temperature", "temp"}:
            return reference_temperature
        raise KeyError(arg_name)
    
    def graphite_exchange_resolver(arg_name: str, x_value: float) -> float:
        key = arg_name.lower()
        if key in {"c_e", "ce", "electrolyte_concentration"}:
            return electrolyte_init_conc
        if key in {"c_s_surf", "cs", "c_s", "c_surf", "c_surf_p"}:
            return x_value * c_n_max
        if key in {"c_s_max", "cs_max", "c_max", "c_surf_max", "max_concentration"}:
            return c_n_max
        if key in {"t", "temperature", "temp"}:
            return reference_temperature
        raise KeyError(arg_name)
    
    def electrolyte_resolver(arg_name: str, x_value: float) -> float:
        key = arg_name.lower()
        if key in {"c_e", "ce", "electrolyte_concentration"}:
            return x_value
        if key in {"t", "temperature", "temp"}:
            return reference_temperature
        raise KeyError(arg_name)
    
    # Subplot configurations
    subplot_configs = [
        {
            "title": "Li Metal Exchange-Current Density",
            "fn": li_metal_exchange_fn,
            "x_values": electrolyte_conc_range,
            "x_label": "Electrolyte concentration [mol.m-3]",
            "y_label": "Exchange-current density [A.m-2]",
            "color": "#1f77b4",
            "resolver": li_metal_resolver,
        },
        {
            "title": "Graphite Particle Diffusivity",
            "fn": graphite_diffusivity_fn,
            "x_values": sto_range,
            "x_label": "Stoichiometry (dimensionless)",
            "y_label": "Diffusivity [m2.s-1]",
            "color": "#ff7f0e",
            "resolver": graphite_diffusivity_resolver,
            "y_tickformat": ".1e",
        },
        {
            "title": "Graphite Exchange-Current Density",
            "fn": graphite_exchange_fn,
            "x_values": sto_range,
            "x_label": "Stoichiometry (dimensionless)",
            "y_label": "Exchange-current density [A.m-2]",
            "color": "#2ca02c",
            "resolver": graphite_exchange_resolver,
        },
        {
            "title": "Electrolyte Diffusivity",
            "fn": electrolyte_diffusivity_fn,
            "x_values": electrolyte_conc_range,
            "x_label": "Electrolyte concentration [mol.m-3]",
            "y_label": "Diffusivity [m2.s-1]",
            "color": "#d62728",
            "resolver": electrolyte_resolver,
            "y_tickformat": ".1e",
        },
        {
            "title": "Electrolyte Conductivity",
            "fn": electrolyte_conductivity_fn,
            "x_values": electrolyte_conc_range,
            "x_label": "Electrolyte concentration [mol.m-3]",
            "y_label": "Conductivity [S.m-1]",
            "color": "#9467bd",
            "resolver": electrolyte_resolver,
        },
    ]
    
    num_cols = 2
    num_rows = math.ceil(len(subplot_configs) / num_cols)
    
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=[cfg["title"] for cfg in subplot_configs],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )
    
    for index, cfg in enumerate(subplot_configs):
        row = index // num_cols + 1
        col = index % num_cols + 1
        y_values = []
        for x_val in cfg["x_values"]:
            try:
                kwargs = _build_kwargs(cfg["fn"], float(x_val), cfg["resolver"])
                expr = cfg["fn"](**kwargs)
                y_val = _evaluate_scalar(expr, param_values)
                y_values.append(y_val)
            except Exception as e:
                st.warning(f"Error evaluating {cfg['title']} at x={x_val}: {e}")
                y_values.append(np.nan)
        
        fig.add_trace(
            go.Scatter(
                x=cfg["x_values"],
                y=y_values,
                mode="lines",
                line=dict(color=cfg["color"], width=3),
                name=cfg["title"],
                hovertemplate=f"<b>{cfg['title']}</b><br>%{{y:.3e}}<br>%{{x:.3f}}<extra></extra>",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        
        fig.update_xaxes(title_text=cfg["x_label"], row=row, col=col, showgrid=True, gridcolor="lightgray")
        
        yaxis_params = {
            "title_text": cfg["y_label"],
            "row": row,
            "col": col,
            "showgrid": True,
            "gridcolor": "lightgray",
        }
        if "y_tickformat" in cfg:
            yaxis_params["tickformat"] = cfg["y_tickformat"]
        fig.update_yaxes(**yaxis_params)
    
    fig.update_layout(
        title="Ecker2015 Graphite Half-Cell: Key Transport and Kinetic Functions",
        height=500 * num_rows,
        width=1200,
        showlegend=False,
        template="plotly_white",
        hovermode="closest",
        margin=dict(t=120, r=40, b=60, l=80),
    )
    
    return fig

# Function to create OCP vs SOC plot
def create_ocp_plot(script_dir: str):
    """Create a plot of the Sauer OCP curve (OCP vs SOC/Stoichiometry)."""
    sauer_csv = os.path.join(script_dir, "sauer_2018_graphite_ocv.csv")
    
    if not os.path.exists(sauer_csv):
        raise FileNotFoundError(f"Sauer OCP file not found at {sauer_csv}")
    
    # Load and sort data
    sauer_data = np.loadtxt(sauer_csv, delimiter=",")
    indices = np.argsort(sauer_data[:, 0])
    sauer_sto = sauer_data[indices, 0]
    sauer_ocv = sauer_data[indices, 1]
    
    # Smooth OCP using a cubic spline (same as used in simulations)
    spline = UnivariateSpline(sauer_sto, sauer_ocv, s=len(sauer_sto) * 1e-4)
    spline_sto = np.linspace(np.min(sauer_sto), np.max(sauer_sto), 400)
    spline_ocv = spline(spline_sto)
    
    # Extend to full range [0, 1]
    sto_lower = 0.0
    sto_upper = 1.0
    extended_sto = np.concatenate(([sto_lower], spline_sto, [sto_upper]))
    extended_ocv = np.concatenate((
        [spline(sto_lower)],
        spline_ocv,
        [spline(sto_upper)],
    ))
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Plot raw data points
    fig.add_trace(go.Scatter(
        x=sauer_sto,
        y=sauer_ocv,
        mode='markers',
        marker=dict(size=4, color='#9467bd', opacity=0.6),
        name='Raw Data',
        hovertemplate="<b>Raw Data</b><br>OCP: %{y:.4f} V<br>SOC: %{x:.4f}<extra></extra>",
    ))
    
    # Plot smoothed curve
    fig.add_trace(go.Scatter(
        x=extended_sto,
        y=extended_ocv,
        mode='lines',
        line=dict(color='#1f77b4', width=3),
        name='Smoothed Curve (Used in Model)',
        hovertemplate="<b>Smoothed Curve</b><br>OCP: %{y:.4f} V<br>SOC: %{x:.4f}<extra></extra>",
    ))
    
    fig.update_layout(
        title="Sauer 2018 Graphite OCP Curve (Open Circuit Potential vs State of Charge)",
        xaxis_title="State of Charge (SOC) / Stoichiometry [-]",
        yaxis_title="Open Circuit Potential [V]",
        template="plotly_white",
        height=600,
        width=900,
        hovermode="closest",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(t=80, r=40, b=60, l=80),
    )
    
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", range=[0, 1])
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")
    
    return fig

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Simulation & Visualization")
    
    # Show parameter function plots if enabled
    if show_function_plots:
        st.subheader("📈 Parameter Function Plots")
        st.markdown("*These plots show the original parameter functions defined in the Ecker2015_graphite_halfcell parameter set. These are read-only visualizations of the base functions.*")
        
        try:
            # Use default parameter values to show the original functions
            # (not affected by parameter updates or scale factors)
            default_param_vals = pybamm.ParameterValues("Ecker2015_graphite_halfcell")
            
            # Create and display the plots
            function_fig = create_parameter_function_plots(default_param_vals)
            st.plotly_chart(function_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating function plots: {str(e)}")
            st.exception(e)
        
        st.markdown("---")
    
    # Show OCP vs SOC plot if enabled
    if show_ocp_plot:
        st.subheader("🔋 OCP vs SOC Plot")
        st.markdown("*This plot shows the Sauer 2018 OCP curve currently used in the model (both raw data and smoothed curve).*")
        
        try:
            # Create and display the OCP plot
            ocp_fig = create_ocp_plot(script_dir)
            st.plotly_chart(ocp_fig, use_container_width=True)
        except FileNotFoundError as e:
            st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error generating OCP plot: {str(e)}")
            st.exception(e)
        
        st.markdown("---")
    
    # Run button
    if st.button("🚀 Run Simulations", type="primary", use_container_width=True):
        if not selected_c_rates:
            st.error("Please select at least one C-rate to simulate!")
        else:
            with st.spinner("Running simulations... This may take a few minutes."):
                # Initialize model
                model = pybamm.lithium_ion.DFN({"working electrode": "positive"})
                param_Gr = pybamm.ParameterValues("Ecker2015_graphite_halfcell")
                
                # Load and process Sauer OCP
                sauer_csv = os.path.join(script_dir, "sauer_2018_graphite_ocv.csv")
                if not os.path.exists(sauer_csv):
                    st.error(f"Sauer OCP file not found at {sauer_csv}")
                    st.stop()
                
                sauer_data = np.loadtxt(sauer_csv, delimiter=",")
                indices = np.argsort(sauer_data[:, 0])
                sauer_sto = sauer_data[indices, 0]
                sauer_ocv = sauer_data[indices, 1]
                
                spline = UnivariateSpline(sauer_sto, sauer_ocv, s=len(sauer_sto) * 1e-4)
                spline_sto = np.linspace(np.min(sauer_sto), np.max(sauer_sto), 400)
                spline_ocv = spline(spline_sto)
                
                sto_lower = 0.0
                sto_upper = 1.0
                extended_sto = np.concatenate(([sto_lower], spline_sto, [sto_upper]))
                extended_ocv = np.concatenate((
                    [spline(sto_lower)],
                    spline_ocv,
                    [spline(sto_upper)],
                ))
                
                def sauer_ocp_function(sto):
                    return pybamm.Interpolant(
                        extended_sto,
                        extended_ocv,
                        sto,
                        extrapolate=True,
                    )
                
                # Update parameters - filter out parameters that don't exist in the parameter set
                valid_param_updates = {}
                for param_name, param_value in param_updates.items():
                    if param_name in param_Gr:
                        valid_param_updates[param_name] = param_value
                    else:
                        st.warning(f"Parameter '{param_name}' not found in parameter set, skipping...")
                
                param_Gr.update({
                    "Positive electrode OCP [V]": sauer_ocp_function,
                    **valid_param_updates
                })
                
                # Apply scale factors using PyBaMM's parameter scaling
                # General function to scale any parameter (works for functions and constants)
                def apply_scale_factor(param_name, scale_value):
                    """Apply a scale factor to a parameter, handling both functions and constants"""
                    if param_name not in param_Gr:
                        return
                    try:
                        original_val = param_Gr[param_name]
                        if callable(original_val):
                            # For function-based parameters, create a wrapper
                            def scaled_func(*args, **kwargs):
                                result = original_val(*args, **kwargs)
                                if isinstance(result, np.ndarray):
                                    return result * scale_value
                                elif hasattr(result, '__iter__') and not isinstance(result, str):
                                    return np.array(result) * scale_value
                                return result * scale_value
                            param_Gr[param_name] = scaled_func
                        else:
                            # For constant parameters, just multiply
                            param_Gr[param_name] = original_val * scale_value
                    except Exception as e:
                        st.warning(f"Could not scale {param_name}: {e}")
                
                # Apply all scale factors
                for param_name, scale_value in scale_factors.items():
                    if scale_value != 1.0:
                        apply_scale_factor(param_name, scale_value)
                
                # Run simulations
                progress_bar = st.progress(0)
                status_text = st.empty()
                simulations = {}
                
                for idx, c_rate in enumerate(selected_c_rates):
                    status_text.text(f"Running {c_rate}C simulation ({idx+1}/{len(selected_c_rates)})...")
                    
                    # Adaptive sampling period
                    if c_rate >= 20:
                        period = "0.05 seconds"
                    elif c_rate >= 10:
                        period = "0.1 seconds"
                    elif c_rate >= 5:
                        period = "0.5 seconds"
                    elif c_rate >= 2:
                        period = "1 second"
                    else:
                        period = "10 seconds"
                    
                    # Create experiment with specified cycles using cutoff voltages from UI
                    # Get cutoff voltages from param_updates (use UI values or defaults)
                    lower_cutoff_v = param_updates.get("Lower voltage cut-off [V]", 0.015)
                    upper_cutoff_v = param_updates.get("Upper voltage cut-off [V]", 0.75)
                    
                    experiment_steps = []
                    for cycle in range(num_cycles):
                        # Use cutoff voltages from UI - format with appropriate precision
                        experiment_steps.append(f"Discharge at {c_rate}C until {lower_cutoff_v:.3f} V")
                        experiment_steps.append(f"Charge at {c_rate}C until {upper_cutoff_v:.2f} V")
                    
                    experiment = pybamm.Experiment(
                        experiment_steps,
                        period=period,
                    )
                    
                    params = param_Gr.copy()
                    sim = pybamm.Simulation(
                        model,
                        parameter_values=params,
                        experiment=experiment,
                        solver=pybamm.CasadiSolver(mode="safe", dt_max=60),
                    )
                    
                    try:
                        sim.solve()
                        simulations[c_rate] = sim
                    except Exception as e:
                        st.warning(f"Simulation failed for {c_rate}C: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(selected_c_rates))
                
                status_text.text("Simulations complete!")
                st.session_state.simulations = simulations
                st.success(f"✅ Completed {len(simulations)} simulation(s)!")

with col2:
    st.header("📋 Current Parameters")
    if st.session_state.simulations:
        st.success(f"✅ {len(st.session_state.simulations)} simulation(s) ready")
        for c_rate in st.session_state.simulations.keys():
            st.text(f"• {c_rate}C")
    else:
        st.info("No simulations run yet. Click 'Run Simulations' to start.")

# Plotting section
if st.session_state.simulations:
    st.markdown("---")
    st.header("📈 Overpotential Visualizations")
    
    # Import plotting functions from the original script
    STACK_COMPONENTS_COMBINED = [
        ("Battery particle concentration overpotential [V]", "Particle concentration overpotential"),
        ("X-averaged battery reaction overpotential [V]", "Reaction overpotential"),
        ("Lithium metal interface reaction overpotential [V]", "Lithium metal reaction overpotential"),
        ("X-averaged battery concentration overpotential [V]", "Electrolyte concentration overpotential"),
        ("X-averaged battery electrolyte ohmic losses [V]", "Ohmic electrolyte overpotential"),
        ("X-averaged battery solid phase ohmic losses [V]", "Ohmic electrode overpotential"),
    ]
    
    COLOR_PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]
    
    def hex_to_rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"
    
    def add_stacked_traces(fig, sim, row_index, col_index, split):
        solution = sim.solution
        time_hours = solution["Time [s]"].entries / 3600
        voltage = solution["Battery voltage [V]"].entries
        color_iter = iter(COLOR_PALETTE)
        
        if not split:
            ocv = solution["Battery open-circuit voltage [V]"].entries
            if ocv.ndim > 1:
                ocv = np.mean(ocv, axis=0)
            initial_ocv = ocv[0]
            baseline = np.full_like(time_hours, initial_ocv, dtype=float)
            fig.add_trace(
                go.Scatter(x=time_hours, y=baseline, mode="lines",
                          line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False),
                row=row_index, col=col_index,
            )
            fig.add_trace(
                go.Scatter(x=time_hours, y=ocv, fill="tonexty", mode="lines",
                          line=dict(color="#636EFA", width=0.8),
                          fillcolor="rgba(99,110,250,0.35)",
                          name="Open-circuit voltage (Sauer OCP)",
                          legendgroup="OCV", showlegend=(row_index == 1 and col_index == 1)),
                row=row_index, col=col_index,
            )
            top = ocv.copy()
            components = STACK_COMPONENTS_COMBINED
        else:
            ocp_n = solution["Battery negative electrode bulk open-circuit potential [V]"].entries
            ocp_p = solution["Battery positive electrode bulk open-circuit potential [V]"].entries
            # Handle spatial dimensions
            if ocp_n.ndim > 1:
                ocp_n = np.mean(ocp_n, axis=0)
            if ocp_p.ndim > 1:
                ocp_p = np.mean(ocp_p, axis=0)
            ocv = ocp_p - ocp_n
            
            # For split plot, show individual OCPs but use same stacking logic as combined
            # Negative OCP is shown as reference at 0V
            neg_ocp_relative = np.zeros_like(time_hours)  # Negative OCP as reference (0V)
            
            # Add invisible baseline for filling
            fig.add_trace(
                go.Scatter(
                    x=time_hours,
                    y=neg_ocp_relative,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row_index,
                col=col_index,
            )
            # Show negative OCP (reference at 0V)
            fig.add_trace(
                go.Scatter(
                    x=time_hours,
                    y=neg_ocp_relative,
                    fill="tonexty",
                    mode="lines",
                    line=dict(color="#2ca02c", width=1.2),
                    fillcolor="rgba(44,160,44,0.35)",
                    name="Negative open-circuit potential (0 V)",
                    legendgroup="neg_ocp",
                    showlegend=(row_index == 1 and col_index == 1),
                ),
                row=row_index,
                col=col_index,
            )
            # Use OCV as the baseline for stacking (same as combined plot)
            # OCV = positive OCP - negative OCP, which is what we stack from
            initial_ocv = ocv[0] if hasattr(ocv, '__iter__') and len(ocv) > 0 else ocv
            baseline = np.full_like(time_hours, initial_ocv, dtype=float)
            fig.add_trace(
                go.Scatter(x=time_hours, y=baseline, mode="lines",
                          line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False),
                row=row_index, col=col_index,
            )
            # Show positive OCP as OCV (top layer, same as combined plot)
            fig.add_trace(
                go.Scatter(
                    x=time_hours,
                    y=ocv,
                    fill="tonexty",
                    mode="lines",
                    line=dict(color="#9467bd", width=0.8),
                    fillcolor="rgba(148,103,189,0.35)",
                    name="Positive open-circuit potential",
                    legendgroup="pos_ocp",
                    showlegend=(row_index == 1 and col_index == 1),
                ),
                row=row_index,
                col=col_index,
            )
            top = ocv.copy()  # Start from OCV (same as combined plot)
            components = STACK_COMPONENTS_COMBINED
        
        typically_positive_vars = ["Lithium metal interface reaction overpotential [V]"]
        positive_components = []
        negative_components = []
        
        for idx, (var_name, display) in enumerate(components):
            try:
                values = solution[var_name].entries
            except KeyError:
                continue
            if "negative" in var_name.lower():
                values = -values
            if values.ndim > 1:
                values = np.mean(values, axis=0)
            
            if var_name in typically_positive_vars:
                positive_components.append((idx, var_name, display, values))
            else:
                negative_components.append((idx, var_name, display, values))
        
        positive_components.sort(key=lambda x: x[0])
        negative_components.sort(key=lambda x: x[0])
        
        ocv_top = ocv.copy()
        for _, var_name, display, values in positive_components:
            color = next(color_iter, "#17becf")
            new_top = ocv_top + values
            fig.add_trace(
                go.Scatter(x=time_hours, y=new_top, fill="tonexty", mode="lines",
                          line=dict(color=color, width=0.6),
                          fillcolor=hex_to_rgba(color, 0.35),
                          name=display, legendgroup=display,
                          showlegend=(row_index == 1 and col_index == 1),
                          hovertemplate=f"{display}: %{{customdata:.6f}} V<extra></extra>",
                          customdata=values),
                row=row_index, col=col_index,
            )
            ocv_top = new_top
        
        ocv_plus_positive_category = ocv.copy()
        for _, _, _, values in positive_components:
            ocv_plus_positive_category = ocv_plus_positive_category + values
        
        if negative_components:
            fig.add_trace(
                go.Scatter(x=time_hours, y=ocv_plus_positive_category,
                          mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
                          hoverinfo="skip", showlegend=False),
                row=row_index, col=col_index,
            )
        
        top = ocv_plus_positive_category.copy()
        for _, var_name, display, values in negative_components:
            color = next(color_iter, "#17becf")
            new_bottom = top + values
            fig.add_trace(
                go.Scatter(x=time_hours, y=new_bottom, fill="tonexty", mode="lines",
                          line=dict(color=color, width=0.6),
                          fillcolor=hex_to_rgba(color, 0.35),
                          name=display, legendgroup=display,
                          showlegend=(row_index == 1 and col_index == 1),
                          hovertemplate=f"{display}: %{{customdata:.6f}} V<extra></extra>",
                          customdata=values),
                row=row_index, col=col_index,
            )
            top = new_bottom
        
        # After stacking, 'top' should equal battery voltage
        # Calculate final voltage: OCV + all components (regardless of category)
        # Same calculation for both combined and split plots
        calculated_voltage = ocv.copy()
        for _, _, _, values in positive_components:
            calculated_voltage = calculated_voltage + values
        for _, _, _, values in negative_components:
            calculated_voltage = calculated_voltage + values
        # 'top' should equal calculated_voltage (they should be identical)
        
        # After stacking all overpotentials, 'top' (which equals calculated_voltage) should equal battery voltage
        # The battery voltage line should be at the bottom of the stack
        if voltage.ndim > 1:
            voltage_flat = np.mean(voltage, axis=0)
        else:
            voltage_flat = voltage
        
        # Calculate the difference to check for mismatches
        max_diff = np.max(np.abs(calculated_voltage - voltage_flat))
        mean_diff = np.mean(np.abs(calculated_voltage - voltage_flat))
        
        if max_diff > 1e-3:  # If difference > 1mV, print warning
            c_rate_label = f"row {row_index}, col {col_index}"
            print(f"Warning [{c_rate_label}]: Stack bottom doesn't match PyBaMM battery voltage.")
            print(f"  Calculated (OCV + overpotentials): {np.mean(calculated_voltage):.6f} V")
            print(f"  PyBaMM 'Battery voltage [V]': {np.mean(voltage_flat):.6f} V")
            print(f"  Max difference: {max_diff:.6f} V, Mean difference: {mean_diff:.6f} V")
            print(f"  This may indicate missing voltage components or sign convention issues.")
        
        # Plot both battery voltage lines at the bottom of the stack
        # 'top' after negative stacking equals OCV + positives + negatives = battery voltage
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=top,  # Bottom of negative stack = OCV + positives + negatives = battery voltage
                mode="lines",
                line=dict(color="black", dash="dash", width=1.5),
                name="Battery voltage (calculated)",
                legendgroup="Battery voltage calc",
                showlegend=(row_index == 1 and col_index == 1),
                hovertemplate="Battery voltage (calc): %{y:.6f} V<extra></extra>",
            ),
            row=row_index, col=col_index,
        )
        
        # Always plot PyBaMM's battery voltage for comparison
        fig.add_trace(
            go.Scatter(
                x=time_hours,
                y=voltage_flat,  # PyBaMM's actual battery voltage
                mode="lines",
                line=dict(color="red", dash="dot", width=1.5),
                name="Battery voltage (PyBaMM)",
                legendgroup="Battery voltage PyBaMM",
                showlegend=(row_index == 1 and col_index == 1),
                hovertemplate="Battery voltage (PyBaMM): %{y:.6f} V<extra></extra>",
            ),
            row=row_index, col=col_index,
        )
    
    # Build figure
    simulations = st.session_state.simulations
    num_c_rates = len(simulations)
    
    if num_c_rates > 0:
        subplot_titles = []
        for c_rate in sorted(simulations.keys()):
            subplot_titles.append(f"{c_rate}C Combined")
            subplot_titles.append(f"{c_rate}C Split")
        
        fig = make_subplots(
            rows=num_c_rates,
            cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
        )
        
        for row_idx, c_rate in enumerate(sorted(simulations.keys()), start=1):
            sim = simulations[c_rate]
            add_stacked_traces(fig, sim, row_idx, 1, split=False)
            add_stacked_traces(fig, sim, row_idx, 2, split=True)
        
        for row_idx in range(1, num_c_rates + 1):
            fig.update_xaxes(title_text="Time [h]", row=row_idx, col=1)
            fig.update_xaxes(title_text="Time [h]", row=row_idx, col=2)
            if row_idx == 1:
                fig.update_yaxes(title_text="Potential [V]", row=row_idx, col=1)
                fig.update_yaxes(title_text="Potential [V]", row=row_idx, col=2)
        
        fig.update_layout(
            title="Voltage Component Breakdown Across C-Rates",
            template="plotly_white",
            hovermode="x unified",
            height=400 * num_c_rates,
            width=1600,
            legend=dict(
                orientation="v",  # Vertical orientation
                yanchor="top",
                y=1.0,  # Position at the top of the plot area
                xanchor="left",
                x=1.02,  # Position to the right of the plot area
                font=dict(size=13, family="Arial, sans-serif"),  # Larger, clearer font
                bgcolor="rgba(255,255,255,0.95)",  # Semi-transparent white background
                bordercolor="lightgray",
                borderwidth=1.5,
                itemwidth=40,  # Width of color boxes for vertical legend
                itemsizing="constant",  # Consistent item sizing
                itemclick="toggleothers",  # Better interaction
                itemdoubleclick="toggle",  # Double-click to isolate
                tracegroupgap=20,  # More spacing between groups
            ),
            margin=dict(t=100, r=220, b=60, l=80)  # Extra right margin for vertical legend
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        st.download_button(
            label="📥 Download Plot as HTML",
            data=fig.to_html(include_plotlyjs="cdn"),
            file_name="overpotential_plot.html",
            mime="text/html"
        )

