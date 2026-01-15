import streamlit as st
from pathlib import Path
import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import ast
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon

# ========================================
# Page config
# ========================================
st.set_page_config(layout="wide", page_title="COO & ANN Results Viewer")

# ========================================
# Configuration
# ========================================
PUBLICATION_THEME = {
    'template': 'plotly_white',
    'font_family': 'Arial, sans-serif',
    'title_font_size': 22,
    'axis_font_size': 18,
    'legend_font_size': 16,
    'tick_font_size': 14,
    'line_width': 3,
    'marker_size': 10,
    'grid_color': 'rgba(200, 200, 200, 0.3)',
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
}

PUBLICATION_COLORS = {
    'primary': [
        '#0066CC',  # Strong Blue
        '#FF6600',  # Strong Orange
        '#00AA00',  # Strong Green
        '#CC0000',  # Strong Red
        '#9933CC',  # Strong Purple
        '#996633',  # Brown
        '#CC3399',  # Magenta
        '#666666',  # Dark Gray
        '#CCCC00',  # Yellow-Green
        '#009999',  # Teal
    ],
}

# ========================================
# Utility Functions
# ========================================


def parse_function_variant(name: str):
    """Parse function name into base function and variant"""
    s = name
    if "_ShiftedRotated" in s:
        return s.replace("_ShiftedRotated", ""), "ShiftedRotated"
    elif "_Rotated" in s:
        return s.replace("_Rotated", ""), "Rotated"
    elif "_Shifted" in s:
        return s.replace("_Shifted", ""), "Shifted"
    return s, "Base"


def extract_seed(filename: str):
    m = re.search(r"seed(\d+)", filename)
    return int(m.group(1)) if m else None


def extract_variant(name: str):
    """Extract variant from directory/file name"""
    s = name.lower()
    if "shiftedrotated" in s or ("shifted" in s and "rotated" in s):
        return "ShiftedRotated"
    if "rotated" in s:
        return "Rotated"
    if "shifted" in s:
        return "Shifted"
    return "Base"


def detect_base_dir():
    candidates = [p for p in Path(".").iterdir()
                  if p.is_dir() and p.name.startswith("Functions_")]
    return sorted(candidates, key=lambda x: x.name)[-1] if candidates else None


def detect_ann_file():
    """Detect ANN benchmark_master.csv file"""
    candidates = [p for p in Path(".").iterdir()
                  if p.is_dir() and p.name.startswith("ANN_")]
    if candidates:
        ann_dir = sorted(candidates, key=lambda x: x.name)[-1]
        csv_file = ann_dir / "benchmark_master.csv"
        if csv_file.exists():
            return csv_file
    return None


def parse_conv_hist(conv_hist_str, filter_outliers=True):
    """Parse convergence history string to list of floats

    Args:
        conv_hist_str: String or list containing convergence values
        filter_outliers: If True, removes extreme penalty values (default: True)

    Returns:
        List of valid convergence values
    """
    try:
        if isinstance(conv_hist_str, str):
            conv_hist_str = conv_hist_str.strip('[]')
            values = [float(x.strip()) for x in conv_hist_str.split(',')]
        elif isinstance(conv_hist_str, list):
            values = [float(x) for x in conv_hist_str]
        else:
            return []

        # Filter out invalid values (inf, nan)
        values = [v for v in values if np.isfinite(v)]

        if not values:
            return []

        # Filter out extreme outliers (penalty values)
        if filter_outliers and len(values) > 1:
            values = filter_extreme_outliers(values)

        return values
    except:
        return []


def filter_extreme_outliers(values, iqr_multiplier=10.0, max_ratio=1000.0):
    """
    Filter extreme outlier values that are likely penalty values.

    Uses two approaches:
    1. IQR-based filtering: Values beyond Q3 + iqr_multiplier * IQR
    2. Ratio-based filtering: Values > max_ratio times the median of reasonable values

    Works with absolute values to handle both positive and negative convergence data.

    Args:
        values: List of convergence values
        iqr_multiplier: Multiplier for IQR-based outlier detection (default: 10)
        max_ratio: Maximum allowed ratio to median (default: 1000)

    Returns:
        Filtered list with extreme outliers replaced by last valid value
    """
    if len(values) < 3:
        return values

    arr = np.array(values)
    # Work with absolute values for outlier detection
    abs_arr = np.abs(arr)

    # First pass: identify obvious penalty values (e.g., 1e12)
    q1 = np.percentile(abs_arr, 25)
    q3 = np.percentile(abs_arr, 75)
    iqr = q3 - q1

    # For convergence data, use a generous upper bound
    if iqr > 0:
        upper_bound = q3 + iqr_multiplier * iqr
    else:
        # If IQR is 0 (all values similar), use ratio-based detection
        median_val = np.median(abs_arr)
        upper_bound = median_val * max_ratio if median_val > 0 else 1e10

    # Also check for extreme ratios to the median of "reasonable" values
    reasonable_mask = abs_arr < upper_bound
    if np.sum(reasonable_mask) > 0:
        reasonable_median = np.median(abs_arr[reasonable_mask])
        if reasonable_median > 0:
            ratio_bound = reasonable_median * max_ratio
            upper_bound = min(upper_bound, ratio_bound)

    # Create mask for valid values (using absolute values for comparison)
    valid_mask = abs_arr <= upper_bound

    # If all values are "outliers" or no outliers found, return original
    if np.sum(valid_mask) == 0 or np.sum(valid_mask) == len(arr):
        return values

    # Replace outliers with the last valid value before them (forward fill logic)
    result = arr.copy()
    last_valid = arr[valid_mask][0]  # Start with first valid value

    for i in range(len(result)):
        if valid_mask[i]:
            last_valid = result[i]
        else:
            result[i] = last_valid

    return result.tolist()


def parse_best_pos(best_pos_str):
    """Parse best position string to list of floats"""
    try:
        if isinstance(best_pos_str, str):
            best_pos_str = best_pos_str.strip('[]')
            return [float(x.strip()) for x in best_pos_str.split(',')]
        elif isinstance(best_pos_str, list):
            return [float(x) for x in best_pos_str]
        else:
            return []
    except:
        return []


def parse_opt_params(opt_params_str):
    """Parse optimizer parameters from string to dictionary"""
    try:
        if isinstance(opt_params_str, str):
            return ast.literal_eval(opt_params_str)
        elif isinstance(opt_params_str, dict):
            return opt_params_str
        else:
            return {}
    except:
        return {}


def decode_vector_to_ann_params(x):
    """Decode 10-dimensional vector to ANN hyperparameters"""
    x = np.clip(np.asarray(x), 0.0, 1.0)

    h1 = int(round(8 + x[0] * (256 - 8)))
    h2 = int(round(8 + x[1] * (256 - 8)))
    alpha = 10 ** (-6 + x[2] * 5)
    lr = 10 ** (-5 + x[3] * 3.5)
    activation = "relu" if x[4] < 1 / \
        3 else ("tanh" if x[4] < 2/3 else "logistic")
    num_layers = 1 if x[5] < 1/3 else (2 if x[5] < 2/3 else 3)
    dropout = float(0.5 * x[6])
    batch_size = 16 if x[7] < 0.25 else (
        32 if x[7] < 0.5 else (64 if x[7] < 0.75 else 128))
    opt_name = "sgd" if x[8] < 1/3 else ("adam" if x[8] < 2/3 else "lbfgs")
    skip = bool(x[9] > 0.5)

    hidden_sizes = [h1, h2][:num_layers]

    return {
        "hidden_layer_1": h1,
        "hidden_layer_2": h2,
        "alpha (L2 reg)": f"{alpha:.6e}",
        "learning_rate": f"{lr:.6e}",
        "activation": activation,
        "num_layers": num_layers,
        "hidden_sizes": hidden_sizes,
        "dropout": f"{dropout:.4f}",
        "batch_size": batch_size,
        "optimizer": opt_name,
        "skip_connection": "Yes" if skip else "No"
    }

# ========================================
# Statistical Functions
# ========================================


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def interpret_effect_size(d):
    """Interpret Cohen's d"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def perform_wilcoxon_test(coo_data, other_data):
    """Perform Wilcoxon signed-rank test"""
    try:
        if len(coo_data) != len(other_data) or len(coo_data) == 0:
            return None, None
        diffs = np.array(coo_data) - np.array(other_data)
        if np.all(diffs == 0):
            return None, 1.0
        statistic, p_value = wilcoxon(coo_data, other_data, alternative='less')
        return statistic, p_value
    except:
        return None, None


def calculate_rankings(df, metric='best_value'):
    """Calculate average rankings across all functions/datasets"""
    rankings = []
    group_col = 'function' if 'function' in df.columns else 'dataset'

    for group in df[group_col].unique():
        group_data = df[df[group_col] == group]

        for seed in group_data['seed'].unique():
            seed_data = group_data[group_data['seed']
                                   == seed].sort_values(metric)
            seed_data['rank'] = range(1, len(seed_data) + 1)
            rankings.append(seed_data[['optimizer', 'rank']])

    all_rankings = pd.concat(rankings)
    avg_rankings = all_rankings.groupby(
        'optimizer')['rank'].mean().sort_values()

    return avg_rankings


def win_tie_loss_analysis(df, baseline='COO', metric='best_value'):
    """Calculate win/tie/loss counts for baseline vs others"""
    results = {}
    optimizers = [opt for opt in df['optimizer'].unique() if opt != baseline]
    group_col = 'function' if 'function' in df.columns else 'dataset'

    for opt in optimizers:
        wins, ties, losses = 0, 0, 0

        for group in df[group_col].unique():
            group_data = df[df[group_col] == group]

            for seed in group_data['seed'].unique():
                baseline_val = group_data[(group_data['optimizer'] == baseline) &
                                          (group_data['seed'] == seed)][metric].values
                other_val = group_data[(group_data['optimizer'] == opt) &
                                       (group_data['seed'] == seed)][metric].values

                if len(baseline_val) > 0 and len(other_val) > 0:
                    if baseline_val[0] < other_val[0]:
                        wins += 1
                    elif baseline_val[0] == other_val[0]:
                        ties += 1
                    else:
                        losses += 1

        total = wins + ties + losses
        results[opt] = {
            'Wins': wins,
            'Ties': ties,
            'Losses': losses,
            'Total': total,
            'Win %': f"{100*wins/total:.1f}%" if total > 0 else "0%"
        }

    return pd.DataFrame(results).T


# ========================================
# Styling Functions
# ========================================

def apply_publication_style(fig, title, xaxis_title, yaxis_title, yaxis_type='linear', trace_type='line'):
    """Apply publication-ready styling with bright white background"""
    fig.update_layout(
        template='plotly_white',
        title=dict(
            text=title,
            font=dict(size=PUBLICATION_THEME['title_font_size'],
                      family=PUBLICATION_THEME['font_family'], color='black'),
            x=0.5, xanchor='center'
        ),
        xaxis=dict(
            title=dict(text=xaxis_title,
                       font=dict(size=PUBLICATION_THEME['axis_font_size'],
                                 family=PUBLICATION_THEME['font_family'], color='black')),
            tickfont=dict(size=PUBLICATION_THEME['tick_font_size'],
                          family=PUBLICATION_THEME['font_family'], color='black'),
            gridcolor=PUBLICATION_THEME['grid_color'],
            showgrid=True, zeroline=True,
            zerolinecolor='rgba(0, 0, 0, 0.3)', zerolinewidth=1.5,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title,
                       font=dict(size=PUBLICATION_THEME['axis_font_size'],
                                 family=PUBLICATION_THEME['font_family'], color='black')),
            tickfont=dict(size=PUBLICATION_THEME['tick_font_size'],
                          family=PUBLICATION_THEME['font_family'], color='black'),
            type=yaxis_type,
            gridcolor=PUBLICATION_THEME['grid_color'],
            showgrid=True, zeroline=True,
            zerolinecolor='rgba(0, 0, 0, 0.3)', zerolinewidth=1.5,
        ),
        legend=dict(
            font=dict(size=PUBLICATION_THEME['legend_font_size'],
                      family=PUBLICATION_THEME['font_family'], color='black'),
            bgcolor='white',
            bordercolor='black',
            borderwidth=2,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family=PUBLICATION_THEME['font_family'],
            font_color="black",
            bordercolor="black"
        ),
        width=1400, height=800,
        margin=dict(l=80, r=40, t=80, b=80)
    )

    if trace_type == 'line':
        fig.update_traces(
            line=dict(width=PUBLICATION_THEME['line_width']),
            marker=dict(size=PUBLICATION_THEME['marker_size'])
        )
    elif trace_type == 'bar':
        fig.update_traces(
            marker=dict(line=dict(color='black', width=1))
        )

    return fig

# ========================================
# Data Loading Functions - FUNCTIONS
# ========================================


def discover_viewer_data(base_dir: Path):
    data = {}
    path_2d = base_dir / "2D"
    if not path_2d.exists():
        return data

    for func_dir in path_2d.iterdir():
        if not func_dir.is_dir():
            continue
        coo_dir = func_dir / "COO"
        if not coo_dir.exists():
            continue

        for variant_dir in coo_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            plotly_dir = variant_dir / "plotly"
            if not plotly_dir.exists():
                continue

            variant = extract_variant(variant_dir.name)

            for html in plotly_dir.glob("*.html"):
                seed = extract_seed(html.name)
                if seed is not None:
                    data.setdefault(func_dir.name, {})\
                        .setdefault(variant, {})[seed] = html
    return data


def load_convergence_csv(base_dir: Path):
    files = list(base_dir.glob("benchmark_master_*.csv"))
    if not files:
        return None

    try:
        df = pd.read_csv(files[0])
        if "conv_hist" not in df.columns:
            return None
        df = df[["optimizer", "function", "seed", "conv_hist"]]
        df = df[df["optimizer"].str.upper() == "COO"]

        parsed = df["function"].apply(parse_function_variant)
        df["base_function"] = parsed.map(lambda x: x[0])
        df["variant"] = parsed.map(lambda x: x[1])
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

        return df
    except Exception:
        return None


def discover_optuna_csvs(base_dir: Path):
    results = {}
    optuna_dir = base_dir / "optuna_tuning_results"
    if not optuna_dir.exists():
        return results

    for csv in optuna_dir.glob("*.csv"):
        tokens = csv.stem.split("_")
        if len(tokens) < 3 or tokens[0].lower() != "trials":
            continue

        optimizer = tokens[1].upper()
        base_function = tokens[2].capitalize()
        variant = extract_variant("_".join(tokens[3:]))

        results.setdefault(base_function, {})\
               .setdefault(variant, {})\
               .setdefault(optimizer, []).append(csv)
    return results


def load_benchmark_data(base_dir: Path):
    """Load benchmark data - tries multiple file patterns"""
    patterns = [
        "benchmark_master_final.csv",
        "benchmark_master_*.csv"
    ]

    for pattern in patterns:
        files = list(base_dir.glob(pattern))
        if files:
            try:
                df = pd.read_csv(files[0])
                required_cols = ["optimizer", "function", "seed", "best_value"]

                if not all(col in df.columns for col in required_cols):
                    continue

                if "time_sec" not in df.columns:
                    df["time_sec"] = 0

                parsed = df["function"].apply(parse_function_variant)
                df["base_function"] = parsed.map(lambda x: x[0])
                df["variant"] = parsed.map(lambda x: x[1])
                df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

                # Filter extreme values globally (overflow/underflow)
                # This removes rows where best_value or time_sec have extreme values
                original_len = len(df)
                df = df[df['best_value'].apply(
                    lambda x: np.isfinite(x) and abs(x) < 1e15)]
                df = df[df['time_sec'].apply(
                    lambda x: np.isfinite(x) and abs(x) < 1e15)]
                filtered_count = original_len - len(df)

                if filtered_count > 0:
                    st.sidebar.warning(
                        f"⚠️ Filtered {filtered_count} rows with extreme values (|value| > 1e15)")

                # Calculate Efficiency Score - DIRECT multiplication (same as ANN)
                # Formula: efficiency = best_value * time_sec
                # Lower is better (fast + accurate = low score)
                df['efficiency_score'] = df['best_value'] * df['time_sec']

                return df
            except Exception as e:
                st.error(f"Error loading {files[0].name}: {e}")
                continue

    return None


def show_html(path: Path):
    st.components.v1.html(
        path.read_text(encoding="utf-8"),
        height=1200,
        scrolling=False
    )


# ========================================
# Data Loading Functions - ANN
# ========================================

def load_ann_data(csv_file):
    """Load ANN benchmark data from CSV"""
    df = pd.read_csv(csv_file)

    # Add efficiency scores if not present
    if 'efficiency_mse_time' not in df.columns:
        df['efficiency_mse_time'] = df['mse'] * df['time_sec']
    if 'efficiency_mse_evals' not in df.columns:
        df['efficiency_mse_evals'] = df['mse'] * df['evals']

    return df


# ========================================
# ANN PAGE FUNCTIONS
# ========================================

def display_ann_convergence_page(df):
    """Display ANN convergence history and optimum parameters page"""
    st.title("🧠 ANN Hyperparameter Optimization - Convergence & Parameters")

    # Sidebar controls
    st.sidebar.header("🎛️ Convergence Controls")

    # Get unique values
    optimizers = sorted(df['optimizer'].unique())
    datasets = sorted(df['dataset'].unique())
    seeds = sorted(df['seed'].unique())

    # Add "All" options at the end
    optimizer_options = optimizers + ['All Optimizers']
    dataset_options = datasets + ['All Datasets']
    seed_options = [int(s) for s in seeds] + ['All Seeds']

    selected_optimizer = st.sidebar.selectbox(
        "Select Optimizer", optimizer_options)
    selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_options)
    selected_seed = st.sidebar.selectbox("Select Seed", seed_options)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 View Mode")
    view_mode = st.sidebar.radio(
        "Display Type",
        ["Convergence History", "Optimum Parameters", "Both"]
    )

    # Filter data
    filtered_df = df.copy()

    if selected_optimizer != 'All Optimizers':
        filtered_df = filtered_df[filtered_df['optimizer']
                                  == selected_optimizer]

    if selected_dataset != 'All Datasets':
        filtered_df = filtered_df[filtered_df['dataset'] == selected_dataset]

    if selected_seed != 'All Seeds':
        filtered_df = filtered_df[filtered_df['seed'] == selected_seed]

    if len(filtered_df) == 0:
        st.warning("⚠️ No data available for the selected filters.")
        return

    # Display convergence history
    if view_mode in ["Convergence History", "Both"]:
        st.markdown("---")
        st.subheader("📈 Convergence History")

        # Create dynamic title
        title_parts = []
        if selected_optimizer != 'All Optimizers':
            title_parts.append(f"Optimizer: {selected_optimizer}")
        if selected_dataset != 'All Datasets':
            title_parts.append(f"Dataset: {selected_dataset}")
        if selected_seed != 'All Seeds':
            title_parts.append(f"Seed: {selected_seed}")

        if title_parts:
            dynamic_title = " | ".join(title_parts)
        else:
            dynamic_title = "All Optimizers, All Datasets, All Seeds"

        fig = go.Figure()

        color_idx = 0
        has_data = False
        for _, row in filtered_df.iterrows():
            conv_hist = parse_conv_hist(row['conv_hist'])
            if len(conv_hist) > 0:
                has_data = True
                # Convert to positive MSE values (use abs for log scale compatibility)
                mse_values = [abs(x) for x in conv_hist]
                iterations = list(range(1, len(mse_values) + 1))

                label = f"{row['optimizer']}"
                if selected_dataset == 'All Datasets':
                    label += f" | {row['dataset']}"
                if selected_seed == 'All Seeds':
                    label += f" | Seed {row['seed']}"

                fig.add_trace(go.Scatter(
                    x=iterations,
                    y=mse_values,
                    mode='lines+markers',
                    name=label,
                    line=dict(
                        color=PUBLICATION_COLORS['primary'][color_idx % len(
                            PUBLICATION_COLORS['primary'])],
                        width=PUBLICATION_THEME['line_width']
                    ),
                    marker=dict(
                        size=PUBLICATION_THEME['marker_size'] - 2,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>Iteration: %{x}<br>MSE: %{y:.6f}<extra></extra>'
                ))

                color_idx += 1

        if has_data:
            fig = apply_publication_style(
                fig,
                title=f"MSE Convergence History<br><sub>{dynamic_title}</sub>",
                xaxis_title="Iteration",
                yaxis_title="Mean Squared Error (MSE)",
                yaxis_type='log'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Info note about data processing
            st.caption(
                "ℹ️ **Note:** Absolute values are used for plotting. Extreme outliers (penalty values) are automatically filtered for better visualization.")
        else:
            st.warning(
                "⚠️ No convergence history data available for the selected filters. The convergence history may be empty or contain invalid values.")

    # Display optimum parameters
    if view_mode in ["Optimum Parameters", "Both"]:
        st.markdown("---")
        st.subheader("🎯 Optimum Parameters")

        for idx, row in filtered_df.iterrows():
            # Create expander for each result
            title_text = f"{row['optimizer']} | {row['dataset']} | Seed {row['seed']} | MSE: {row['mse']:.6f}"

            with st.expander(title_text, expanded=(len(filtered_df) == 1)):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📊 Performance Metrics**")
                    metrics_data = {
                        'Metric': ['MSE', 'Time (seconds)', 'Evaluations', 'Status'],
                        'Value': [
                            f"{row['mse']:.6f}",
                            f"{row['time_sec']:.2f}",
                            f"{int(row['evals'])}",
                            row['status']
                        ]
                    }
                    metrics_df = pd.DataFrame(metrics_data)

                    styled_metrics = metrics_df.style.set_properties(**{
                        'text-align': 'left',
                        'font-size': '14px',
                        'padding': '8px',
                        'background-color': 'white',
                        'color': 'black'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [
                            ('font-size', '16px'),
                            ('text-align', 'center'),
                            ('font-weight', 'bold'),
                            ('background-color', '#e0e0e0'),
                            ('color', 'black'),
                            ('border', '2px solid black')
                        ]},
                        {'selector': 'td', 'props': [
                            ('border', '1px solid #ccc')
                        ]}
                    ])

                    st.dataframe(styled_metrics,
                                 use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("**🔧 Decoded ANN Hyperparameters**")
                    best_pos = parse_best_pos(row['best_pos'])
                    if len(best_pos) == 10:  # Must be 10-dimensional
                        # Decode the vector to actual hyperparameters
                        decoded_params = decode_vector_to_ann_params(best_pos)

                        params_data = {
                            'Hyperparameter': [
                                'Hidden Layer 1 Size',
                                'Hidden Layer 2 Size',
                                'Number of Layers',
                                'Hidden Sizes Used',
                                'Alpha (L2 Reg)',
                                'Learning Rate',
                                'Activation Function',
                                'Dropout Rate',
                                'Batch Size',
                                'Optimizer',
                                'Skip Connection'
                            ],
                            'Value': [
                                str(decoded_params['hidden_layer_1']),
                                str(decoded_params['hidden_layer_2']),
                                str(decoded_params['num_layers']),
                                str(decoded_params['hidden_sizes']),
                                decoded_params['alpha (L2 reg)'],
                                decoded_params['learning_rate'],
                                decoded_params['activation'],
                                decoded_params['dropout'],
                                str(decoded_params['batch_size']),
                                decoded_params['optimizer'],
                                decoded_params['skip_connection']
                            ]
                        }
                        params_df = pd.DataFrame(params_data)

                        styled_params = params_df.style.set_properties(**{
                            'text-align': 'left',
                            'font-size': '14px',
                            'padding': '8px',
                            'background-color': 'white',
                            'color': 'black'
                        }).set_table_styles([
                            {'selector': 'th', 'props': [
                                ('font-size', '16px'),
                                ('text-align', 'center'),
                                ('font-weight', 'bold'),
                                ('background-color', '#e0e0e0'),
                                ('color', 'black'),
                                ('border', '2px solid black')
                            ]},
                            {'selector': 'td', 'props': [
                                ('border', '1px solid #ccc')
                            ]}
                        ])

                        st.dataframe(
                            styled_params, use_container_width=True, hide_index=True)
                    else:
                        st.info(
                            f"Best position has {len(best_pos)} dimensions (expected 10). Cannot decode to hyperparameters.")
                        # Show raw values as fallback
                        if len(best_pos) > 0:
                            param_names = [
                                f"Parameter {i+1}" for i in range(len(best_pos))]
                            params_data = {
                                'Parameter': param_names,
                                'Value': [f"{val:.6f}" for val in best_pos]
                            }
                            params_df = pd.DataFrame(params_data)

                            styled_params = params_df.style.set_properties(**{
                                'text-align': 'left',
                                'font-size': '14px',
                                'padding': '8px',
                                'background-color': 'white',
                                'color': 'black'
                            }).set_table_styles([
                                {'selector': 'th', 'props': [
                                    ('font-size', '16px'),
                                    ('text-align', 'center'),
                                    ('font-weight', 'bold'),
                                    ('background-color', '#e0e0e0'),
                                    ('color', 'black'),
                                    ('border', '2px solid black')
                                ]},
                                {'selector': 'td', 'props': [
                                    ('border', '1px solid #ccc')
                                ]}
                            ])

                            st.dataframe(
                                styled_params, use_container_width=True, hide_index=True)

                # Display optimizer-specific parameters if available
                if pd.notna(row.get('opt_params', None)) and row['opt_params']:
                    st.markdown("**⚙️ Optimizer Configuration**")
                    opt_params = parse_opt_params(row['opt_params'])
                    if opt_params:
                        opt_params_list = []
                        for key, val in opt_params.items():
                            opt_params_list.append({
                                'Parameter': key,
                                'Value': str(val)
                            })
                        opt_params_df = pd.DataFrame(opt_params_list)

                        styled_opt = opt_params_df.style.set_properties(**{
                            'text-align': 'left',
                            'font-size': '14px',
                            'padding': '8px',
                            'background-color': 'white',
                            'color': 'black'
                        }).set_table_styles([
                            {'selector': 'th', 'props': [
                                ('font-size', '16px'),
                                ('text-align', 'center'),
                                ('font-weight', 'bold'),
                                ('background-color', '#e0e0e0'),
                                ('color', 'black'),
                                ('border', '2px solid black')
                            ]},
                            {'selector': 'td', 'props': [
                                ('border', '1px solid #ccc')
                            ]}
                        ])

                        st.dataframe(
                            styled_opt, use_container_width=True, hide_index=True)


def display_ann_statistical_analysis(df):
    """Display ANN statistical analysis page"""
    st.title("📊 ANN Hyperparameter Optimization - Statistical Analysis")

    st.sidebar.header("🎛️ Analysis Controls")

    # Metric selection
    metric = st.sidebar.selectbox(
        "Select Metric",
        ["mse", "time_sec", "evals", "efficiency_mse_time", "efficiency_mse_evals"]
    )

    # Metric labels
    metric_labels = {
        'mse': 'Mean Squared Error (MSE)',
        'time_sec': 'Execution Time (seconds)',
        'evals': 'Function Evaluations',
        'efficiency_mse_time': 'Efficiency Score (MSE × Time)',
        'efficiency_mse_evals': 'Efficiency Score (MSE × Evals)'
    }

    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Performance Summary", "Wilcoxon Test", "Effect Size Analysis",
         "Win/Tie/Loss Analysis", "Ranking Analysis"]
    )

    # Analysis scope
    analysis_scope = st.sidebar.radio(
        "Analysis Scope",
        ["All Datasets", "Specific Dataset"]
    )

    # Dataset selection for specific dataset
    if analysis_scope == "Specific Dataset":
        specific_dataset = st.sidebar.selectbox(
            "Select Dataset",
            sorted(df['dataset'].unique())
        )
        filtered_df = df[df['dataset'] == specific_dataset].copy()
        scope_title = f"Dataset: {specific_dataset}"
    else:
        filtered_df = df.copy()
        scope_title = "All Datasets"

    # Baseline optimizer
    baseline = st.sidebar.selectbox(
        "Baseline Optimizer",
        sorted(df['optimizer'].unique()),
        index=0
    )

    st.sidebar.markdown("---")

    # Display analysis
    st.subheader(f"📈 {analysis_type} - {metric_labels[metric]}")
    st.markdown(f"**Scope:** {scope_title}")

    if analysis_type == "Performance Summary":
        display_ann_performance_summary(
            filtered_df, metric, metric_labels[metric], scope_title)
    elif analysis_type == "Wilcoxon Test":
        display_ann_wilcoxon_test(
            filtered_df, metric, metric_labels[metric], baseline, scope_title)
    elif analysis_type == "Effect Size Analysis":
        display_ann_effect_size(filtered_df, metric,
                                metric_labels[metric], baseline, scope_title)
    elif analysis_type == "Win/Tie/Loss Analysis":
        display_ann_win_tie_loss(
            filtered_df, metric, metric_labels[metric], baseline, scope_title)
    elif analysis_type == "Ranking Analysis":
        display_ann_ranking(filtered_df, metric,
                            metric_labels[metric], scope_title)


def display_ann_performance_summary(df, metric, metric_label, scope_title):
    """Display performance summary statistics for ANN"""
    st.markdown(f"### Performance Summary: {metric_label} ({scope_title})")

    summary_data = []
    for optimizer in sorted(df['optimizer'].unique()):
        opt_data = df[df['optimizer'] == optimizer][metric]
        summary_data.append({
            'Optimizer': optimizer,
            'Mean': opt_data.mean(),
            'Median': opt_data.median(),
            'Std Dev': opt_data.std(),
            'Min': opt_data.min(),
            'Max': opt_data.max(),
            'Count': len(opt_data)
        })

    summary_df = pd.DataFrame(summary_data).sort_values('Mean')

    styled_summary = summary_df.style.format({
        'Mean': '{:.6f}',
        'Median': '{:.6f}',
        'Std Dev': '{:.6f}',
        'Min': '{:.6f}',
        'Max': '{:.6f}',
        'Count': '{:.0f}'
    }).set_properties(**{
        'text-align': 'center',
        'font-size': '16px',
        'padding': '12px',
        'background-color': 'white',
        'color': 'black'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('font-size', '18px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('background-color', '#e0e0e0'),
            ('color', 'black'),
            ('border', '2px solid black')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #ccc')
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#f5f5f5')
        ]}
    ])

    st.dataframe(styled_summary, use_container_width=True, hide_index=True)

    # Box plot
    fig = go.Figure()

    for idx, optimizer in enumerate(sorted(df['optimizer'].unique())):
        opt_data = df[df['optimizer'] == optimizer][metric]
        fig.add_trace(go.Box(
            y=opt_data,
            name=optimizer,
            marker=dict(
                color=PUBLICATION_COLORS['primary'][idx %
                                                    len(PUBLICATION_COLORS['primary'])],
                line=dict(color='black', width=2)
            ),
            boxmean='sd'
        ))

    fig = apply_publication_style(
        fig,
        title=f"Distribution of {metric_label}<br><sub>{scope_title}</sub>",
        xaxis_title="Optimizer",
        yaxis_title=metric_label,
        yaxis_type='log' if metric in [
            'mse', 'efficiency_mse_time', 'efficiency_mse_evals'] else 'linear',
        trace_type='box'
    )

    st.plotly_chart(fig, use_container_width=True)


def display_ann_wilcoxon_test(df, metric, metric_label, baseline, scope_title):
    """Display Wilcoxon test for ANN"""
    st.markdown(
        f"### Wilcoxon Signed-Rank Test: {baseline} vs Others ({scope_title})")

    st.markdown("""
    **Null Hypothesis (H₀):** The baseline and competitor have equal performance distributions.  
    **Alternative Hypothesis (H₁):** The baseline performs better (lower values).  
    **Significance Level:** α = 0.05
    """)

    competitors = [opt for opt in df['optimizer'].unique() if opt != baseline]
    results = []

    for competitor in competitors:
        baseline_vals = []
        competitor_vals = []

        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            for seed in dataset_data['seed'].unique():
                base_val = dataset_data[(dataset_data['optimizer'] == baseline) &
                                        (dataset_data['seed'] == seed)][metric].values
                comp_val = dataset_data[(dataset_data['optimizer'] == competitor) &
                                        (dataset_data['seed'] == seed)][metric].values

                if len(base_val) > 0 and len(comp_val) > 0:
                    baseline_vals.append(base_val[0])
                    competitor_vals.append(comp_val[0])

        if len(baseline_vals) > 0:
            statistic, p_value = perform_wilcoxon_test(
                baseline_vals, competitor_vals)
            d = cohens_d(baseline_vals, competitor_vals)

            results.append({
                'Competitor': competitor,
                'Statistic': statistic if statistic is not None else 'N/A',
                'p-value': p_value if p_value is not None else 1.0,
                'Significant?': '✅ Yes' if (p_value is not None and p_value < 0.05) else '❌ No',
                "Cohen's d": d,
                'Effect Size': interpret_effect_size(d)
            })

    results_df = pd.DataFrame(results).sort_values('p-value')

    def highlight_significant(val):
        if val == '✅ Yes':
            return 'background-color: #90EE90; font-weight: bold; color: black'
        else:
            return 'color: black'

    styled_df = results_df.style.format({
        'Statistic': lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else str(x),
        'p-value': lambda x: f'{x:.6f}' if pd.notna(x) and x < 1.0 else '1.000000',
        "Cohen's d": '{:.3f}'
    }).map(highlight_significant, subset=['Significant?']).set_properties(**{
        'text-align': 'center',
        'font-size': '16px',
        'padding': '12px',
        'background-color': 'white',
        'color': 'black'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('font-size', '18px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('background-color', '#e0e0e0'),
            ('color', 'black'),
            ('border', '2px solid black')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #ccc')
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#f5f5f5')
        ]}
    ])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    sig_count = sum(results_df['Significant?'] == '✅ Yes')
    total_count = len(results_df)

    if sig_count > 0:
        st.success(
            f"✅ **{baseline}** shows statistically significant improvement over **{sig_count}/{total_count}** algorithms (p < 0.05)")
    else:
        st.warning(
            f"⚠️ No statistically significant differences detected at α = 0.05")


def display_ann_effect_size(df, metric, metric_label, baseline, scope_title):
    """Display effect size analysis for ANN"""
    st.markdown(
        f"### Effect Size Analysis: {baseline} vs Others ({scope_title})")

    st.markdown("""
    **Cohen's d** measures the standardized difference between two means:
    - **|d| < 0.2**: Negligible effect
    - **0.2 ≤ |d| < 0.5**: Small effect
    - **0.5 ≤ |d| < 0.8**: Medium effect
    - **|d| ≥ 0.8**: Large effect
    
    **Negative d**: Baseline performs better  
    **Positive d**: Competitor performs better
    """)

    competitors = [opt for opt in df['optimizer'].unique() if opt != baseline]
    effect_sizes = []

    for competitor in competitors:
        baseline_vals = []
        competitor_vals = []

        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            for seed in dataset_data['seed'].unique():
                base_val = dataset_data[(dataset_data['optimizer'] == baseline) &
                                        (dataset_data['seed'] == seed)][metric].values
                comp_val = dataset_data[(dataset_data['optimizer'] == competitor) &
                                        (dataset_data['seed'] == seed)][metric].values

                if len(base_val) > 0 and len(comp_val) > 0:
                    baseline_vals.append(base_val[0])
                    competitor_vals.append(comp_val[0])

        if len(baseline_vals) > 0:
            d = cohens_d(baseline_vals, competitor_vals)
            interp = interpret_effect_size(d)

            effect_sizes.append({
                'Competitor': competitor,
                "Cohen's d": d,
                'Magnitude': abs(d),
                'Interpretation': interp,
                'Favors': baseline if d < 0 else competitor
            })

    effect_df = pd.DataFrame(effect_sizes).sort_values(
        'Magnitude', ascending=False)

    def color_favors(val):
        if val == baseline:
            return 'background-color: #90EE90; font-weight: bold; color: black'
        else:
            return 'background-color: #FFB6C6; color: black'

    styled_effect = effect_df.style.format({
        "Cohen's d": '{:.3f}',
        'Magnitude': '{:.3f}'
    }).map(color_favors, subset=['Favors']).set_properties(**{
        'text-align': 'center',
        'font-size': '16px',
        'padding': '12px',
        'background-color': 'white',
        'color': 'black'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('font-size', '18px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('background-color', '#e0e0e0'),
            ('color', 'black'),
            ('border', '2px solid black')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #ccc')
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#f5f5f5')
        ]}
    ])

    st.dataframe(styled_effect, use_container_width=True, hide_index=True)

    # Visualization
    fig = go.Figure()

    colors_map = {baseline: '#00AA00', 'Competitor': '#CC0000'}
    bar_colors = [colors_map[baseline] if fav == baseline else colors_map['Competitor']
                  for fav in effect_df['Favors']]

    fig.add_trace(go.Bar(
        x=effect_df['Competitor'],
        y=effect_df["Cohen's d"],
        marker=dict(color=bar_colors, line=dict(color='black', width=2)),
        text=effect_df["Cohen's d"].round(2),
        textposition='outside',
        textfont=dict(size=16, color='black'),
        customdata=effect_df['Interpretation'],
        hovertemplate='<b>%{x}</b><br>Cohen\'s d: %{y:.3f}<br>Effect: %{customdata}<extra></extra>'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=2)
    fig.add_hline(y=-0.8, line_dash="dot", line_color="green", line_width=2)
    fig.add_hline(y=0.8, line_dash="dot", line_color="red", line_width=2)

    fig = apply_publication_style(
        fig,
        title=f"Effect Sizes: {baseline} vs Competitors<br><sub>{scope_title}</sub>",
        xaxis_title="Competitor",
        yaxis_title="Cohen's d",
        trace_type='bar'
    )

    fig.update_layout(margin=dict(l=80, r=200, t=100, b=80))

    st.plotly_chart(fig, use_container_width=True)

    large_effects = len(effect_df[(effect_df['Interpretation'] == 'Large') & (
        effect_df["Cohen's d"] < 0)])

    if large_effects > 0:
        st.success(
            f"✅ **{baseline}** shows **large practical significance** over **{large_effects}** algorithm(s)")


def display_ann_win_tie_loss(df, metric, metric_label, baseline, scope_title):
    """Display win/tie/loss analysis for ANN"""
    st.markdown(
        f"### Win/Tie/Loss Analysis: {baseline} vs Others ({scope_title})")

    wtl_df = win_tie_loss_analysis(df, baseline=baseline, metric=metric)

    styled_wtl = wtl_df.style.set_properties(**{
        'text-align': 'center',
        'font-size': '16px',
        'padding': '12px',
        'background-color': 'white',
        'color': 'black'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('font-size', '18px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('background-color', '#e0e0e0'),
            ('color', 'black'),
            ('border', '2px solid black')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #ccc')
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#f5f5f5')
        ]}
    ])

    st.dataframe(styled_wtl, use_container_width=True)

    # Visualization
    fig = go.Figure()

    competitors = wtl_df.index.tolist()
    wins = wtl_df['Wins'].values
    ties = wtl_df['Ties'].values
    losses = wtl_df['Losses'].values

    fig.add_trace(go.Bar(name='Wins', x=competitors, y=wins,
                         marker=dict(color='#00AA00', line=dict(
                             color='black', width=2)),
                         text=wins, textposition='auto'))
    fig.add_trace(go.Bar(name='Ties', x=competitors, y=ties,
                         marker=dict(color='#CCCC00', line=dict(
                             color='black', width=2)),
                         text=ties, textposition='auto'))
    fig.add_trace(go.Bar(name='Losses', x=competitors, y=losses,
                         marker=dict(color='#CC0000', line=dict(
                             color='black', width=2)),
                         text=losses, textposition='auto'))

    fig = apply_publication_style(
        fig,
        title=f"Win/Tie/Loss: {baseline} vs Competitors<br><sub>{scope_title}</sub>",
        xaxis_title="Competitor",
        yaxis_title="Count",
        trace_type='bar'
    )

    fig.update_layout(barmode='stack')

    st.plotly_chart(fig, use_container_width=True)


def display_ann_ranking(df, metric, metric_label, scope_title):
    """Display ranking analysis for ANN"""
    st.markdown(f"### Average Ranking Analysis ({scope_title})")

    st.markdown("""
    Rankings are computed for each dataset-seed combination, then averaged.  
    **Lower rank = Better performance**
    """)

    rankings = calculate_rankings(df, metric=metric)

    ranking_df = pd.DataFrame({
        'Optimizer': rankings.index,
        'Average Rank': rankings.values,
        'Rank Position': range(1, len(rankings) + 1)
    })

    styled_ranking = ranking_df.style.format({
        'Average Rank': '{:.2f}'
    }).set_properties(**{
        'text-align': 'center',
        'font-size': '16px',
        'padding': '12px',
        'background-color': 'white',
        'color': 'black'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('font-size', '18px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('background-color', '#e0e0e0'),
            ('color', 'black'),
            ('border', '2px solid black')
        ]},
        {'selector': 'td', 'props': [
            ('border', '1px solid #ccc')
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#f5f5f5')
        ]}
    ])

    st.dataframe(styled_ranking, use_container_width=True, hide_index=True)

    # Visualization
    fig = go.Figure()

    colors = [PUBLICATION_COLORS['primary'][i % len(PUBLICATION_COLORS['primary'])]
              for i in range(len(ranking_df))]

    fig.add_trace(go.Bar(
        x=ranking_df['Optimizer'],
        y=ranking_df['Average Rank'],
        marker=dict(color=colors, line=dict(color='black', width=2)),
        text=ranking_df['Average Rank'].round(2),
        textposition='outside',
        textfont=dict(size=16, color='black')
    ))

    fig = apply_publication_style(
        fig,
        title=f"Average Rankings by {metric_label}<br><sub>{scope_title}</sub>",
        xaxis_title="Optimizer",
        yaxis_title="Average Rank (Lower is Better)",
        trace_type='bar'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"🏆 **{ranking_df.iloc[0]['Optimizer']}** achieves the best average ranking!")


# ========================================
# Main App Navigation
# ========================================

st.title("COO & ANN Results Viewer")

# Sidebar navigation
st.sidebar.title("🎛️ Navigation")
st.sidebar.markdown("### Select Section")
app_section = st.sidebar.radio(
    "Section",
    ["🔧 Functions", "🧠 ANN"]
)

# ========================================
# ANN SECTION
# ========================================
if app_section == "🧠 ANN":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ANN Pages")
    ann_page = st.sidebar.radio(
        "Page",
        ["Convergence & Parameters", "Statistical Analysis"]
    )

    # Try to load ANN data
    ann_file = detect_ann_file()

    if ann_file is None:
        st.error("⚠️ ANN benchmark_master.csv file not found!")
        st.info(
            "Please ensure there is a folder starting with 'ANN_' containing 'benchmark_master.csv'")
    else:
        try:
            ann_df = load_ann_data(ann_file)
            st.sidebar.success(f"✅ Loaded: {len(ann_df)} records")

            if ann_page == "Convergence & Parameters":
                display_ann_convergence_page(ann_df)
            elif ann_page == "Statistical Analysis":
                display_ann_statistical_analysis(ann_df)
        except Exception as e:
            st.error(f"Error loading ANN data: {str(e)}")
            st.exception(e)

# ========================================
# FUNCTIONS SECTION (Original Code)
# ========================================
else:
    # Detect base directory
    BASE_DIR = detect_base_dir()

    if BASE_DIR is None:
        st.error("❌ No Functions_* folder found in current directory.")
        st.info("""
        **Expected structure:**
        ```
        Functions_YYYYMMDD_HHMMSS/
        ├── 2D/
        ├── optuna_tuning_results/
        ├── benchmark_master_final.csv (or benchmark_master_*.csv)
        ```
        """)
        st.stop()

    st.sidebar.success(f"✅ Using: `{BASE_DIR.name}`")

    # Load data
    with st.spinner("Loading data..."):
        viewer_data = discover_viewer_data(BASE_DIR)
        conv_df = load_convergence_csv(BASE_DIR)
        optuna_data = discover_optuna_csvs(BASE_DIR)
        benchmark_df = load_benchmark_data(BASE_DIR)

    # Check data availability
    data_status = {
        "3D Viewer": len(viewer_data) > 0,
        "Convergence": conv_df is not None and len(conv_df) > 0,
        "Optuna": len(optuna_data) > 0,
        "Benchmark": benchmark_df is not None and len(benchmark_df) > 0,
    }

    if not any(data_status.values()):
        st.error("❌ No data found!")
        st.write("**Data Status:**")
        for key, status in data_status.items():
            st.write(f"- {key}: {'✅' if status else '❌'}")
        st.stop()

    with st.sidebar.expander("📊 Data Status"):
        for key, status in data_status.items():
            st.write(f"{'✅' if status else '❌'} {key}")

    # Determine available pages
    available_pages = []
    if data_status["3D Viewer"] or data_status["Convergence"]:
        available_pages.append("Function Viewer")
    if data_status["Optuna"]:
        available_pages.append("Optuna Tuning")
    if data_status["Benchmark"]:
        available_pages.append("Benchmark")
        available_pages.append("Statistical Analysis")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Function Pages")
    page = st.sidebar.radio("Page", available_pages, index=0)

    # ========================================
    # PAGE 1 — FUNCTION VIEWER
    # ========================================
    if page == "Function Viewer":
        st.header("📈 Function Viewer")

        available_modes = []
        if data_status["Convergence"]:
            available_modes.append("Convergence History")
        if data_status["3D Viewer"]:
            available_modes.append("3D Plot")

        if not available_modes:
            st.error("No data available")
            st.stop()

        if conv_df is not None and len(conv_df) > 0:
            available_functions = sorted(conv_df["base_function"].unique())
        elif viewer_data:
            available_functions = sorted(viewer_data.keys())
        else:
            st.error("No data")
            st.stop()

        fn = st.sidebar.selectbox("Function", available_functions)

        if conv_df is not None and len(conv_df) > 0:
            available_variants = sorted(
                conv_df[conv_df["base_function"] == fn]["variant"].unique()
            )
        elif fn in viewer_data:
            available_variants = sorted(viewer_data[fn].keys())
        else:
            available_variants = []

        if not available_variants:
            st.error(f"No variants for {fn}")
            st.stop()

        var = st.sidebar.selectbox("Variant", available_variants)
        view_mode = st.sidebar.radio("View Mode", available_modes)
        y_scale = st.sidebar.radio("Y-axis Scale", ["Linear", "Log"])

        if view_mode == "Convergence History":
            if conv_df is None or len(conv_df) == 0:
                st.error("No convergence data")
                st.stop()

            conv_mode = st.sidebar.radio(
                "Convergence Mode", ["Single Seed", "All Seeds"])

            df_sel = conv_df[
                (conv_df["base_function"] == fn) &
                (conv_df["variant"] == var)
            ]

            if len(df_sel) == 0:
                st.warning(f"No data for {fn} ({var})")
                st.stop()

            available_seeds = sorted(df_sel["seed"].unique())
            seed = st.sidebar.selectbox("Seed", available_seeds)

            fig = go.Figure()
            colors = PUBLICATION_COLORS['primary']

            if conv_mode == "Single Seed":
                row = df_sel[df_sel["seed"] == seed]
                raw_conv = ast.literal_eval(row.iloc[0]["conv_hist"])
                # Use absolute values for log scale compatibility
                conv = [abs(x) for x in raw_conv]

                fig.add_trace(go.Scatter(
                    x=list(range(len(conv))),
                    y=conv,
                    mode="lines+markers",
                    name=f"Seed {seed}",
                    line=dict(color=colors[0],
                              width=PUBLICATION_THEME['line_width']),
                    marker=dict(
                        size=PUBLICATION_THEME['marker_size'], color=colors[0]),
                    hovertemplate='<b>Iteration: %{x}</b><br>Value: %{y:.6e}<extra></extra>'
                ))

                title = f"COO Convergence — {fn} ({var}) | Seed {seed}"

            else:
                all_hists = {}
                min_len = np.inf

                for _, r in df_sel.iterrows():
                    raw_h = ast.literal_eval(r["conv_hist"])
                    # Use absolute values for log scale compatibility
                    h = [abs(x) for x in raw_h]
                    all_hists[int(r["seed"])] = h
                    min_len = min(min_len, len(h))

                seeds = sorted(all_hists)
                values = np.array([all_hists[s][:min_len] for s in seeds])
                mean = values.mean(axis=0)
                std = values.std(axis=0)

                final_vals = {s: all_hists[s][min_len - 1] for s in seeds}
                best_seed = min(final_vals, key=final_vals.get)

                for i, s in enumerate(seeds):
                    is_best = (s == best_seed)
                    fig.add_trace(go.Scatter(
                        x=list(range(min_len)),
                        y=all_hists[s][:min_len],
                        mode="lines",
                        name=f"Seed {s}" + (" ⭐" if is_best else ""),
                        line=dict(color=colors[i % len(colors)],
                                  width=4 if is_best else 2),
                        opacity=1.0 if is_best else 0.5,
                        hovertemplate=f'<b>Seed {s}</b><br>Iteration: %{{x}}<br>Value: %{{y:.6e}}<extra></extra>'
                    ))

                fig.add_trace(go.Scatter(
                    x=list(range(min_len)),
                    y=mean,
                    mode="lines",
                    name="Mean",
                    line=dict(color='black', width=4, dash='dash'),
                    hovertemplate='<b>Mean</b><br>Iteration: %{x}<br>Value: %{y:.6e}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=list(range(min_len)),
                    y=mean + std,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(min_len)),
                    y=mean - std,
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.2)",
                    line=dict(width=0),
                    name="±1 Std",
                    hoverinfo='skip'
                ))

                fig.add_trace(go.Scatter(
                    x=[min_len - 1],
                    y=[final_vals[best_seed]],
                    mode="markers",
                    name=f"Best (Seed {best_seed})",
                    marker=dict(size=16, symbol="star", color="gold",
                                line=dict(color='black', width=2)),
                    hovertemplate=f'<b>Best Final</b><br>Seed: {best_seed}<br>Value: {final_vals[best_seed]:.6e}<extra></extra>'
                ))

                title = f"COO Convergence — {fn} ({var}) | All Seeds (n={len(seeds)})"

            fig = apply_publication_style(
                fig, title=title, xaxis_title="Iteration",
                yaxis_title="Objective Value",
                yaxis_type="log" if y_scale == "Log" else "linear",
                trace_type='line'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Info note about data processing
            st.caption(
                "ℹ️ **Note:** Absolute values are used for plotting to ensure log scale compatibility.")

        else:  # 3D Plot
            if not viewer_data:
                st.error("No 3D visualization data available")
                st.stop()

            if fn not in viewer_data:
                st.error(f"No 3D data for function: {fn}")
                st.stop()

            if var not in viewer_data[fn]:
                st.error(f"No 3D data for {fn} ({var})")
                st.stop()

            available_seeds_3d = sorted(viewer_data[fn][var].keys())
            seed = st.sidebar.selectbox("Seed", available_seeds_3d)

            st.subheader(f"🌄 {fn} ({var}) | Seed {seed}")
            show_html(viewer_data[fn][var][seed])

    # ========================================
    # PAGE 2 — OPTUNA
    # ========================================
    elif page == "Optuna Tuning":
        st.header("🎯 Optuna Tuning Results")

        if not optuna_data:
            st.error("No Optuna data")
            st.stop()

        fn = st.sidebar.selectbox("Function", sorted(optuna_data.keys()))
        var = st.sidebar.selectbox("Variant", sorted(optuna_data[fn].keys()))

        csv_map = optuna_data[fn][var]
        all_opts = sorted(csv_map.keys())

        mode = st.sidebar.radio("Selection", ["ALL", "Custom"])
        opts = all_opts if mode == "ALL" else st.sidebar.multiselect(
            "Optimizers", all_opts, default=all_opts[:2] if len(all_opts) >= 2 else all_opts
        )

        if not opts:
            st.warning("⚠️ Select at least one optimizer")
            st.stop()

        y_scale = st.sidebar.radio("Y-axis Scale", ["Linear", "Log"])

        fig = go.Figure()
        colors = PUBLICATION_COLORS['primary']

        for i, opt in enumerate(opts):
            show_legend = True
            for csv in csv_map[opt]:
                df = pd.read_csv(csv)
                fig.add_trace(go.Scatter(
                    x=df.iloc[:, 0],
                    y=df.iloc[:, 1],
                    mode="lines+markers",
                    name=opt,
                    legendgroup=opt,
                    showlegend=show_legend,
                    line=dict(color=colors[i % len(colors)],
                              width=PUBLICATION_THEME['line_width']),
                    marker=dict(size=PUBLICATION_THEME['marker_size'],
                                color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{opt}</b><br>Trial: %{{x}}<br>Value: %{{y:.6e}}<extra></extra>'
                ))
                show_legend = False

        title = f"Optuna HPO — {fn} ({var})"

        fig = apply_publication_style(
            fig, title=title, xaxis_title="Trial",
            yaxis_title="Objective Value",
            yaxis_type="log" if y_scale == "Log" else "linear",
            trace_type='line'
        )

        st.plotly_chart(fig, use_container_width=True)

    # ========================================
    # PAGE 3 — BENCHMARK
    # ========================================
    elif page == "Benchmark":
        st.header("📊 Benchmark Results")

        if benchmark_df is None or len(benchmark_df) == 0:
            st.error("No benchmark data")
            st.stop()

        fn = st.sidebar.selectbox(
            "Function", sorted(benchmark_df["base_function"].unique())
        )
        var = st.sidebar.selectbox(
            "Variant",
            sorted(benchmark_df[benchmark_df["base_function"]
                   == fn]["variant"].unique())
        )

        df = benchmark_df[
            (benchmark_df["base_function"] == fn) &
            (benchmark_df["variant"] == var)
        ]

        mode = st.sidebar.radio("Selection", ["ALL", "Custom"])
        all_opts = sorted(df["optimizer"].unique())
        opts = all_opts if mode == "ALL" else st.sidebar.multiselect(
            "Optimizers", all_opts, default=all_opts[:3] if len(all_opts) >= 3 else all_opts
        )

        if not opts:
            st.warning("⚠️ Select at least one optimizer")
            st.stop()

        metric = st.sidebar.radio(
            "Metric", ["best_value", "time_sec", "efficiency_score"])

        metric_display = {
            'best_value': 'Best Value',
            'time_sec': 'Time (seconds)',
            'efficiency_score': 'Efficiency Score'
        }

        if metric == "efficiency_score":
            st.sidebar.info("""
            **Efficiency Score** combines:
            - Solution quality (best_value)
            - Computational time (time_sec)
            
            **Formula:** best_value × time_sec
            
            **Lower score = Better**
            
            Rewards algorithms that find good solutions quickly!
            """)

        y_scale = st.sidebar.radio("Y-axis Scale", ["Linear", "Log"])

        df_filtered = df[df["optimizer"].isin(opts)].sort_values("seed")

        # Line Plot
        st.subheader(f"📈 {metric_display[metric]} vs Seed")

        fig_line = go.Figure()
        colors = PUBLICATION_COLORS['primary']

        for i, opt in enumerate(opts):
            dfo = df_filtered[df_filtered["optimizer"] == opt]
            fig_line.add_trace(go.Scatter(
                x=dfo["seed"],
                y=dfo[metric],
                mode="lines+markers",
                name=opt,
                line=dict(color=colors[i % len(colors)],
                          width=PUBLICATION_THEME['line_width']),
                marker=dict(size=PUBLICATION_THEME['marker_size'],
                            color=colors[i % len(colors)]),
                hovertemplate=f'<b>{opt}</b><br>Seed: %{{x}}<br>{metric_display[metric]}: %{{y:.6e}}<extra></extra>'
            ))

        fig_line = apply_publication_style(
            fig_line,
            title=f"{metric_display[metric]} — {fn} ({var})",
            xaxis_title="Seed",
            yaxis_title=metric_display[metric],
            yaxis_type="log" if y_scale == "Log" else "linear",
            trace_type='line'
        )

        st.plotly_chart(fig_line, use_container_width=True)

        # Box Plot
        st.subheader(f"📦 {metric_display[metric]} Distribution")

        fig_box = go.Figure()
        colors_pub = PUBLICATION_COLORS['primary']

        for i, opt in enumerate(opts):
            dfo = df_filtered[df_filtered["optimizer"] == opt]
            fig_box.add_trace(go.Box(
                y=dfo[metric],
                name=opt,
                boxpoints="all",
                jitter=0.4,
                pointpos=0.0,
                marker=dict(color=colors_pub[i % len(colors_pub)], size=8, opacity=0.7,
                            line=dict(color='black', width=1)),
                line=dict(color=colors_pub[i % len(colors_pub)], width=2),
                fillcolor=colors_pub[i % len(colors_pub)],
                opacity=0.7,
                hovertemplate=f'<b>{opt}</b><br>{metric_display[metric]}: %{{y:.6e}}<extra></extra>'
            ))

        fig_box = apply_publication_style(
            fig_box,
            title=f"{metric_display[metric]} Distribution — {fn} ({var})",
            xaxis_title="Optimizer",
            yaxis_title=metric_display[metric],
            yaxis_type="log" if y_scale == "Log" else "linear",
            trace_type='box'
        )

        st.plotly_chart(fig_box, use_container_width=True)

        # Summary Statistics
        st.subheader("📋 Summary Statistics")

        summary_data = []
        for opt in opts:
            dfo = df_filtered[df_filtered["optimizer"] == opt]
            stats = {
                'Optimizer': opt,
                'Mean': dfo[metric].mean(),
                'Std': dfo[metric].std(),
                'Median': dfo[metric].median(),
                'Min': dfo[metric].min(),
                'Max': dfo[metric].max(),
                'Count': len(dfo)
            }
            summary_data.append(stats)

        summary_df = pd.DataFrame(summary_data)

        styled_summary = summary_df.style.format({
            'Mean': '{:.6e}',
            'Std': '{:.6e}',
            'Median': '{:.6e}',
            'Min': '{:.6e}',
            'Max': '{:.6e}',
            'Count': '{:d}'
        }).set_properties(**{
            'text-align': 'center',
            'font-size': '16px',
            'padding': '12px',
            'background-color': 'white',
            'color': 'black'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('font-size', '18px'),
                ('text-align', 'center'),
                ('font-weight', 'bold'),
                ('background-color', '#e0e0e0'),
                ('color', 'black'),
                ('border', '2px solid black')
            ]},
            {'selector': 'td', 'props': [
                ('border', '1px solid #ccc'),
                ('background-color', 'white'),
                ('color', 'black')
            ]},
            {'selector': 'tr:hover', 'props': [
                ('background-color', '#f5f5f5')
            ]}
        ])

        st.dataframe(styled_summary, use_container_width=True, hide_index=True)

    # ========================================
    # PAGE 4 — STATISTICAL ANALYSIS
    # ========================================
    elif page == "Statistical Analysis":
        st.header("📊 Statistical Analysis")

        if benchmark_df is None or len(benchmark_df) == 0:
            st.error("No benchmark data")
            st.stop()

        st.markdown("""
        This page performs rigorous statistical tests to validate COO's performance against baseline algorithms.
        """)

        baseline = st.sidebar.selectbox("Baseline Algorithm",
                                        sorted(
                                            benchmark_df["optimizer"].unique()),
                                        index=list(sorted(benchmark_df["optimizer"].unique())).index('COO') if 'COO' in benchmark_df["optimizer"].unique() else 0)

        metric = st.sidebar.radio(
            "Metric", ["best_value", "time_sec", "efficiency_score"])

        if metric == "efficiency_score":
            st.sidebar.info("""
            **Efficiency Score** = best_value × time_sec
            
            - Rewards fast + accurate solutions
            - Lower score = Better performance
            - Direct multiplication (consistent with ANN)
            """)

        analysis_scope = st.sidebar.radio("Analysis Scope", [
            "All Functions (Overall)",
            "Specific Function + Variant",
            "Specific Function (All Variants)"
        ])

        if analysis_scope == "Specific Function + Variant":
            fn = st.sidebar.selectbox("Function", sorted(
                benchmark_df["base_function"].unique()))
            var = st.sidebar.selectbox("Variant", sorted(
                benchmark_df[benchmark_df["base_function"] == fn]["variant"].unique()))
            filtered_df = benchmark_df[
                (benchmark_df["base_function"] == fn) &
                (benchmark_df["variant"] == var)
            ]
            scope_title = f"{fn} ({var})"
        elif analysis_scope == "Specific Function (All Variants)":
            fn = st.sidebar.selectbox("Function", sorted(
                benchmark_df["base_function"].unique()))
            filtered_df = benchmark_df[benchmark_df["base_function"] == fn]
            scope_title = f"{fn} (All Variants)"
        else:
            filtered_df = benchmark_df
            scope_title = "All Functions"

        analysis_type = st.sidebar.radio("Analysis Type", [
            "Overall Ranking",
            "Win/Tie/Loss Analysis",
            "Pairwise Comparison",
            "Effect Size Analysis"
        ])

        # Overall Ranking
        if analysis_type == "Overall Ranking":
            st.subheader(f"🏆 Average Rankings: {scope_title}")

            rankings = calculate_rankings(filtered_df, metric)

            st.markdown("**Lower rank is better (1 = best)**")

            rank_df = pd.DataFrame({
                'Optimizer': rankings.index,
                'Average Rank': rankings.values,
                'Rank Position': range(1, len(rankings) + 1)
            })

            styled_rank = rank_df.style.format({
                'Average Rank': '{:.2f}'
            }).set_properties(**{
                'text-align': 'center',
                'font-size': '16px',
                'padding': '12px',
                'background-color': 'white',
                'color': 'black'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('font-size', '18px'),
                    ('text-align', 'center'),
                    ('font-weight', 'bold'),
                    ('background-color', '#e0e0e0'),
                    ('color', 'black'),
                    ('border', '2px solid black')
                ]},
                {'selector': 'td', 'props': [
                    ('border', '1px solid #ccc'),
                    ('background-color', 'white'),
                    ('color', 'black')
                ]},
                {'selector': 'tr:hover', 'props': [
                    ('background-color', '#f5f5f5')
                ]}
            ])

            st.dataframe(styled_rank, use_container_width=True,
                         hide_index=True)

            # Visualization
            fig = go.Figure()
            colors = PUBLICATION_COLORS['primary']

            fig.add_trace(go.Bar(
                x=rankings.index,
                y=rankings.values,
                marker=dict(
                    color=[colors[i % len(colors)]
                           for i in range(len(rankings))],
                    line=dict(color='black', width=1)
                ),
                text=rankings.values.round(2),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg Rank: %{y:.2f}<extra></extra>'
            ))

            fig = apply_publication_style(
                fig,
                title=f"Average Rankings: {scope_title}",
                xaxis_title="Optimizer",
                yaxis_title="Average Rank",
                yaxis_type='linear',
                trace_type='bar'
            )

            st.plotly_chart(fig, use_container_width=True)

            best_optimizer = rankings.index[0]
            st.success(
                f"✅ **{best_optimizer}** achieves the best average rank of {rankings.iloc[0]:.2f}")

            if baseline == best_optimizer:
                st.info(f"🎉 **{baseline}** outranks all other algorithms!")

        # Win/Tie/Loss Analysis
        elif analysis_type == "Win/Tie/Loss Analysis":
            st.subheader(
                f"⚔️ Win/Tie/Loss: {baseline} vs Others ({scope_title})")

            wtl_results = win_tie_loss_analysis(filtered_df, baseline, metric)

            styled_wtl = wtl_results.style.set_properties(**{
                'text-align': 'center',
                'font-size': '16px',
                'padding': '12px',
                'background-color': 'white',
                'color': 'black'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('font-size', '18px'),
                    ('text-align', 'center'),
                    ('font-weight', 'bold'),
                    ('background-color', '#e0e0e0'),
                    ('color', 'black'),
                    ('border', '2px solid black')
                ]},
                {'selector': 'td', 'props': [
                    ('border', '1px solid #ccc'),
                    ('background-color', 'white'),
                    ('color', 'black')
                ]},
                {'selector': 'tr:hover', 'props': [
                    ('background-color', '#f5f5f5')
                ]}
            ])

            st.dataframe(styled_wtl, use_container_width=True)

            # Visualization
            fig = go.Figure()
            colors_wtl = ['#00AA00', '#CCCC00', '#CC0000']

            for idx, stat in enumerate(['Wins', 'Ties', 'Losses']):
                fig.add_trace(go.Bar(
                    name=stat,
                    x=wtl_results.index,
                    y=wtl_results[stat],
                    text=wtl_results[stat],
                    textposition='auto',
                    textfont=dict(size=16, color='black'),
                    marker=dict(color=colors_wtl[idx], line=dict(
                        color='black', width=2)),
                    hovertemplate=f'<b>%{{x}}</b><br>{stat}: %{{y}}<extra></extra>'
                ))

            fig.update_layout(
                title=dict(
                    text=f"{baseline} Performance vs Other Algorithms ({scope_title})",
                    font=dict(size=22, family='Arial, sans-serif',
                              color='black'),
                    x=0.5, xanchor='center'
                ),
                xaxis=dict(
                    title=dict(text="Opponent Algorithm",
                               font=dict(size=18, color='black')),
                    tickfont=dict(size=16, color='black'),
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showgrid=True
                ),
                yaxis=dict(
                    title=dict(text="Count", font=dict(
                        size=18, color='black')),
                    tickfont=dict(size=16, color='black'),
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showgrid=True
                ),
                barmode='group',
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                width=1400,
                height=800,
                legend=dict(
                    font=dict(size=16, family='Arial, sans-serif',
                              color='black'),
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=2,
                ),
                margin=dict(l=80, r=40, t=80, b=80)
            )

            st.plotly_chart(fig, use_container_width=True)

            total_comparisons = wtl_results['Wins'].sum(
            ) + wtl_results['Losses'].sum()
            total_wins = wtl_results['Wins'].sum()
            win_rate = 100 * total_wins / total_comparisons if total_comparisons > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Wins", f"{total_wins}",
                          help=f"{baseline} wins")
            with col2:
                st.metric("Total Losses", f"{wtl_results['Losses'].sum()}")
            with col3:
                st.metric("Overall Win Rate", f"{win_rate:.1f}%")

        # Pairwise Comparison
        elif analysis_type == "Pairwise Comparison":
            st.subheader(
                f"🔬 Pairwise Statistical Tests: {baseline} vs Others ({scope_title})")

            st.markdown("""
            **Wilcoxon Signed-Rank Test**: Tests if there is a significant difference between paired samples.
            - **Null Hypothesis (H₀)**: No difference between algorithms
            - **Alternative (H₁)**: Baseline performs better (one-sided test)
            - **Significance Level (α)**: 0.05
            - **p < 0.05**: Reject H₀ → Baseline is significantly better ✅
            """)

            competitors = [
                opt for opt in filtered_df["optimizer"].unique() if opt != baseline]

            results = []

            for competitor in competitors:
                baseline_vals = []
                competitor_vals = []

                for func in filtered_df['function'].unique():
                    func_data = filtered_df[filtered_df['function'] == func]

                    for seed in func_data['seed'].unique():
                        base_val = func_data[(func_data['optimizer'] == baseline) &
                                             (func_data['seed'] == seed)][metric].values
                        comp_val = func_data[(func_data['optimizer'] == competitor) &
                                             (func_data['seed'] == seed)][metric].values

                        if len(base_val) > 0 and len(comp_val) > 0:
                            baseline_vals.append(base_val[0])
                            competitor_vals.append(comp_val[0])

                if len(baseline_vals) > 0:
                    stat, p_value = perform_wilcoxon_test(
                        baseline_vals, competitor_vals)

                    mean_baseline = np.mean(baseline_vals)
                    mean_competitor = np.mean(competitor_vals)

                    effect_size = cohens_d(baseline_vals, competitor_vals)
                    effect_interp = interpret_effect_size(effect_size)

                    results.append({
                        'Competitor': competitor,
                        'n': len(baseline_vals),
                        f'{baseline} Mean': mean_baseline,
                        f'{competitor} Mean': mean_competitor,
                        'p-value': p_value if p_value is not None else np.nan,
                        'Significant?': '✅ Yes' if p_value and p_value < 0.05 else '❌ No',
                        "Cohen's d": effect_size,
                        'Effect Size': effect_interp
                    })

            results_df = pd.DataFrame(results)

            def highlight_significant(val):
                if val == '✅ Yes':
                    return 'background-color: #90EE90; color: black; font-weight: bold'
                elif val == '❌ No':
                    return 'background-color: #FFB6C6; color: black'
                return 'background-color: white; color: black'

            # Get dynamic column names for formatting
            baseline_col = f'{baseline} Mean'
            format_dict = {
                baseline_col: '{:.6e}',
                'p-value': lambda x: f'{x:.6f}' if pd.notna(x) and x < 1.0 else '1.000000',
                "Cohen's d": '{:.3f}'
            }
            # Add formatting for competitor mean columns
            for comp in competitors:
                comp_col = f'{comp} Mean'
                if comp_col in results_df.columns:
                    format_dict[comp_col] = '{:.6e}'

            styled_df = results_df.style.format(format_dict).map(
                highlight_significant, subset=['Significant?']
            ).set_properties(**{
                'text-align': 'center',
                'font-size': '16px',
                'padding': '12px',
                'background-color': 'white',
                'color': 'black'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('font-size', '18px'),
                    ('text-align', 'center'),
                    ('font-weight', 'bold'),
                    ('background-color', '#e0e0e0'),
                    ('color', 'black'),
                    ('border', '2px solid black')
                ]},
                {'selector': 'td', 'props': [
                    ('border', '1px solid #ccc')
                ]},
                {'selector': 'tr:hover', 'props': [
                    ('background-color', '#f5f5f5')
                ]}
            ])

            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            sig_count = sum(results_df['Significant?'] == '✅ Yes')
            total_count = len(results_df)

            if sig_count > 0:
                st.success(
                    f"✅ **{baseline}** shows statistically significant improvement over **{sig_count}/{total_count}** algorithms (p < 0.05)")
            else:
                st.warning(
                    f"⚠️ No statistically significant differences detected at α = 0.05")

        # Effect Size Analysis
        elif analysis_type == "Effect Size Analysis":
            st.subheader(
                f"📏 Effect Size Analysis: {baseline} vs Others ({scope_title})")

            st.markdown("""
            **Cohen's d** measures the standardized difference between two means:
            - **|d| < 0.2**: Negligible effect
            - **0.2 ≤ |d| < 0.5**: Small effect
            - **0.5 ≤ |d| < 0.8**: Medium effect
            - **|d| ≥ 0.8**: Large effect
            
            **Negative d**: Baseline performs better  
            **Positive d**: Competitor performs better
            """)

            competitors = [
                opt for opt in filtered_df["optimizer"].unique() if opt != baseline]

            effect_sizes = []

            for competitor in competitors:
                baseline_vals = []
                competitor_vals = []

                for func in filtered_df['function'].unique():
                    func_data = filtered_df[filtered_df['function'] == func]

                    for seed in func_data['seed'].unique():
                        base_val = func_data[(func_data['optimizer'] == baseline) &
                                             (func_data['seed'] == seed)][metric].values
                        comp_val = func_data[(func_data['optimizer'] == competitor) &
                                             (func_data['seed'] == seed)][metric].values

                        if len(base_val) > 0 and len(comp_val) > 0:
                            baseline_vals.append(base_val[0])
                            competitor_vals.append(comp_val[0])

                if len(baseline_vals) > 0:
                    d = cohens_d(baseline_vals, competitor_vals)
                    interp = interpret_effect_size(d)

                    effect_sizes.append({
                        'Competitor': competitor,
                        "Cohen's d": d,
                        'Magnitude': abs(d),
                        'Interpretation': interp,
                        'Favors': baseline if d < 0 else competitor
                    })

            effect_df = pd.DataFrame(effect_sizes).sort_values(
                'Magnitude', ascending=False)

            def color_favors(val):
                if val == baseline:
                    return 'background-color: #90EE90; font-weight: bold; color: black'
                else:
                    return 'background-color: #FFB6C6; color: black'

            styled_effect = effect_df.style.format({
                "Cohen's d": '{:.3f}',
                'Magnitude': '{:.3f}'
            }).map(color_favors, subset=['Favors']).set_properties(**{
                'text-align': 'center',
                'font-size': '16px',
                'padding': '12px',
                'background-color': 'white',
                'color': 'black'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('font-size', '18px'),
                    ('text-align', 'center'),
                    ('font-weight', 'bold'),
                    ('background-color', '#e0e0e0'),
                    ('color', 'black'),
                    ('border', '2px solid black')
                ]},
                {'selector': 'td', 'props': [
                    ('border', '1px solid #ccc')
                ]},
                {'selector': 'tr:hover', 'props': [
                    ('background-color', '#f5f5f5')
                ]}
            ])

            st.dataframe(styled_effect, use_container_width=True,
                         hide_index=True)

            # Visualization
            fig = go.Figure()

            colors_map = {baseline: '#00AA00', 'Competitor': '#CC0000'}
            bar_colors = [colors_map[baseline] if fav == baseline else colors_map['Competitor']
                          for fav in effect_df['Favors']]

            fig.add_trace(go.Bar(
                x=effect_df['Competitor'],
                y=effect_df["Cohen's d"],
                marker=dict(color=bar_colors, line=dict(
                    color='black', width=2)),
                text=effect_df["Cohen's d"].round(2),
                textposition='outside',
                textfont=dict(size=16, color='black'),
                customdata=effect_df['Interpretation'],
                hovertemplate='<b>%{x}</b><br>Cohen\'s d: %{y:.3f}<br>Effect: %{customdata}<extra></extra>'
            ))

            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=2,
                          annotation_text="No Effect", annotation_position="right",
                          annotation=dict(font=dict(size=14, color='black')))
            fig.add_hline(y=-0.8, line_dash="dot", line_color="green", line_width=2,
                          annotation_text="Large (favors baseline)", annotation_position="right",
                          annotation=dict(font=dict(size=14, color='green')))
            fig.add_hline(y=0.8, line_dash="dot", line_color="red", line_width=2,
                          annotation_text="Large (favors competitor)", annotation_position="right",
                          annotation=dict(font=dict(size=14, color='red')))

            fig.update_layout(
                title=dict(
                    text=f"Effect Sizes: {baseline} vs Competitors ({scope_title})",
                    font=dict(size=22, family='Arial, sans-serif',
                              color='black'),
                    x=0.5, xanchor='center'
                ),
                xaxis=dict(
                    title=dict(text="Competitor", font=dict(
                        size=18, color='black')),
                    tickfont=dict(size=16, color='black'),
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showgrid=True
                ),
                yaxis=dict(
                    title=dict(text="Cohen's d", font=dict(
                        size=18, color='black')),
                    tickfont=dict(size=16, color='black'),
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=2
                ),
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=700,
                width=1400,
                margin=dict(l=80, r=200, t=80, b=80)
            )

            st.plotly_chart(fig, use_container_width=True)

            large_effects = len(effect_df[(effect_df['Interpretation'] == 'Large') & (
                effect_df["Cohen's d"] < 0)])

            if large_effects > 0:
                st.success(
                    f"✅ **{baseline}** shows **large practical significance** over **{large_effects}** algorithm(s)")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <small>
    <b>COO & ANN Viewer v5.0</b><br>
    Developed by  
    <a href="https://scholar.google.com/citations?user=Es-kJk4AAAAJ&hl=en" target="_blank">
        Dr. Sandip Garai
    </a>
    &nbsp;·&nbsp;
    <a href="https://scholar.google.com/citations?user=0dQ7Sf8AAAAJ&hl=en&oi=ao" target="_blank">
        Dr. Kanaka K K
    </a><br>
    📧 <a href="mailto:drgaraislab@gmail.com">Contact</a>
    </small>
    """,
    unsafe_allow_html=True
)
