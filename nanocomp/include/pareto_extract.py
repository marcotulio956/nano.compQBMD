#!/usr/bin/env python3
"""
pareto_extract.py

Pareto front analysis and visualization for QBMD optimization results.

Usage (CLI):
    python pareto_extract.py path/to/ga_output.csv

Usage (Module):
    from include.pareto_extract import load_pareto_results, display_best_solutions, visualize_pareto_2d, visualize_pareto_3d
    
    df = load_pareto_results("path/to/pareto_full_*.csv")
    display_best_solutions(df)
    visualize_pareto_2d(df)
    visualize_pareto_3d(df)
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.plotting import scatter_matrix
from typing import Optional, List, Tuple

# -------------------- Default settings --------------------
# Default objective columns for QBMD optimization
DEFAULT_OBJECTIVES = [
    "max_energy_eV",
    "max_photocurrent",
    "quality_factor",
    "prominence",
]

DEFAULT_PARAM_COLS = ["RW", "RQWt", "RQBt", "MQWt", "LW", "LQWt", "LQBt"]

DEFAULT_LABELS = {
    "max_energy_eV": "Energia do Pico (eV)",
    "max_photocurrent": "Intensidade da Fotocorrente",
    "quality_factor": "Fator de Qualidade",
    "prominence": "Proeminência",
}

# Direction for optimization (all are maximized in QBMD)
DEFAULT_DIRECTION = {
    "max_photocurrent": "max",
    "quality_factor": "max",
    "prominence": "max",
    "max_energy_eV": "max",
}

# Plotting options
DEFAULT_PLOT_DIR = "pareto_plots"
# ----------------------------------------------------------------


def load_pareto_results(results_path: Optional[str] = None, results_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load Pareto front results from CSV file.
    
    Parameters
    ----------
    results_path : str, optional
        Direct path to a pareto_full_*.csv file
    results_dir : str, optional
        Directory containing pareto_full_*.csv files (will use most recent)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Pareto front results
    """
    if results_path is not None:
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")
        df = pd.read_csv(results_path)
        print(f"Loaded {len(df)} Pareto-optimal solutions from {results_path}")
        return df
    
    if results_dir is not None:
        search_dirs = [Path(results_dir)]
    else:
        # Try to find results in common locations
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parents[1]
        search_dirs = [
            repo_root / "linux_executable" / "optimization_results",
            Path("."),
        ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            csv_files = list(search_dir.glob("pareto_full_*.csv"))
            if csv_files:
                csv_files.sort(reverse=True)  # Most recent first
                latest_file = csv_files[0]
                df = pd.read_csv(latest_file)
                print(f"Loaded {len(df)} Pareto-optimal solutions from {latest_file}")
                return df
    
    raise FileNotFoundError("No pareto_full_*.csv files found:"+results_path+" Run the GA optimization first.")


def display_best_solutions(df: pd.DataFrame, 
                          objective_cols: Optional[List[str]] = None,
                          param_cols: Optional[List[str]] = None) -> None:
    """
    Display the best solutions for each objective.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Pareto front results
    objective_cols : list, optional
        List of objective column names
    param_cols : list, optional
        List of parameter column names
    """
    if objective_cols is None:
        objective_cols = [col for col in DEFAULT_OBJECTIVES if col in df.columns]
    if param_cols is None:
        param_cols = [col for col in DEFAULT_PARAM_COLS if col in df.columns]
    
    print("\n" + "="*70)
    print("MELHORES SOLUÇÕES POR OBJETIVO")
    print("="*70)
    
    for col in objective_cols:
        if col not in df.columns:
            continue
            
        best_idx = df[col].idxmax()
        best_row = df.loc[best_idx]
        label = DEFAULT_LABELS.get(col, col)
        
        print(f"\nMelhor {label}:")
        print(f"   Valor: {best_row[col]:.6f}")
        print(f"   Parâmetros: ", end="")
        params = [f"{p}={best_row[p]}" for p in param_cols if p in df.columns]
        print(", ".join(params))
        
        print(f"   Outros objetivos:")
        for other_col in objective_cols:
            if other_col != col and other_col in df.columns:
                other_label = DEFAULT_LABELS.get(other_col, other_col)
                print(f"     - {other_label}: {best_row[other_col]:.6f}")


def visualize_pareto_2d(df: pd.DataFrame,
                        objective_cols: Optional[List[str]] = None,
                        output_dir: Optional[str] = None,
                        show: bool = True,
                        save: bool = True) -> Optional[str]:
    """
    Visualize 2D projections of the Pareto front.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Pareto front results
    objective_cols : list, optional
        List of objective column names (default: 4 QBMD objectives)
    output_dir : str, optional
        Directory to save plots
    show : bool
        Whether to display the plot
    save : bool
        Whether to save the plot to file
        
    Returns
    -------
    str or None
        Path to saved file if save=True
    """
    if objective_cols is None:
        objective_cols = [col for col in DEFAULT_OBJECTIVES if col in df.columns]
    
    labels = [DEFAULT_LABELS.get(col, col) for col in objective_cols]
    
    # Generate all pairs
    n_obj = len(objective_cols)
    pairs = [(i, j) for i in range(n_obj) for j in range(i+1, n_obj)]
    
    n_plots = len(pairs)
    if n_plots == 0:
        print("Not enough objectives to create 2D plots")
        return None
    
    # Create subplots
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        ax.scatter(
            df[objective_cols[i]], 
            df[objective_cols[j]], 
            c='red', 
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        ax.set_xlabel(labels[i], fontsize=11, fontweight='bold')
        ax.set_ylabel(labels[j], fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{labels[i]} vs {labels[j]}", fontsize=10)
    
    # Hide empty subplots
    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Frente de Pareto - Projeções 2D', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = None
    if save:
        if output_dir is None:
            script_dir = Path(__file__).resolve().parent
            output_dir = script_dir.parents[1] / "nanocomp" / DEFAULT_PLOT_DIR
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'pareto_front_projections.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualização da Frente de Pareto salva: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(output_file) if output_file else None


def visualize_pareto_3d(df: pd.DataFrame,
                        objective_cols: Optional[List[str]] = None,
                        color_col: Optional[str] = None,
                        output_dir: Optional[str] = None,
                        show: bool = True,
                        save: bool = True) -> Optional[str]:
    """
    Visualize 3D projection of the Pareto front.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Pareto front results
    objective_cols : list, optional
        List of 3 objective column names for x, y, z axes
    color_col : str, optional
        Column to use for coloring points (default: 4th objective or None)
    output_dir : str, optional
        Directory to save plots
    show : bool
        Whether to display the plot
    save : bool
        Whether to save the plot to file
        
    Returns
    -------
    str or None
        Path to saved file if save=True
    """
    if objective_cols is None:
        available = [col for col in DEFAULT_OBJECTIVES if col in df.columns]
        if len(available) < 3:
            print("Not enough objectives for 3D visualization")
            return None
        objective_cols = available[:3]
        if color_col is None and len(available) > 3:
            color_col = available[3]
    
    labels = [DEFAULT_LABELS.get(col, col) for col in objective_cols]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if color_col and color_col in df.columns:
        scatter = ax.scatter(
            df[objective_cols[0]], 
            df[objective_cols[1]], 
            df[objective_cols[2]],
            c=df[color_col],
            cmap='viridis', 
            s=60, 
            alpha=0.8
        )
        color_label = DEFAULT_LABELS.get(color_col, color_col)
        plt.colorbar(scatter, label=color_label, shrink=0.6)
    else:
        ax.scatter(
            df[objective_cols[0]], 
            df[objective_cols[1]], 
            df[objective_cols[2]],
            c='red',
            s=60, 
            alpha=0.8
        )
    
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title('Frente de Pareto 3D')
    
    plt.tight_layout()
    
    output_file = None
    if save:
        if output_dir is None:
            script_dir = Path(__file__).resolve().parent
            output_dir = script_dir.parents[1] / "nanocomp" / DEFAULT_PLOT_DIR
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'pareto_front_3d.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualização 3D da Frente de Pareto salva: {output_file}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return str(output_file) if output_file else None


def get_statistics(df: pd.DataFrame, 
                   objective_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Get summary statistics for objectives.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Pareto front results
    objective_cols : list, optional
        List of objective column names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with min, max, mean statistics
    """
    if objective_cols is None:
        objective_cols = [col for col in DEFAULT_OBJECTIVES if col in df.columns]
    
    stats = []
    for col in objective_cols:
        if col in df.columns:
            stats.append({
                'objective': DEFAULT_LABELS.get(col, col),
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            })
    
    return pd.DataFrame(stats)


def analyze_parameter_objective_relationships(df: pd.DataFrame,
                                               objective_cols: Optional[List[str]] = None,
                                               param_cols: Optional[List[str]] = None,
                                               top_percent: float = 20.0) -> dict:
    """
    Analyze relationships between input parameters and objectives.
    
    For each objective, identifies the top performers and analyzes
    the parameter ranges that lead to high values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Pareto front results
    objective_cols : list, optional
        List of objective column names
    param_cols : list, optional
        List of parameter column names
    top_percent : float
        Percentage of top solutions to analyze (default: 20%)
        
    Returns
    -------
    dict
        Dictionary with analysis results for each objective
    """
    if objective_cols is None:
        objective_cols = [col for col in DEFAULT_OBJECTIVES if col in df.columns]
    if param_cols is None:
        param_cols = [col for col in DEFAULT_PARAM_COLS if col in df.columns]
    
    n_top = max(1, int(len(df) * top_percent / 100))
    
    analysis = {}
    
    for obj in objective_cols:
        if obj not in df.columns:
            continue
            
        # Get top performers for this objective
        top_df = df.nlargest(n_top, obj)
        
        obj_analysis = {
            'objective': DEFAULT_LABELS.get(obj, obj),
            'top_n': n_top,
            'parameters': {}
        }
        
        for param in param_cols:
            if param not in df.columns:
                continue
                
            param_stats = {
                'global_min': df[param].min(),
                'global_max': df[param].max(),
                'global_mean': df[param].mean(),
                'top_min': top_df[param].min(),
                'top_max': top_df[param].max(),
                'top_mean': top_df[param].mean(),
                'top_std': top_df[param].std(),
            }
            
            # Calculate correlation between parameter and objective
            if len(df) > 2:
                param_stats['correlation'] = df[param].corr(df[obj])
            else:
                param_stats['correlation'] = 0.0
                
            obj_analysis['parameters'][param] = param_stats
            
        analysis[obj] = obj_analysis
    
    return analysis


def print_parameter_analysis(analysis: dict) -> None:
    """
    Print a formatted analysis of parameter-objective relationships.
    
    Parameters
    ----------
    analysis : dict
        Analysis dictionary from analyze_parameter_objective_relationships()
    """
    print("\n" + "="*80)
    print("ANÁLISE DE RELAÇÕES PARÂMETROS-OBJETIVOS")
    print("="*80)
    
    for obj_key, obj_data in analysis.items():
        print(f"\n{'─'*80}")
        print(f"GOAL: {obj_data['objective']}")
        print(f"   (Análise dos top {obj_data['top_n']} indivíduos)")
        print(f"{'─'*80}")
        
        # Table header
        print(f"\n{'Parâmetro':<10} │ {'Global':^25} │ {'Top Performers':^25} │ {'Corr':^8}")
        print(f"{'':<10} │ {'Min':^8} {'Max':^8} {'Mean':^8} │ {'Min':^8} {'Max':^8} {'Mean':^8} │ {'':<8}")
        print(f"{'─'*10}─┼─{'─'*25}─┼─{'─'*25}─┼─{'─'*8}")
        
        for param, stats in obj_data['parameters'].items():
            corr_str = f"{stats['correlation']:.3f}" if stats['correlation'] != 0 else "N/A"
            corr_indicator = ""
            if abs(stats['correlation']) > 0.5:
                corr_indicator = "↑" if stats['correlation'] > 0 else "↓"
            
            print(f"{param:<10} │ {stats['global_min']:>8.2f} {stats['global_max']:>8.2f} {stats['global_mean']:>8.2f} │ "
                  f"{stats['top_min']:>8.2f} {stats['top_max']:>8.2f} {stats['top_mean']:>8.2f} │ {corr_str:>6} {corr_indicator}")
    
    # Summary of key findings
    print("\n" + "="*80)
    print("CONCLUSÕES E RECOMENDAÇÕES")
    print("="*80)
    
    for obj_key, obj_data in analysis.items():
        print(f"\nGOAL Para maximizar {obj_data['objective']}:")
        
        recommendations = []
        for param, stats in obj_data['parameters'].items():
            corr = stats['correlation']
            
            if abs(corr) > 0.3:
                if corr > 0:
                    recommendations.append(f"   • {param}: valores MAIORES tendem a melhorar (corr={corr:.2f})")
                else:
                    recommendations.append(f"   • {param}: valores MENORES tendem a melhorar (corr={corr:.2f})")
            
            # Check if top performers use a narrow range
            global_range = stats['global_max'] - stats['global_min']
            if global_range > 0:
                top_range = stats['top_max'] - stats['top_min']
                if top_range < global_range * 0.5:
                    recommendations.append(f"   • {param}: faixa preferida [{stats['top_min']:.2f}, {stats['top_max']:.2f}]")
        
        if recommendations:
            for rec in recommendations[:5]:  # Limit to top 5 recommendations
                print(rec)
        else:
            print("   • Sem correlações significativas detectadas")


def get_conclusions_markdown(analysis: dict) -> str:
    """
    Generate markdown text with conclusions about parameter-objective relationships.
    
    Parameters
    ----------
    analysis : dict
        Analysis dictionary from analyze_parameter_objective_relationships()
        
    Returns
    -------
    str
        Markdown formatted conclusions
    """
    lines = []
    lines.append("## 📋 Conclusões da Análise do Pareto Front\n")
    lines.append("### Relações entre Parâmetros de Entrada e Objetivos\n")
    
    for obj_key, obj_data in analysis.items():
        lines.append(f"\n#### {obj_data['objective']}\n")
        lines.append(f"Análise baseada nos top {obj_data['top_n']} indivíduos do Pareto front.\n")
        
        # Find strong correlations
        strong_pos = []
        strong_neg = []
        preferred_ranges = []
        
        for param, stats in obj_data['parameters'].items():
            corr = stats['correlation']
            if corr > 0.3:
                strong_pos.append((param, corr))
            elif corr < -0.3:
                strong_neg.append((param, corr))
            
            global_range = stats['global_max'] - stats['global_min']
            if global_range > 0:
                top_range = stats['top_max'] - stats['top_min']
                if top_range < global_range * 0.5:
                    preferred_ranges.append((param, stats['top_min'], stats['top_max']))
        
        if strong_pos:
            lines.append("**Correlações positivas (valores maiores ajudam):**\n")
            for param, corr in sorted(strong_pos, key=lambda x: -x[1]):
                lines.append(f"- `{param}`: correlação = {corr:.3f}\n")
        
        if strong_neg:
            lines.append("\n**Correlações negativas (valores menores ajudam):**\n")
            for param, corr in sorted(strong_neg, key=lambda x: x[1]):
                lines.append(f"- `{param}`: correlação = {corr:.3f}\n")
        
        if preferred_ranges:
            lines.append("\n**Faixas de valores preferidas nos melhores indivíduos:**\n")
            for param, vmin, vmax in preferred_ranges:
                lines.append(f"- `{param}`: [{vmin:.2f}, {vmax:.2f}]\n")
        
        if not strong_pos and not strong_neg and not preferred_ranges:
            lines.append("*Sem correlações fortes detectadas para este objetivo.*\n")
    
    lines.append("\n---\n")
    lines.append("*Nota: Correlações acima de |0.3| são consideradas relevantes. ")
    lines.append("Faixas preferidas indicam parâmetros onde os melhores indivíduos convergem.*\n")
    
    return "".join(lines)


# -------------------- Legacy CLI functions --------------------
def read_csv(path):
    df = pd.read_csv(path)
    return df

def build_objective_matrix(df, objectives, direction):
    """
    Convert chosen objectives to a numeric matrix where *higher is better* for all columns.
    """
    arrs = []
    used = []
    for obj in objectives:
        if obj not in df.columns:
            continue
        col = df[obj].astype(float).to_numpy(copy=True)
        dirn = direction.get(obj, "max")
        if dirn == "max":
            arrs.append(col)
        else:
            arrs.append(-col)
        used.append(obj)
    if not arrs:
        return None, []
    mat = np.vstack(arrs).T
    return mat, used

def is_pareto_efficient(scores):
    """
    Vectorized check for Pareto efficiency (non-dominated points).
    """
    n_points = scores.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if not is_efficient[i]:
            continue
        comp = scores >= scores[i]
        all_ge = np.all(comp, axis=1)
        strictly_greater = np.any(scores > scores[i], axis=1)
        dominated_by_i = all_ge & strictly_greater
        dominated_by_i[i] = False
        is_efficient[dominated_by_i] = False
    return is_efficient

def main(csv_path):
    if not os.path.exists(csv_path):
        print("File not found:", csv_path)
        sys.exit(1)

    df = read_csv(csv_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Build objective matrix
    obj_matrix, used_obj_names = build_objective_matrix(df, DEFAULT_OBJECTIVES, DEFAULT_DIRECTION)
    if obj_matrix is None:
        print("No valid objectives found in CSV")
        sys.exit(1)
        
    print("Using objectives:", used_obj_names)
    
    # Compute pareto mask
    pareto_mask = is_pareto_efficient(obj_matrix)
    pareto_df = df.loc[pareto_mask].copy().reset_index(drop=True)

    print(f"Pareto front size: {len(pareto_df)} points out of {len(df)}.")

    # Save pareto CSV
    base, ext = os.path.splitext(csv_path)
    out_path = base + "_pareto.csv"
    pareto_df.to_csv(out_path, index=False)
    print(f"Wrote Pareto front CSV to: {out_path}")

    # Display best solutions
    display_best_solutions(pareto_df)
    
    # Visualize
    visualize_pareto_2d(pareto_df, show=False)
    visualize_pareto_3d(pareto_df, show=False)

    print("\nDone.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pareto_extract.py path/to/ga_output.csv")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
