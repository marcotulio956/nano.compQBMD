"""
Otimização multi-objetivo de detectores QBMD usando NSGA-II.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

from .config import loadConfig
from .run_prog import run_qbmd, _default_result
from .models import SimulationResult

# Get the repository root directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]

# Parameter column names in CSV files
PARAM_COLS = ["RW", "RQWt", "RQBt", "MQWt", "LW", "LQWt", "LQBt"]


class InitialPopulationSampling(Sampling):
    """
    Custom sampling that initializes from a CSV file with previous results.
    If the CSV has fewer individuals than pop_size, fills remaining with random sampling.
    """
    
    def __init__(self, initial_pop, bounds, step_nm: float = 0.1):
        """
        Parameters
        ----------
        initial_pop : np.ndarray
            Initial population array of shape (n_individuals, n_vars)
        bounds : dict
            Bounds dictionary with 'lower' and 'upper' lists
        """
        super().__init__()
        self.initial_pop = initial_pop
        self.bounds = bounds
        self.step_nm = step_nm
    
    def _do(self, problem, n_samples, **kwargs):
        n_initial = len(self.initial_pop)
        n_vars = problem.n_var
        
        # Start with initial population
        X = np.zeros((n_samples, n_vars))
        
        if n_initial >= n_samples:
            # Use first n_samples from initial population
            X = self.initial_pop[:n_samples].copy()
        else:
            # Use all initial population and fill rest randomly
            X[:n_initial] = self.initial_pop.copy()
            
            # Random sampling for remaining individuals
            xl = np.array(self.bounds['lower'])
            xu = np.array(self.bounds['upper'])
            
            for i in range(n_initial, n_samples):
                X[i] = xl + np.random.random(n_vars) * (xu - xl)

        # Quantize / sanitize the returned population
        for i in range(len(X)):
            X[i] = _quantize_individual(X[i], step_nm=self.step_nm)

        return X


def load_initial_population(csv_path, param_cols=None):
    """
    Load initial population from a CSV file.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file with previous optimization results
    param_cols : list, optional
        List of parameter column names. Defaults to PARAM_COLS.
        
    Returns
    -------
    np.ndarray
        Array of shape (n_individuals, 7) with parameter values
    """
    if param_cols is None:
        param_cols = PARAM_COLS
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Initial population file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check that all required columns exist
    missing_cols = [col for col in param_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required parameter columns: {missing_cols}")
    
    # Extract parameter values
    pop = df[param_cols].values
    
    print(f"!WARMED UP! Loaded {len(pop)} individuals from {csv_path.name}")
    
    return pop


def _quantize_individual(individual, step_nm: float = 0.1):
    """Return a quantized copy of an individual.

    - Ensure ARG1 (index 0) and ARG5 (index 4) are integers (rounded).
    - Quantize thickness-like floats (indices 1,2,3,5,6) to multiples of `step_nm`.
    """
    ind = np.array(individual, dtype=float).copy()

    # Enforce integer variables
    ind[0] = int(round(ind[0]))
    ind[4] = int(round(ind[4]))

    if step_nm is None or step_nm <= 0:
        return ind

    # Indices corresponding to thickness floats: RQWt, RQBt, MQWt, LQWt, LQBt
    float_indices = [1, 2, 3, 5, 6]
    for idx in float_indices:
        try:
            val = float(ind[idx])
        except Exception:
            continue
        ind[idx] = round(val / step_nm) * step_nm

    return ind


class QuantizedRandomSampling(FloatRandomSampling):
    """Random sampling wrapper that quantizes individuals to the given step."""
    def __init__(self, step_nm: float = 0.1):
        super().__init__()
        self.step_nm = step_nm

    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)
        for i in range(len(X)):
            X[i] = _quantize_individual(X[i], step_nm=self.step_nm)
        return X


class QBMDOptimizationProblem(Problem):
    """
    Problema de otimização multi-objetivo para detectores QBMD.
    
    Variáveis de decisão (7 parâmetros do simulador):
    - ARG1: Número de períodos da primeira estrutura (int)
    - ARG2: Parâmetro material 1 (float)
    - ARG3: Parâmetro material 2 (float)
    - ARG4: Espessura/composição (float)
    - ARG5: Número de períodos da segunda estrutura (int)
    - ARG6: Parâmetro material 3 (float)
    - ARG7: Parâmetro material 4 (float)
    
    Objetivos (maximizar):
    1. Energia do pico principal
    2. Intensidade da fotocorrente
    3. Fator de qualidade
    4. Proeminência do pico
    """
    
    def __init__(self, config, bounds, repo_root=None, save_frequency=5):
        """
        Parâmetros
        ----------
        config : dict
            Configuração carregada do TOML
        bounds : dict
            Limites inferior e superior para cada variável
            Formato: {'lower': [x1_min, x2_min, ...], 'upper': [x1_max, x2_max, ...]}
        repo_root : Path, optional
            Root directory of the repository (for finding executable)
        save_frequency : int, optional
            Save intermediate results every X simulations (0 to disable)
        """
        self.config = config
        self.simulation_counter = 0
        self.repo_root = Path(repo_root) if repo_root else _REPO_ROOT
        self.save_frequency = save_frequency
        
        # Track all evaluated individuals and their objectives
        self.all_parameters = []
        self.all_objectives = []
        
        # Output directory for GA runs
        self.out_root = (self.repo_root / "linux_executable").resolve()
        
        # Output directory for intermediate results
        self.results_dir = self.repo_root / "linux_executable" / "optimization_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Peak detection threshold from config
        self.peak_threshold = config["analysis"]["peak_threshold"]
        
        # Simulation timeout from config
        self.timeout = config["simulator"]["timeout"]
        # Discretization step (nm) for thickness parameters.
        # Use the configured simulator step if provided, but enforce a minimum
        # equal to `optimization.monolayer_thickness` (one atomic layer).
        sim_step = config.get("simulator", {}).get("discretization_step_nm", None)
        mono_step = config.get("optimization", {}).get("monolayer_thickness", None)
        if mono_step is None:
            # Fallback to a small default if the optimization section omits monolayer_thickness
            mono_step = 0.3
            print(
                f"monolayer_thickness {mono_step} nm; using as minimum step."
            )

        # Final discretization step used for quantization (nm)
        self.discretization_step_nm = mono_step
        
        # Definir limites das variáveis
        xl = np.array(bounds['lower'])
        xu = np.array(bounds['upper'])
        
        # 7 variáveis, 4 objetivos, 0 restrições
        super().__init__(
            n_var=7,
            n_obj=4,
            n_ieq_constr=0,
            xl=xl,
            xu=xu
        )
    
    def _save_intermediate_results(self):
        """Save current best individuals to CSV."""
        if len(self.all_parameters) == 0:
            return
            
        # Stack all data
        params = np.array(self.all_parameters)
        objectives = np.array(self.all_objectives)
        combined = np.hstack([params, objectives])
        
        # Save to file named by simulation count
        filename = self.results_dir / f"intermediate_sim_{self.simulation_counter:05d}.csv"
        np.savetxt(
            filename,
            combined,
            delimiter=",",
            header="RW,RQWt,RQBt,MQWt,LW,LQWt,LQBt,max_energy_eV,max_photocurrent,quality_factor,prominence",
            comments=""
        )
        print(f"Resultados intermediários salvos: {filename.name}\n")
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Avalia a população de soluções.
        
        Parâmetros
        ----------
        X : array (pop_size, 7)
            Matriz com os indivíduos da população
        out : dict
            Dicionário para armazenar os objetivos
        """
        F = []  # Matriz de objetivos
        
        for idx, individual in enumerate(X):
            # Executar simulação usando run_qbmd
            self.simulation_counter += 1
            
            # Final sanitize / quantize the individual before running the sim
            q_ind = _quantize_individual(individual, step_nm=self.discretization_step_nm)

            try:
                # Call run_qbmd with return_results=True to get SimulationResult
                out_dir, result = run_qbmd(
                    RW=int(q_ind[0]),
                    RQWt=q_ind[1],
                    RQBt=q_ind[2],
                    MQWt=q_ind[3],
                    LW=int(q_ind[4]),
                    LQWt=q_ind[5],
                    LQBt=q_ind[6],
                    thickness_units="nm",
                    create_param_folder=True,
                    out_root=str(self.out_root),
                    timeout=self.timeout,
                    return_results=True,
                    peak_threshold=self.peak_threshold,
                    simulation_id=self.simulation_counter,
                )
            except Exception as e:
                print(f"  Erro ao executar simulação {self.simulation_counter}: {e}")
                result = _default_result()
            
            # Store raw objectives (positive values for storage)
            raw_objectives = [
                result.max_energy,
                result.max_photocurrent,
                result.quality_factor,
                result.prominence
            ]
            
            # NSGA-II minimiza, então negamos para maximizar
            objectives = [-obj for obj in raw_objectives]
            
            F.append(objectives)
            
            # Store for intermediate saving
            self.all_parameters.append(individual.copy())
            self.all_objectives.append(raw_objectives)
            
            # Log de progresso
            if self.simulation_counter % 5 == 0:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Simulação {self.simulation_counter}")
                print(f"  Parâmetros: {individual}")
                print(f"  Energia: {result.max_energy:.2f} eV")
                print(f"  Intensidade: {result.max_photocurrent:.6e}")
                print(f"  Q: {result.quality_factor:.2f}")
                print(f"  Proeminência: {result.prominence:.4f}")
            
            # Save intermediate results at specified frequency
            if self.save_frequency > 0 and self.simulation_counter % self.save_frequency == 0:
                self._save_intermediate_results()
        
        out["F"] = np.array(F)


def run_optimization(config_path=None, repo_root=None, initial_population_csv=None):
    """Executa a otimização multi-objetivo com NSGA-II
    
    Parameters
    ----------
    config_path : str, optional
        Path to config.toml file. If None, uses default in include directory.
    repo_root : str or Path, optional
        Repository root directory. If None, auto-detected.
    initial_population_csv : str or Path, optional
        Path to CSV file with initial population from previous GA runs.
        The CSV should have columns: RW, RQWt, RQBt, MQWt, LW, LQWt, LQBt.
        If provided, the GA will be warm-started with these individuals.
    """
    
    # Auto-detect if config_path is actually a CSV file (common mistake)
    # This happens when user calls run_optimization(csv_file) without keyword
    if config_path is not None and str(config_path).endswith('.csv'):
        # User likely meant to pass initial_population_csv
        initial_population_csv = config_path
        config_path = None
    
    # Resolve paths
    if repo_root is None:
        repo_root = _REPO_ROOT
    else:
        repo_root = Path(repo_root)
    
    # Carregar configuração
    if config_path is None:
        config_path = _SCRIPT_DIR / "config.toml"
    config = loadConfig(str(config_path))
    
    # Carregar limites das variáveis do arquivo de configuração
    bounds_config = config["optimization"]["bounds"]
    bounds = {
        'lower': [
            bounds_config["arg1_min"],  # ARG1: mínimo períodos
            bounds_config["arg2_min"],  # ARG2: mínimo valor do parâmetro
            bounds_config["arg3_min"],  # ARG3: mínimo valor do parâmetro
            bounds_config["arg4_min"],  # ARG4: mínimo espessura
            bounds_config["arg5_min"],  # ARG5: mínimo períodos
            bounds_config["arg6_min"],  # ARG6: mínimo valor do parâmetro
            bounds_config["arg7_min"]   # ARG7: mínimo valor do parâmetro
        ],
        'upper': [
            bounds_config["arg1_max"],  # ARG1: máximo períodos
            bounds_config["arg2_max"],  # ARG2: máximo valor do parâmetro
            bounds_config["arg3_max"],  # ARG3: máximo valor do parâmetro
            bounds_config["arg4_max"],  # ARG4: máximo espessura
            bounds_config["arg5_max"],  # ARG5: máximo períodos
            bounds_config["arg6_max"],  # ARG6: máximo valor do parâmetro
            bounds_config["arg7_max"]   # ARG7: máximo valor do parâmetro
        ]
    }
    
    # Get save frequency from config (default to 5 if not specified)
    save_frequency = config["optimization"].get("save_frequency", 5)
    
    # Criar problema de otimização
    problem = QBMDOptimizationProblem(
        config=config, 
        bounds=bounds, 
        repo_root=repo_root,
        save_frequency=save_frequency
    )
    
    # Determine sampling strategy
    if initial_population_csv is not None:
        # Load initial population from CSV for warm start
        initial_pop = load_initial_population(initial_population_csv)
        sampling = InitialPopulationSampling(initial_pop, bounds, step_nm=problem.discretization_step_nm)
        warm_start = True
    else:
        # Random sampling
        sampling = QuantizedRandomSampling(step_nm=problem.discretization_step_nm)
        warm_start = False
    
    # Configurar NSGA-II
    algorithm = NSGA2(
        pop_size=config["optimization"]["population_size"],
        sampling=sampling,
        crossover=SBX(prob=config["optimization"]["crossover_prob"], eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # Critério de parada
    termination = get_termination("n_gen", config["optimization"]["num_generations"])
    
    # Executar otimização
    print("="*70)
    print("OTIMIZAÇÃO NSGA-II - QUANTUM BRAGG MIRROR DETECTOR")
    print("="*70)
    print(f"População: {config['optimization']['population_size']}")
    print(f"Gerações: {config['optimization']['num_generations']}")
    print(f"Total de simulações: {config['optimization']['population_size'] * config['optimization']['num_generations']}")
    if warm_start:
        print(f"*População Base: Inicializando com população customizada")
    if save_frequency > 0:
        print(f"Salvando resultados a cada: {save_frequency} simulações")
        print(f"Diretório: nanocomp/simulation_results/")
    print("="*70)
    
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True,
        save_history=True
    )
    
    # Processar resultados
    print("\n" + "="*70)
    print("OTIMIZAÇÃO CONCLUÍDA")
    print("="*70)
    print(f"Soluções na Frente de Pareto: {len(res.X)}")
    
    # Converter objetivos de volta (remover negativo)
    pareto_objectives = -res.F
    
    # Salvar resultados
    save_results(res, pareto_objectives, config)
    
    return res


def save_results(res, pareto_objectives, config, output_base_dir=None):
    """Salva os resultados da otimização"""
    
    if output_base_dir is None:
        output_dir = _REPO_ROOT / "nanocomp" / "optimization_results"
    else:
        output_dir = Path(output_base_dir) / "optimization_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar parâmetros (variáveis de decisão)
    params_file = output_dir / f"pareto_parameters_{timestamp}.csv"
    np.savetxt(
        params_file,
        res.X,
        delimiter=",",
        header="ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7",
        comments=""
    )
    
    # Salvar objetivos
    objectives_file = output_dir / f"pareto_objectives_{timestamp}.csv"
    np.savetxt(
        objectives_file,
        pareto_objectives,
        delimiter=",",
        header="max_energy_eV,max_photocurrent,quality_factor,prominence",
        comments=""
    )
    
    # Salvar resultados combinados
    combined = np.hstack([res.X, pareto_objectives])
    combined_file = output_dir / f"pareto_full_{timestamp}.csv"
    np.savetxt(
        combined_file,
        combined,
        delimiter=",",
        header="ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7,max_energy_eV,max_photocurrent,quality_factor,prominence",
        comments=""
    )
    
    print(f"\n Resultados salvos em: {output_dir}")
    print(f"   - Parâmetros: {params_file.name}")
    print(f"   - Objetivos: {objectives_file.name}")
    print(f"   - Completo: {combined_file.name}")


if __name__ == "__main__":
    results = run_optimization()
