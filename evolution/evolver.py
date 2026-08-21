"""
CMA-ES optimization of PFSM logistic parameters (L1 and L2).

Parameter vector layout (24 values total, normalized to [-1, 1]):
  [0:6]   L1 - per state (3 states x 2 params: coeff, intercept)
  [6:24]  L2 - per transition (9 transitions x 2 params: coeff, intercept)

Transitions order: (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)

Fitness:
  Aggregation : minimize  mean_dist_to_com - avg_neighbors
  Diffusion   : minimize  avg_neighbors + 1 / (mean_dist_to_com + eps)
"""

import json
import subprocess
import os
import numpy as np
import cma
import copy

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
STATE_EST_DIR  = os.path.join(SIM_ROOT, "state_estimations")
SIM_OUTPUT_DIR = SIM_ROOT
RUN_SCRIPT     = os.path.join(SIM_ROOT, "offline_build_and_run.sh")

L1_PATH       = os.path.join(STATE_EST_DIR, "l1.json")
L2_PATH       = os.path.join(STATE_EST_DIR, "l2.json")
OFF_FOOD_PATH = os.path.join(STATE_EST_DIR, "off_food_transitions.json")

AGENT_IDS   = list(range(37, 46))
STATES      = [0, 1, 2]
TRANSITIONS = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

NEIGHBOR_RADIUS_MM = 0.5
EPS                = 1e-6

# ─── Parameter scaling ────────────────────────────────────────────────────────

COEFF_RANGE     = (-0.5, 0.5)
INTERCEPT_RANGE = (-50.0, 50.0)
HEIGHT_RANGE    = (0.0, 1.0)
L1_HEIGHT_RANGE  = (0.0, 1.0)
L2_HEIGHT_RANGE  = (0.0, 2.0)

'''PARAM_RANGES = [
    COEFF_RANGE if i % 2 == 0 else INTERCEPT_RANGE
    for i in range(24)
]'''
# triplets: coeff, intercept, height — 1 for L1, 9 for L2
'''PARAM_RANGES = [
    COEFF_RANGE if i % 3 == 0 else
    INTERCEPT_RANGE if i % 3 == 1 else
    HEIGHT_RANGE
    for i in range(30)  # 3 + 27
]'''

PARAM_RANGES = []

# L1: only state 2
PARAM_RANGES += [
    COEFF_RANGE,
    INTERCEPT_RANGE,
    L1_HEIGHT_RANGE,
]

# L2
# one fixed height per source state:
# choose (0,0), (1,1), (2,2) as reference transitions
REFERENCE_TRANSITIONS = {(0, 0), (1, 1), (2, 2)}

for src, dst in TRANSITIONS:
    PARAM_RANGES += [
        COEFF_RANGE,
        INTERCEPT_RANGE,
    ]

    if (src, dst) not in REFERENCE_TRANSITIONS:
        PARAM_RANGES += [L2_HEIGHT_RANGE]

L2_INDEX = {}
idx = 3

for src, dst in TRANSITIONS:
    coeff_idx = idx
    intercept_idx = idx + 1
    idx += 2

    if (src, dst) in REFERENCE_TRANSITIONS:
        height_idx = None
    else:
        height_idx = idx
        idx += 1

    L2_INDEX[(src, dst)] = (coeff_idx, intercept_idx, height_idx)


DOMAIN_W = 60.0
DOMAIN_H = 60.0

LOG_EVERY   = 20
LOG_DIR     = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
L1_STATES = [2]  # only evolve L1 for state 2

def denormalize(x: np.ndarray) -> np.ndarray:
    """Map normalized [-1, 1] vector to actual parameter ranges."""
    out = np.empty_like(x)
    for i, (lo, hi) in enumerate(PARAM_RANGES):
        out[i] = lo + (np.clip(x[i], -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)
    return out


# ─── Static data ──────────────────────────────────────────────────────────────

with open(OFF_FOOD_PATH) as f:
    OFF_FOOD = json.load(f)

# ─── JSON helpers ─────────────────────────────────────────────────────────────

def _neutral_entry(coeff: float, intercept: float, height:float, p_off_food: float) -> dict:
    return {
        "p_off_food":      p_off_food,
        "tau":             -1,
        "model_coeff":     coeff,
        "model_intercept": intercept,
        "model_height":    height,
        "mean":            0,
        "std":             1,
        "p_relevant":      0,
        "sign":            1,
    }


def write_l1(params: np.ndarray):
    l1 = {}
    for idx, state in enumerate(L1_STATES):
        coeff      = float(params[idx * 3])
        intercept  = float(params[idx * 3 + 1])
        height     = float(params[idx * 3 + 2])
        p_off_food = OFF_FOOD[str(state)][str(state)]
        l1[str(state)] = _neutral_entry(coeff, intercept, height, p_off_food)
    with open(L1_PATH, "w") as f:
        json.dump(l1, f, indent=2)


def write_l2(params):
    l2 = {str(s): {} for s in STATES}

    for src, dst in TRANSITIONS:
        coeff_idx, intercept_idx, height_idx = L2_INDEX[(src, dst)]

        coeff = float(params[coeff_idx])
        intercept = float(params[intercept_idx])

        if height_idx is None:
            height = 1.0
        else:
            height = float(params[height_idx])

        p_off_food = OFF_FOOD[str(src)][str(dst)]

        l2[str(src)][str(dst)] = _neutral_entry(
            coeff, intercept, height, p_off_food
        )

    with open(L2_PATH, "w") as f:
        json.dump(l2, f, indent=2)


# ─── Simulator interface ───────────────────────────────────────────────────────

def run_simulator(seed: int):
    result = subprocess.run(
        ["bash", RUN_SCRIPT, str(seed)],
        cwd=SIM_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[SIM STDERR]", result.stderr[-2000:])
        raise RuntimeError(f"Simulator exited with code {result.returncode}")


def load_output() -> dict:
    path = os.path.join(SIM_OUTPUT_DIR, f"aagent_data.json")
    with open(path) as f:
        return json.load(f)


N_ENSEMBLE = 5

def evaluate(x_norm: np.ndarray) -> dict:
    params = denormalize(x_norm)
    write_l1(params)
    write_l2(params)

    all_avg_n       = []

    for run in range(N_ENSEMBLE):
        run_simulator(seed=run)

        path = os.path.join(SIM_OUTPUT_DIR, "agent_data.json")
        with open(path) as f:
            data = json.load(f)

        avg_n         = float(data["avg_neighbors"])
        '''pos           = np.array(data["positions"])   # (n_agents, n_frames, 2)
        n_agents, n_frames, _ = pos.shape

        com           = pos.mean(axis=0)
        dist_to_com   = np.linalg.norm(pos - com[None, :, :], axis=2)
        fractions     = compute_largest_cluster_fractions(pos)

        all_avg_n.append(avg_n)
        all_dist_com.append(float(dist_to_com.mean()))
        all_fractions.append(fractions)'''
        all_avg_n.append(avg_n)

    # average metrics across ensemble
    #mean_fractions = np.mean(all_fractions, axis=0)  # (n_frames,)

    return {
        "avg_neighbors":             float(np.mean(all_avg_n))#,
        #"mean_dist_to_com":          float(np.mean(all_dist_com)),
        #"mean_cluster_size":         float(np.mean(mean_fractions) * n_agents),
        #"largest_cluster_fractions": mean_fractions,
    }


import json
import numpy as np
from pathlib import Path
from scipy.stats import kurtosis
from scipy.ndimage import label


# --- loaded once, at module import time ---
_SIM_PARAMS = json.loads((Path(SIM_OUTPUT_DIR) / "simulation_parameters.json").read_text())
WORM_COUNT = _SIM_PARAMS["WORM_COUNT"]
GRID_N = _SIM_PARAMS["GRID_N"]
N_STEPS = _SIM_PARAMS["N_STEPS"]


def resolve_count_grid_path(sim_output_dir: Path) -> Path:
    """log_matrices picks dense vs sparse at runtime and encodes it in the filename."""
    dense = sim_output_dir / "agent_count_grid_b.dat"
    sparse = sim_output_dir / "agent_count_grid_sparse_b.dat"
    if sparse.exists():
        return sparse
    if dense.exists():
        return dense
    raise FileNotFoundError(
        f"Neither {dense.name} nor {sparse.name} found in {sim_output_dir}"
    )


def clear_old_count_grids(sim_output_dir: Path) -> None:
    """Delete stale outputs before a run so a leftover file from a prior sparsity outcome
    can't get picked up by resolve_count_grid_path."""
    for name in ("agent_count_grid_b.dat", "agent_count_grid_sparse_b.dat"):
        p = sim_output_dir / name
        if p.exists():
            p.unlink()


def load_count_grid(path: Path) -> np.ndarray:
    """
    Loads agent_count_grid[_sparse]_b.dat -> dense (n_steps, grid_n, grid_n) int32 array.
    Both formats share a header: int32 n_steps, int32 grid_n.
    """
    is_sparse = "_sparse" in path.stem

    with open(path, "rb") as f:
        n_steps, grid_n = np.fromfile(f, dtype=np.int32, count=2)

        if not is_sparse:
            expected = n_steps * grid_n * grid_n
            flat = np.fromfile(f, dtype=np.int32, count=expected)
            if flat.size != expected:
                raise ValueError(f"{path}: expected {expected} ints, got {flat.size}")
            return flat.reshape(n_steps, grid_n, grid_n)

        # sparse: per timestep -> int32 t, int32 nnz, then nnz * (int32 i, int32 j, int32 value)
        # Vectorized: read the whole file body at once as a flat int32 stream, then
        # walk timestep boundaries using numpy slicing instead of per-timestep fromfile calls.
        rest = np.fromfile(f, dtype=np.int32)

        grid = np.zeros((n_steps, grid_n, grid_n), dtype=np.int32)
        pos = 0
        for t in range(n_steps):
            t_val, nnz = rest[pos], rest[pos + 1]
            pos += 2
            if nnz > 0:
                block = rest[pos: pos + 3 * nnz].reshape(nnz, 3)
                grid[t_val, block[:, 0], block[:, 1]] = block[:, 2]
                pos += 3 * nnz
        return grid


def cluster_metric_per_frame(
        grids: np.ndarray,
        worm_count: int,
        min_cluster_size: int = 4
) -> np.ndarray:

    n_steps = grids.shape[0]
    metrics = np.zeros(n_steps, dtype=np.float64)

    structure_8 = np.ones((3, 3), dtype=np.int8)

    for t in range(n_steps):
        grid = grids[t]
        occupied = grid > 0

        labeled, n_clusters = label(
            occupied,
            structure=structure_8
        )

        if n_clusters == 0:
            continue

        cluster_sums = np.bincount(
            labeled.ravel(),
            weights=grid.ravel(),
            minlength=n_clusters + 1
        )

        cluster_sizes = np.bincount(
            labeled.ravel(),
            minlength=n_clusters + 1
        )

        valid = cluster_sizes >= min_cluster_size
        valid[0] = False

        if not valid.any():
            continue

        metrics[t] = cluster_sums[valid].max() / worm_count

    return metrics


def evaluate_count_grid(x_norm: np.ndarray) -> dict:
    """
    Runs simulator, then evaluates each seed. Opens agent_count_grid[_sparse]_b.dat and computes
    time-averaged cluster metric and kurtosis of count distribution.
    :param x_norm: L1 and L2 parameters to evaluate
    :return:
    """
    params = denormalize(x_norm)
    write_l1(params)
    write_l2(params)

    sim_output_dir = Path(SIM_OUTPUT_DIR)

    ensemble_kurtosis = []
    ensemble_cluster_metric = []
    for run in range(N_ENSEMBLE):
        clear_old_count_grids(sim_output_dir)
        run_simulator(seed=run)

        grid_path = resolve_count_grid_path(sim_output_dir)
        grids = load_count_grid(grid_path)  # (n_steps, grid_n, grid_n)

        # time-averaged kurtosis: kurtosis of the per-cell count distribution at each
        # timestep, averaged over all timesteps
        #flat_per_t = grids.reshape(grids.shape[0], -1)  # (n_steps, grid_n*grid_n)
        #kurt_per_t = kurtosis(flat_per_t, axis=1, fisher=True)
        #ensemble_kurtosis.append(np.mean(kurt_per_t))

        # time-averaged cluster metric
        cluster_per_t = cluster_metric_per_frame(grids, WORM_COUNT)
        ensemble_cluster_metric.append(np.mean(cluster_per_t))

    return {
        #"kurtosis": float(np.mean(ensemble_kurtosis)),
        "cluster_metric": float(np.mean(ensemble_cluster_metric)),
    }

def fitness_aggregation(metrics: dict) -> float:
    #return -np.mean(metrics["largest_cluster_fractions"])
    return -metrics["cluster_metric"]

def fitness_diffusion(metrics: dict) -> float:
    # minimize neighbors, maximize spread → minimize avg_n + 1/(dist_to_com + eps)
    #return metrics["avg_neighbors"] + 1.0 / (metrics["mean_dist_to_com"] + EPS)
    #return -metrics["diffusion_coefficient"]  # maximize D
    return metrics["cluster_metric"]
# ─── CMA-ES ───────────────────────────────────────────────────────────────────

def run_cmaes(fitness_fn, label: str) -> tuple[np.ndarray, float]:
    x0 = np.zeros(len(PARAM_RANGES))
    n_params = len(PARAM_RANGES)
    es = cma.CMAEvolutionStrategy(
        x0,
        0.6,            # sigma in normalized space — 0.5 is a quarter of [-1,1]
        {
            "maxiter": 200,
            "popsize": 14,  # 4 + floor(3 * ln(24))
            "verbose": 1,
            "tolx":    1e-6,
            "tolfun":  1e-6,
            "bounds": [[-1.0] * n_params, [1.0] * n_params],
        },
    )

    print(f"\n{'='*60}")
    print(f"  Starting CMA-ES: {label}")
    print(f"{'='*60}\n")

    best_params = None
    best_fitness = np.inf
    iteration = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = []

        for i, sol in enumerate(solutions):
            print(f"  [{label}] iter={iteration:03d} cand={i:02d} ...", end=" ", flush=True)
            try:
                metrics = evaluate_count_grid(sol)
                fit     = fitness_fn(metrics)
                print(f"fit={fit:.4f}  cluster metric={metrics['cluster_metric']:.3f}  "
                      f"kurtosis={metrics['kurtosis']:.3f}  ")
            except Exception as e:
                fit = 1e6
                print(f"ERROR ({e})")

            fitnesses.append(fit)

            if fit < best_fitness:
                best_fitness = fit
                best_params  = sol.copy()

        es.tell(solutions, fitnesses)
        best_idx = int(np.argmin(fitnesses))
        print(f"\n  [{label}] iter={iteration:03d} summary | "
              f"best_fit={fitnesses[best_idx]:.4f} | "
              f"mean_fit={np.mean(fitnesses):.4f} | "
              f"sigma={es.sigma:.4f}")

        if iteration % LOG_EVERY == 0:
            # re-evaluate best candidate of this iteration to get the full fractions
            best_sol = solutions[best_idx]
            params   = denormalize(best_sol)
            write_l1(params)
            write_l2(params)
            run_simulator(seed=0)  # fixed seed for logging

            #path = os.path.join(SIM_OUTPUT_DIR, "agent_data.json")
            #with open(path) as f:
            #    data = json.load(f)
            #pos       = np.array(data["positions"])
            #fractions = compute_largest_cluster_fractions(pos)

            log_path = os.path.join(LOG_DIR, f"{label}_iter{iteration:04d}_cluster_fractions.json")
            with open(log_path, "w") as f:
                json.dump({
                    "iteration":  iteration,
                    "fitness":    fitnesses[best_idx],
                    #"fractions":  fractions.tolist(),   # one value per timestep
                }, f, indent=2)
            print(f"  [LOG] Saved cluster fractions → {log_path}")

        iteration += 1
    print(f"  [{label}] Stopped because: {es.stop()}")
    print(f"\n[{label}] Done. Best fitness = {best_fitness:.4f}")
    return best_params, best_fitness


# ─── Result saving ─────────────────────────────────────────────────────────────

def save_result(x_norm: np.ndarray, label: str):
    params = denormalize(x_norm)
    write_l1(params)
    write_l2(params)
    suffix = label.lower()
    for src, dst in [
        (L1_PATH, L1_PATH.replace(".json", f"_{suffix}.json")),
        (L2_PATH, L2_PATH.replace(".json", f"_{suffix}.json")),
    ]:
        with open(src) as f:
            data = json.load(f)
        with open(dst, "w") as f:
            json.dump(data, f, indent=2)
    print(f"[{label}] Saved optimized JSONs with _{suffix} suffix.")

from itertools import combinations

SWEEP_SINGLE = [
    "baseline",
    "l1_only",
    "l2_s0_only",
    "l2_s1_only",
    "l2_s2_only",
    "all",
]

SWEEP_PAIRWISE = [
    "l1_l2_s0",
    "l1_l2_s1",
    "l1_l2_s2",
    "l2_s0_l2_s1",
    "l2_s0_l2_s2",
    "l2_s1_l2_s2",
]

COMPONENTS = ["l1", "l2_s0", "l2_s1", "l2_s2"]

def _blank_l1():
    l1 = {}
    for state in STATES:
        p_off_food = OFF_FOOD[str(state)][str(state)]
        l1[str(state)] = _neutral_entry(-1.0, -1.0, -1.0, p_off_food)
    return l1

def _blank_l2():
    l2 = {str(s): {} for s in STATES}
    for src, dst in TRANSITIONS:
        p_off_food = OFF_FOOD[str(src)][str(dst)]
        l2[str(src)][str(dst)] = _neutral_entry(-1.0, -1.0, -1.0, p_off_food)
    return l2

def write_empty_l1l2():
    with open(L1_PATH, "w") as f:
        json.dump(_blank_l1(), f, indent=2)
    with open(L2_PATH, "w") as f:
        json.dump(_blank_l2(), f, indent=2)

def write_subset_l1l2(active_csv: str, suffix: str):
    """
    active_csv examples:
      "l1"
      "l2_s0"
      "l1,l2_s0"
      "l1,l2_s0,l2_s1"
      "l2_s0,l2_s1,l2_s2"
      "l1,l2_s0,l2_s1,l2_s2"
    """
    active = {x.strip() for x in active_csv.split(",") if x.strip()}

    with open(os.path.join(STATE_EST_DIR, f"l1_{suffix}.json")) as f:
        l1_opt = json.load(f)
    with open(os.path.join(STATE_EST_DIR, f"l2_{suffix}.json")) as f:
        l2_opt = json.load(f)

    l1 = _blank_l1()
    l2 = _blank_l2()

    if "l1" in active:
        for state, entry in l1_opt.items():
            l1[state] = copy.deepcopy(entry)

    for src in STATES:
        token = f"l2_s{src}"
        if token in active:
            for dst, entry in l2_opt[str(src)].items():
                if dst == str(src):
                    continue
                l2[str(src)][dst] = copy.deepcopy(entry)

    with open(L1_PATH, "w") as f:
        json.dump(l1, f, indent=2)
    with open(L2_PATH, "w") as f:
        json.dump(l2, f, indent=2)

# ─── Entry point ──────────────────────────────────────────────────────────────

import argparse

if __name__ == "__main__":
    os.makedirs(STATE_EST_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="PFSM CMA-ES optimizer")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a",  action="store_true", help="optimize aggregation")
    group.add_argument("-d",  action="store_true", help="optimize diffusion")
    group.add_argument("-ad", action="store_true", help="optimize both")
    args = parser.parse_args()

    if args.a or args.ad:
        agg_params, agg_fit = run_cmaes(fitness_aggregation, "AGGREGATION")
        save_result(agg_params, "aggregation")
        print(f"  Aggregation fitness : {agg_fit:.4f}")

    if args.d or args.ad:
        dif_params, dif_fit = run_cmaes(fitness_diffusion, "DIFFUSION")
        save_result(dif_params, "diffusion")
        print(f"  Diffusion fitness   : {dif_fit:.4f}")