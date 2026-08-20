from evolver import resolve_count_grid_path, load_count_grid, cluster_metric_per_frame, SIM_OUTPUT_DIR, clear_old_count_grids, run_simulator, \
    denormalize, write_l1, write_l2, PARAM_RANGES
from scipy.stats import kurtosis
from pathlib import Path
import json
import numpy as np
N_ENSEMBLE = 100

def random_x_norm(rng=None) -> np.ndarray:
    """
    Generate a random normalized parameter vector in [-1, 1].

    The actual parameter ranges are handled by denormalize().
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.uniform(-1.0, 1.0, size=len(PARAM_RANGES))

sim_output_dir = Path(SIM_OUTPUT_DIR)
with open("simulation_parameters.json", 'r') as f:
    params = json.load(f)
WORM_COUNT = params["WORM_COUNT"]
ensemble_kurtosis = []
ensemble_cluster_metric = []
for run in range(N_ENSEMBLE):
    x_norm = random_x_norm()
    params = denormalize(x_norm)
    write_l1(params)
    write_l2(params)

    clear_old_count_grids(sim_output_dir)
    run_simulator(seed=run)

    grid_path = resolve_count_grid_path(sim_output_dir)
    grids = load_count_grid(grid_path)  # (n_steps, grid_n, grid_n)

    # time-averaged kurtosis: kurtosis of the per-cell count distribution at each
    # timestep, averaged over all timesteps
    flat_per_t = grids.reshape(grids.shape[0], -1)  # (n_steps, grid_n*grid_n)
    kurt_per_t = kurtosis(flat_per_t, axis=1, fisher=True)
    ensemble_kurtosis.append(np.mean(kurt_per_t))

    # time-averaged cluster metric
    cluster_per_t = cluster_metric_per_frame(grids, WORM_COUNT)
    ensemble_cluster_metric.append(np.mean(cluster_per_t))

    print(f"RUN {run}\n\tmean kurtosis: {ensemble_kurtosis[-1]}\n\tmean cluster metric: {ensemble_cluster_metric[-1]}\n")

print(f"baseline:\n kurtosis: \n\tmean: {np.mean(ensemble_kurtosis)} std: {np.std(ensemble_kurtosis)}\n\t max: {np.max(ensemble_kurtosis)} min: {np.min(ensemble_kurtosis)}\n cluster metric: \n\t mean:{np.mean(ensemble_cluster_metric)} std: {np.std(ensemble_cluster_metric)}\n\t max: {np.max(ensemble_cluster_metric)} min: {np.min(ensemble_cluster_metric)}")
