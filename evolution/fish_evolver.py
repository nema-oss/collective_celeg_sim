"""
CMA-ES optimization of PFSM logistic parameters (L1 and L2) for a two-fish
(rummy nose tetra) social-conformity task.

Setup per evaluation run
------------------------
  * 2 agents are initialised INIT_DIST apart with the SAME heading.
  * Each agent reads the other's heading and computes a = cos(theta_i - theta_j).
  * `a` feeds the logistic models:
        L1 : probability of EXITING the current state, given a
        L2 : probability of transitioning from state i to state j, given a
  * On entering a new state the agent samples a duration, turn angle and speed
    (handled inside the simulator).

Metrics
-------
  A = 1/2 * sum_i a_i  (i = 1,2), time-averaged   -> mean alignment
  B = fraction of timesteps where both agents are in the same state
  score = W_A * A + W_B * B                        (0.5 / 0.5 by default)

  Social      : maximise score  (CMA minimises -score)
  Anti-social : minimise score  (CMA minimises +score)

Parameter vector (normalised to [-1, 1], then mapped to real ranges):
  L1 : state 2 ONLY x (coeff, intercept, height)                 =  3 values
  L2 : 9 transitions x (coeff, intercept) + height for every
       non-reference transition (reference = (0,0),(1,1),(2,2),
       height fixed to 1.0)                                      = 18 + 6 = 24
  Total 27 values.

!! ASSUMPTIONS TO CHECK (marked with  # ASSUMPTION  below) !!
  - the simulator accepts  --seed  and  --init-dist
  - positions_b.dat is int32 [n_steps, n_agents] header, then float32
    (x, y, angle) per agent per logged frame, time first
  - states_b.dat is int32: [n_steps, n_agents] header, then n_steps*n_agents
    int32 state ids (row-major, time first)
  - the simulator computes  p = height / (1 + exp(-(coeff * a + intercept)))
"""

import argparse
import copy
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import cma
import numpy as np

# --- Paths ------------------------------------------------------------------

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SIM_ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
STATE_EST_DIR  = os.path.join(SIM_ROOT, "state_estimations")
SIM_OUTPUT_DIR = SIM_ROOT
RUN_SCRIPT     = os.path.join(SIM_ROOT, "offline_build_and_run.sh")

L1_PATH       = os.path.join(STATE_EST_DIR, "l1.json")
L2_PATH       = os.path.join(STATE_EST_DIR, "l2.json")
OFF_FOOD_PATH = os.path.join(STATE_EST_DIR, "off_food_transitions.json")

POSITIONS_PATH  = os.path.join(SIM_OUTPUT_DIR, "positions_b.dat")
STATES_PATH     = os.path.join(SIM_OUTPUT_DIR, "states_b.dat")

# --- Experiment configuration -----------------------------------------------

STATES      = [0, 1, 2]
L1_STATES   = [2]           # L1 (exit probability) is only evolved for state 2
TRANSITIONS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
REFERENCE_TRANSITIONS = {(0, 0), (1, 1), (2, 2)}   # height fixed to 1.0

N_AGENTS   = 2
INIT_DIST  = 5.0            # initial inter-agent distance (simulator units)  # ASSUMPTION
N_ENSEMBLE = 5              # seeds per candidate
MAXITER    = 20
POPSIZE    = 14
SIGMA0     = 0.6

W_A, W_B       = 0.5, 0.5   # score = W_A*A + W_B*B
RESCALE_A      = False      # True -> use (A+1)/2 so A and B are both in [0, 1]
EVAL_EVERY     = 1          # files are already thinned by LOG_EVERY in the sim
BURN_IN_FRAC   = 0.0        # fraction of the run to discard at the start

# Parameter ranges. `a` lives in [-1, 1], so coefficients are much larger than
# the 0.027 used for pixel-scale features in the C. elegans script.
COEFF_RANGE     = (-6.0, 6.0)
INTERCEPT_RANGE = (-3.0, 3.0)
L1_HEIGHT_RANGE = (0.0, 1.0)
L2_HEIGHT_RANGE = (0.0, 2.0)

# --- Parameter layout ---------------------------------------------------------

PARAM_RANGES = []
L1_INDEX = {}   # state -> (coeff_idx, intercept_idx, height_idx)
L2_INDEX = {}   # (src, dst) -> (coeff_idx, intercept_idx, height_idx | None)

for st in L1_STATES:
    i = len(PARAM_RANGES)
    PARAM_RANGES += [COEFF_RANGE, INTERCEPT_RANGE, L1_HEIGHT_RANGE]
    L1_INDEX[st] = (i, i + 1, i + 2)

for src, dst in TRANSITIONS:
    i = len(PARAM_RANGES)
    PARAM_RANGES += [COEFF_RANGE, INTERCEPT_RANGE]
    if (src, dst) in REFERENCE_TRANSITIONS:
        L2_INDEX[(src, dst)] = (i, i + 1, None)
    else:
        PARAM_RANGES += [L2_HEIGHT_RANGE]
        L2_INDEX[(src, dst)] = (i, i + 1, i + 2)

N_PARAMS = len(PARAM_RANGES)


def denormalize(x: np.ndarray) -> np.ndarray:
    """Map a normalised [-1, 1] vector to actual parameter ranges."""
    out = np.empty(N_PARAMS)
    for i, (lo, hi) in enumerate(PARAM_RANGES):
        out[i] = lo + (np.clip(x[i], -1.0, 1.0) + 1.0) / 2.0 * (hi - lo)
    return out


# --- JSON writers -------------------------------------------------------------

with open(OFF_FOOD_PATH) as f:
    OFF_FOOD = json.load(f)


def _entry(coeff, intercept, height, p_off_food) -> dict:
    return {
        "p_off_food":      p_off_food,
        "tau":             -1,
        "model_coeff":     float(coeff),
        "model_intercept": float(intercept),
        "model_height":    float(height),
        "mean":            0,
        "std":             1,
        "p_relevant":      0,
        "sign":            1,
    }


def write_l1(params: np.ndarray):
    l1 = {}
    for st in L1_STATES:
        ci, ii, hi = L1_INDEX[st]
        l1[str(st)] = _entry(params[ci], params[ii], params[hi],
                             OFF_FOOD[str(st)][str(st)])
    with open(L1_PATH, "w") as f:
        json.dump(l1, f, indent=2)


def write_l2(params: np.ndarray):
    l2 = {str(s): {} for s in STATES}
    for src, dst in TRANSITIONS:
        ci, ii, hi = L2_INDEX[(src, dst)]
        height = 1.0 if hi is None else params[hi]
        l2[str(src)][str(dst)] = _entry(params[ci], params[ii], height,
                                        OFF_FOOD[str(src)][str(dst)])
    with open(L2_PATH, "w") as f:
        json.dump(l2, f, indent=2)


# --- Simulator interface --------------------------------------------------------

def run_simulator(seed: int):
    cmd = ["bash", RUN_SCRIPT, "--seed", str(seed)]                                   # ASSUMPTION
    result = subprocess.run(cmd, cwd=SIM_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print("[SIM STDOUT]\n", result.stdout)
        print("[SIM STDERR]\n", result.stderr)
        raise RuntimeError(f"Simulator exited with code {result.returncode}")


def clear_old_outputs():
    """Remove stale outputs so a failed run can never be scored on old data."""
    for p in (POSITIONS_PATH, STATES_PATH):
        if os.path.exists(p):
            os.remove(p)


def load_trajectory(path: str = POSITIONS_PATH) -> np.ndarray:
    """-> float array (n_agents, n_frames, 3) holding (x, y, angle).
    positions_b.dat: int32 [n_steps, worm_count] header, then float32 (x, y, angle)
    per agent per logged frame, time first, then agent.                   # ASSUMPTION
    """
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n_steps, n_agents = (int(v) for v in np.fromfile(f, dtype=np.int32, count=2))
        per = (size - 8) / (n_steps * n_agents)
        if per != 12:
            raise ValueError(f"{path}: {per} bytes per agent-step, expected 12 "
                             f"(float32 x, y, angle)")
        flat = np.fromfile(f, dtype=np.float32, count=n_steps * n_agents * 3)
    if n_agents != N_AGENTS:
        raise ValueError(f"{path}: worm_count={n_agents}, expected {N_AGENTS}")
    return flat.reshape(n_steps, n_agents, 3).transpose(1, 0, 2).astype(float)


def load_states(path: str = STATES_PATH) -> np.ndarray:
    """-> int array (n_steps, n_agents)."""                                 # ASSUMPTION
    with open(path, "rb") as f:
        n_steps, n_agents = np.fromfile(f, dtype=np.int32, count=2)
        flat = np.fromfile(f, dtype=np.int32, count=int(n_steps) * int(n_agents))
    if flat.size != n_steps * n_agents:
        raise ValueError(f"{path}: expected {n_steps * n_agents} ints, got {flat.size}")
    return flat.reshape(n_steps, n_agents)


# --- Metrics ----------------------------------------------------------------------

def mean_alignment(traj: np.ndarray) -> float:
    """A = time-average of 1/2 * (a_1 + a_2), a_i = cos(theta_i - theta_j),
    using the logged angle (traj[..., 2])."""
    th = traj[..., 2]                                   # (n_agents, n_frames)
    a_12 = np.cos(th[0] - th[1])
    #a_21 = np.cos(th[1] - th[0])
    a_t = a_12 #0.5 * (a_12 + a_21)
    start = int(len(a_t) * BURN_IN_FRAC)
    return float(a_t[start::EVAL_EVERY].mean())


def state_agreement(states: np.ndarray) -> float:
    """B = fraction of timesteps where both agents are in the same state."""
    start = int(states.shape[0] * BURN_IN_FRAC)
    s = states[start:]
    return float((s[::EVAL_EVERY, 0] == s[::EVAL_EVERY, 1]).mean())


def combine(A: float, B: float) -> float:
    A_eff = 0.5 * (A + 1.0) if RESCALE_A else A
    return W_A * A_eff + W_B * B


# --- Evaluation --------------------------------------------------------------------

def evaluate(x_norm: np.ndarray) -> dict:
    params = denormalize(x_norm)
    write_l1(params)
    write_l2(params)

    As, Bs = [], []
    t_sim = t_eval = 0.0
    for run in range(N_ENSEMBLE):
        clear_old_outputs()
        t0 = time.time()
        run_simulator(seed=run)
        t1 = time.time()
        As.append(mean_alignment(load_trajectory()))
        Bs.append(state_agreement(load_states()))
        t_sim += t1 - t0
        t_eval += time.time() - t1

    A, B = float(np.mean(As)), float(np.mean(Bs))
    scores = [combine(a, b) for a, b in zip(As, Bs)]
    return {"A": A, "B": B, "score": combine(A, B), "score_std": float(np.std(scores)),
            "t_sim": t_sim, "t_eval": t_eval}


def fitness_social(m: dict) -> float:
    return -m["score"]          # CMA-ES minimises -> maximise score


def fitness_antisocial(m: dict) -> float:
    return m["score"]           # minimise score


# --- CMA-ES ---------------------------------------------------------------------------

def _fmt_hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _ckpt_path(label: str) -> str:
    return os.path.join(STATE_EST_DIR, f"cma_state_{label.lower()}.pkl")


def run_cmaes(fitness_fn, label: str, x0=None, sigma0=SIGMA0, resume=False):
    """
    resume=True : continue from the pickled CMA-ES state (mean, sigma, covariance),
                  running MAXITER MORE iterations.
    x0 given    : warm start from a saved best vector with a fresh covariance.
    """
    ckpt = _ckpt_path(label)
    if resume:
        with open(ckpt, "rb") as f:
            es = pickle.loads(f.read())
        es.opts.set({"maxiter": es.countiter + MAXITER})
        print(f"[{label}] Resumed from {ckpt} at iter {es.countiter}, sigma={es.sigma:.4f}")
    else:
        es = _new_es(x0, sigma0)
    return _run_loop(es, fitness_fn, label, ckpt)


def _new_es(x0, sigma0):
    return cma.CMAEvolutionStrategy(
        np.zeros(N_PARAMS) if x0 is None else np.asarray(x0, dtype=float),
        sigma0,
        {
            "maxiter": MAXITER,
            "popsize": POPSIZE,
            "verbose": 1,
            "tolx":    1e-6,
            "tolfun":  1e-6,
            "bounds":  [[-1.0] * N_PARAMS, [1.0] * N_PARAMS],
        },
    )


def _run_loop(es, fitness_fn, label, ckpt):
    print(f"\n{'=' * 60}\n  CMA-ES: {label}  ({N_PARAMS} params)\n{'=' * 60}\n")

    best_params, best_fitness = None, np.inf
    if es.countiter > 0:                       # resumed: keep the best so far
        best_params, best_fitness = es.result.xbest.copy(), float(es.result.fbest)
    iteration, n_eval, total_time = es.countiter, 0, 0.0
    total_est = es.popsize * MAXITER
    run_start = time.time()

    while not es.stop():
        solutions = es.ask()
        fitnesses = []

        for i, sol in enumerate(solutions):
            print(f"  [{label}] iter={iteration:03d} cand={i:02d} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                m = evaluate(sol)
                fit = fitness_fn(m)
                print(f"fit={fit:+.4f}  A={m['A']:+.3f}  B={m['B']:.3f}  "
                      f"score={m['score']:+.3f} (std {m['score_std']:.3f})  "
                      f"sim={m['t_sim']:.1f}s eval={m['t_eval']:.1f}s")
            except Exception as e:
                fit = 1e6
                print(f"ERROR ({e})")
            dt = time.time() - t0
            n_eval += 1
            total_time += dt

            if n_eval == 1:
                print(f"  [{label}] first eval {dt:.1f}s -> rough total "
                      f"{_fmt_hms(dt * total_est)} ({total_est} evals)")

            fitnesses.append(fit)
            if fit < best_fitness:
                best_fitness, best_params = fit, sol.copy()

        es.tell(solutions, fitnesses)
        with open(ckpt, "wb") as f:           # checkpoint every iteration
            f.write(es.pickle_dumps())
        avg = total_time / n_eval
        print(f"\n  [{label}] iter={iteration:03d} | best={min(fitnesses):+.4f} | "
              f"mean={np.mean(fitnesses):+.4f} | sigma={es.sigma:.4f} | "
              f"elapsed={_fmt_hms(time.time() - run_start)} | "
              f"est. remaining={_fmt_hms(avg * (total_est - n_eval))}")
        iteration += 1

    print(f"  [{label}] Stopped because: {es.stop()}")
    print(f"[{label}] Done. Best fitness = {best_fitness:+.4f}")
    return best_params, best_fitness


# --- Result saving ------------------------------------------------------------------------

def save_result(x_norm: np.ndarray, label: str):
    params = denormalize(x_norm)
    write_l1(params)
    write_l2(params)
    suffix = label.lower()
    for src in (L1_PATH, L2_PATH):
        dst = src.replace(".json", f"_{suffix}.json")
        with open(src) as f:
            data = json.load(f)
        with open(dst, "w") as f:
            json.dump(data, f, indent=2)
    # also keep the raw normalised vector for reproducibility / CMA restarts
    np.save(os.path.join(STATE_EST_DIR, f"cma_best_{suffix}.npy"), x_norm)
    print(f"[{label}] Saved l1_{suffix}.json, l2_{suffix}.json and cma_best_{suffix}.npy")


# --- Entry point ----------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(STATE_EST_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Rummy nose tetra PFSM CMA-ES optimizer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s",  action="store_true", help="optimize social (maximize 0.5A+0.5B)")
    group.add_argument("-a",  action="store_true", help="optimize anti-social (minimize 0.5A+0.5B)")
    group.add_argument("-sa", action="store_true", help="optimize both, sequentially")
    parser.add_argument("--resume", action="store_true",
                        help="continue from cma_state_<label>.pkl for MAXITER more iterations")
    parser.add_argument("--warm", action="store_true",
                        help="warm start from cma_best_<label>.npy with a fresh covariance")
    parser.add_argument("--sigma", type=float, default=0.2,
                        help="initial step size for --warm (default 0.2)")
    args = parser.parse_args()

    def _kw(label):
        if args.resume:
            return {"resume": True}
        if args.warm:
            x0 = np.load(os.path.join(STATE_EST_DIR, f"cma_best_{label.lower()}.npy"))
            return {"x0": x0, "sigma0": args.sigma}
        return {}

    if args.s or args.sa:
        p, fit = run_cmaes(fitness_social, "SOCIAL", **_kw("SOCIAL"))
        save_result(p, "social")
        print(f"  Social score      : {-fit:+.4f}")

    if args.a or args.sa:
        p, fit = run_cmaes(fitness_antisocial, "ANTISOCIAL", **_kw("ANTISOCIAL"))
        save_result(p, "antisocial")
        print(f"  Anti-social score : {fit:+.4f}")