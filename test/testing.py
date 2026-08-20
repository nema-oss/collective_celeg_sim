#!/usr/bin/env python3
"""
Diagnostic and validation suite for the C. elegans ABM pheromone field.

Expected files:
    agent_data.json
    agent_count_grid[_sparse]_b.dat
    pheromone_grid[_sparse]_b.dat

The current CUDA logger writes:
    binary header: int32 n_steps, int32 grid_n
    dense: all values as int32/float32
    sparse: for each timestep:
        int32 timestep
        int32 nnz
        nnz * (int32 i, int32 j, value)

Examples:
    python test_pheromone_abm.py --data-dir ./output
    python test_pheromone_abm.py --data-dir ./output \
        --dt 0.01 --secretion 1.0 --evaporation 0.1 --width 10 --height 10
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------

def read_header(f):
    raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("File is too short to contain the 8-byte header.")
    n_steps, grid_n = struct.unpack("<ii", raw)
    return n_steps, grid_n


def load_dense_binary(path: Path, dtype, expected_shape=None):
    with path.open("rb") as f:
        n_steps, grid_n = read_header(f)
        data = np.fromfile(f, dtype=dtype)

    expected = n_steps * grid_n * grid_n
    if data.size != expected:
        raise ValueError(
            f"{path}: expected {expected} values, found {data.size}"
        )

    arr = data.reshape(n_steps, grid_n, grid_n)

    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(
            f"{path}: shape {arr.shape}, expected {expected_shape}"
        )

    return arr, n_steps, grid_n


def load_sparse_binary(path: Path, dtype):
    value_dtype = np.dtype(dtype)
    records = np.dtype([
        ("i", "<i4"),
        ("j", "<i4"),
        ("value", value_dtype),
    ])

    with path.open("rb") as f:
        n_steps, grid_n = read_header(f)

        # Sparse representation is not one globally fixed-size array:
        # each timestep has its own nnz.
        frames = np.zeros((n_steps, grid_n, grid_n), dtype=value_dtype)

        for expected_t in range(n_steps):
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(
                    f"{path}: truncated while reading timestep {expected_t}"
                )

            t, nnz = struct.unpack("<ii", raw)

            if t != expected_t:
                raise ValueError(
                    f"{path}: timestep index {t}, expected {expected_t}"
                )
            if nnz < 0:
                raise ValueError(f"{path}: negative nnz={nnz} at t={t}")

            raw_records = f.read(nnz * records.itemsize)
            if len(raw_records) != nnz * records.itemsize:
                raise ValueError(
                    f"{path}: truncated sparse records at timestep {t}"
                )

            rec = np.frombuffer(raw_records, dtype=records, count=nnz)

            if np.any(rec["i"] < 0) or np.any(rec["i"] >= grid_n):
                raise ValueError(f"{path}: invalid i index at timestep {t}")
            if np.any(rec["j"] < 0) or np.any(rec["j"] >= grid_n):
                raise ValueError(f"{path}: invalid j index at timestep {t}")

            frames[t, rec["i"], rec["j"]] = rec["value"]

        # Check for unexpected trailing data.
        trailing = f.read(1)
        if trailing:
            raise ValueError(f"{path}: unexpected data after final timestep")

    return frames, n_steps, grid_n


def find_grid_file(data_dir: Path, stem: str):
    sparse = data_dir / f"{stem}_sparse_b.dat"
    dense = data_dir / f"{stem}_b.dat"

    if sparse.exists():
        return sparse, True
    if dense.exists():
        return dense, False

    raise FileNotFoundError(
        f"Could not find {sparse.name} or {dense.name} in {data_dir}"
    )


def load_grid(data_dir: Path, stem: str, dtype):
    path, is_sparse = find_grid_file(data_dir, stem)

    if is_sparse:
        arr, n_steps, grid_n = load_sparse_binary(path, dtype)
    else:
        arr, n_steps, grid_n = load_dense_binary(path, dtype)

    return arr, path, n_steps, grid_n


# ---------------------------------------------------------------------
# Agent JSON loading
# ---------------------------------------------------------------------

def load_agent_json(path: Path):
    with path.open() as f:
        data = json.load(f)

    if "positions" not in data:
        raise ValueError("agent_data.json does not contain 'positions'")

    positions_raw = data["positions"]

    # Expected current structure is:
    # positions[agent][t] = [x, y]
    positions = np.asarray(positions_raw, dtype=np.float64)

    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(
            f"Unexpected positions shape {positions.shape}; "
            "expected (n_agents, n_steps, 2)"
        )

    states = None

    # NOTE: current C++ logger appears to write "states" inside each
    # agent_data object but never pushes those values to a top-level
    # "states" structure. Support both a corrected format and the
    # currently intended per-agent format.
    if "states" in data:
        states_raw = data["states"]
        states = np.asarray(states_raw)
    elif "sub_states" in data:
        states_raw = data["sub_states"]
        states = np.asarray(states_raw)

    # If your final JSON is changed to:
    # {"positions": [[...], ...], "states": [[...], ...]}
    # this will work directly. The current C++ code likely does not
    # produce that due to the bug noted below.

    return positions, states, data


# ---------------------------------------------------------------------
# Numerical diagnostics
# ---------------------------------------------------------------------

def print_basic_summary(agent_positions, agent_states, counts, phi):
    print("\n=== BASIC SUMMARY ===")
    print(f"Agents:              {agent_positions.shape[0]}")
    print(f"Agent-position steps: {agent_positions.shape[1]}")
    print(f"Grid steps:          {phi.shape[0]}")
    print(f"Grid size:           {phi.shape[1]} x {phi.shape[2]}")
    print(f"Agent count dtype:   {counts.dtype}")
    print(f"Pheromone dtype:     {phi.dtype}")
    print(f"Max agents/cell:     {counts.max()}")
    print(f"Max pheromone:       {phi.max():.8g}")
    print(f"Min pheromone:       {phi.min():.8g}")
    if agent_states is not None:
        print(f"State array shape:   {agent_states.shape}")


def test_shapes_and_counts(agent_positions, counts, phi):
    ok = True

    if counts.shape[0] != phi.shape[0]:
        print("[FAIL] agent-count and pheromone step counts differ")
        ok = False
    else:
        print("[PASS] agent-count and pheromone step counts agree")

    n_agents, n_pos_steps, _ = agent_positions.shape
    if n_pos_steps != counts.shape[0]:
        print(
            f"[WARN] JSON has {n_pos_steps} position steps but "
            f"grid files have {counts.shape[0]}"
        )
    else:
        print("[PASS] JSON and grid step counts agree")

    if np.any(counts < 0):
        print("[FAIL] negative agent counts")
        ok = False
    else:
        print("[PASS] all agent counts are non-negative")

    return ok


def test_pheromone_finiteness(phi, max_concentration=None, tol=1e-7):
    ok = True

    if not np.all(np.isfinite(phi)):
        bad = np.size(phi) - np.count_nonzero(np.isfinite(phi))
        print(f"[FAIL] pheromone has {bad} NaN/Inf values")
        ok = False
    else:
        print("[PASS] pheromone contains only finite values")

    min_phi = float(phi.min())
    if min_phi < -tol:
        print(f"[FAIL] negative pheromone found: min={min_phi}")
        ok = False
    else:
        print(f"[PASS] pheromone is non-negative: min={min_phi:.6g}")

    if max_concentration is not None:
        max_phi = float(phi.max())
        if max_phi > max_concentration + tol:
            print(
                f"[FAIL] max pheromone {max_phi} exceeds "
                f"MAX_CONCENTRATION={max_concentration}"
            )
            ok = False
        else:
            print("[PASS] pheromone respects MAX_CONCENTRATION")

    return ok


def test_mass_budget(phi, counts, dt, secretion, evaporation, dx, dy,
                     rtol=2e-3, atol=1e-5):
    """
    Tests the global mass balance for:
        dPhi/dt = D laplacian(Phi) - k Phi + s rho

    For periodic or correctly implemented zero-flux boundaries, the
    spatial integral of the Laplacian is zero, so:
        M[t+1] = M[t] + dt * (s * N_agents - k * M[t])

    where M = integral(Phi dA), and total agents = sum(counts).

    IMPORTANT:
    This tests the conservation implied by your discretisation.
    If your CUDA code uses a non-conservative boundary approximation,
    small discrepancies may appear at the edges.
    """
    if dt is None or secretion is None or evaporation is None:
        print("[SKIP] mass-budget test requires --dt, --secretion, --evaporation")
        return None

    area = dx * dy
    mass = phi.sum(axis=(1, 2)) * area
    total_agents = counts.sum(axis=(1, 2)).astype(np.float64)

    predicted = np.empty_like(mass)
    predicted[0] = mass[0]

    # If phi[0] is initialized to zero and your logger records after
    # the first update, the exact alignment may be shifted by one step.
    # Therefore we compare the recurrence directly from stored frame 0.
    for t in range(len(mass) - 1):
        predicted[t + 1] = (
                mass[t]
                + dt * (secretion * total_agents[t] - evaporation * mass[t])
        )

    err = mass - predicted
    scale = np.maximum(np.abs(predicted), atol)
    rel = np.abs(err) / scale

    max_abs = float(np.max(np.abs(err)))
    max_rel = float(np.max(rel))

    passed = np.allclose(mass, predicted, rtol=rtol, atol=atol)

    print(
        f"[{'PASS' if passed else 'FAIL'}] mass budget: "
        f"max_abs_error={max_abs:.6g}, max_relative_error={max_rel:.6g}"
    )

    return passed


def test_equilibrium_for_stationary_density(phi, counts, dt, evaporation,
                                            secretion, dx, dy,
                                            stationary_window=20):
    """
    Useful when the worm configuration is approximately stationary.
    For constant local density, the expected equilibrium is:
        Phi* = (secretion / evaporation) * rho
    globally, for a homogeneous stationary source.

    This test is deliberately weak: it only examines the global
    concentration relation during a late window.
    """
    if dt is None or secretion is None or evaporation is None:
        print("[SKIP] equilibrium test requires --dt, --secretion, --evaporation")
        return None

    if evaporation <= 0:
        print("[SKIP] equilibrium test requires evaporation > 0")
        return None

    if phi.shape[0] < stationary_window:
        print("[SKIP] not enough timesteps for equilibrium test")
        return None

    # Global spatial mean density.
    area = dx * dy
    rho_global = counts.sum(axis=(1, 2)) / (phi.shape[1] * phi.shape[2] * area)

    phi_mean = phi.mean(axis=(1, 2))
    expected = (secretion / evaporation) * rho_global

    # Compare only the final window.
    sl = slice(-stationary_window, None)
    ratio = phi_mean[sl] / np.maximum(expected[sl], 1e-30)

    finite = np.isfinite(ratio)
    if not np.any(finite):
        print("[SKIP] equilibrium ratio is non-finite")
        return None

    ratio_med = float(np.median(ratio[finite]))

    print(
        f"[INFO] late-window global Phi/Phi* ratio: {ratio_med:.4g}"
    )
    print(
        "[INFO] This is meaningful only if the density is approximately "
        "stationary and homogeneous."
    )
    return ratio_med


# ---------------------------------------------------------------------
# Spatial / qualitative diagnostics
# ---------------------------------------------------------------------

def summarize_spreading(phi, dx, dy, n_frames=5):
    """
    For a localized pulse/source, report:
      - peak value
      - center of mass
      - total mass
      - second spatial moment around the COM

    For a static or moving worm source, these are still useful
    diagnostics but should not be interpreted as a pure diffusion
    coefficient estimate.
    """
    n_steps = phi.shape[0]
    frame_ids = np.unique(
        np.linspace(0, n_steps - 1, min(n_frames, n_steps), dtype=int)
    )

    yy, xx = np.indices((phi.shape[1], phi.shape[2]))
    x = (xx + 0.5) * dx
    y = (yy + 0.5) * dy

    print("\n=== PHEROMONE FIELD SNAPSHOTS ===")
    for t in frame_ids:
        f = np.maximum(phi[t], 0.0)
        mass = f.sum() * dx * dy

        if mass > 0:
            xcm = float((f * x).sum() * dx * dy / mass)
            ycm = float((f * y).sum() * dx * dy / mass)
            r2 = (x - xcm) ** 2 + (y - ycm) ** 2
            r2mean = float((f * r2).sum() * dx * dy / mass)
        else:
            xcm = ycm = float("nan")
            r2mean = float("nan")

        print(
            f"t={t:6d}  mass={mass:12.6g}  "
            f"peak={f.max():12.6g}  "
            f"COM=({xcm:8.4f},{ycm:8.4f})  "
            f"<r²>={r2mean:12.6g}"
        )


# ---------------------------------------------------------------------
# Visual diagnostics
# ---------------------------------------------------------------------

def plot_snapshots(agent_positions, counts, phi, dx, dy,
                   output: Path, n_snapshots=6):
    n_steps = min(agent_positions.shape[1], phi.shape[0])

    frames = np.unique(
        np.linspace(0, n_steps - 1, min(n_snapshots, n_steps), dtype=int)
    )

    fig, axes = plt.subplots(
        2, len(frames),
        figsize=(4.0 * len(frames), 7.0),
        squeeze=False
    )

    if len(frames) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, t in enumerate(frames):
        ax = axes[0, col]
        im = ax.imshow(
            phi[t],
            origin="lower",
            extent=[0, phi.shape[2] * dx, 0, phi.shape[1] * dy],
            aspect="equal"
        )
        pos_t = agent_positions[:, min(t, agent_positions.shape[1] - 1), :]
        ax.scatter(pos_t[:, 0], pos_t[:, 1], s=12, facecolors="none", edgecolors="white")
        ax.set_title(f"pheromone + worms, t={t}")
        ax.set_xlabel("x [simulation units]")
        ax.set_ylabel("y [simulation units]")
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax2 = axes[1, col]
        im2 = ax2.imshow(
            counts[t],
            origin="lower",
            extent=[0, counts.shape[2] * dx, 0, counts.shape[1] * dy],
            aspect="equal",
            vmin=0
        )
        ax2.scatter(pos_t[:, 0], pos_t[:, 1], s=12, facecolors="none", edgecolors="white")
        ax2.set_title(f"agent count, t={t}")
        ax2.set_xlabel("x [simulation units]")
        ax2.set_ylabel("y [simulation units]")
        fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"[WROTE] {output}")


def plot_global_timeseries(counts, phi, dx, dy, output: Path):
    area = dx * dy

    total_agents = counts.sum(axis=(1, 2))
    phi_mass = phi.sum(axis=(1, 2)) * area
    phi_mean = phi.mean(axis=(1, 2))
    phi_max = phi.max(axis=(1, 2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(total_agents, label="total agents in count grid")
    ax.plot(phi_mass, label="total pheromone mass [u·area]")
    ax.plot(phi_mean, label="mean pheromone [u]")
    ax.plot(phi_max, label="max pheromone [u]")
    ax.set_xlabel("timestep")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"[WROTE] {output}")


def make_animation(agent_positions, phi, dx, dy, output: Path, stride=1):
    n_steps = min(agent_positions.shape[1], phi.shape[0])
    frame_ids = np.arange(0, n_steps, stride)

    vmax = np.percentile(phi, 99.5)
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        phi[0],
        origin="lower",
        extent=[0, phi.shape[2] * dx, 0, phi.shape[1] * dy],
        aspect="equal",
        vmin=0,
        vmax=vmax
    )
    pos0 = agent_positions[:, 0, :]
    scat = ax.scatter(
        pos0[:, 0], pos0[:, 1],
        s=15,
        facecolors="none",
        edgecolors="white"
    )
    ax.set_xlabel("x [simulation units]")
    ax.set_ylabel("y [simulation units]")
    ax.set_title("Pheromone field + worms")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("pheromone [u]")

    def update(frame_idx):
        t = frame_ids[frame_idx]
        im.set_data(phi[t])
        pos = agent_positions[:, min(t, agent_positions.shape[1] - 1), :]
        scat.set_offsets(pos)
        ax.set_title(f"Pheromone field + worms, t={t}")
        return im, scat

    anim = FuncAnimation(
        fig, update,
        frames=len(frame_ids),
        interval=50,
        blit=True
    )

    # Pillow is widely available; if not, matplotlib will report it.
    anim.save(output, writer="pillow", fps=max(1, int(20 / max(stride, 1))))
    plt.close(fig)
    print(f"[WROTE] {output}")


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--agent-json", default="agent_data.json")
    parser.add_argument(
        "--dt", type=float, default=None,
        help="ABM timestep in seconds"
    )
    parser.add_argument(
        "--secretion", type=float, default=None,
        help="k_sec in u*area/(worm*s)"
    )
    parser.add_argument(
        "--evaporation", type=float, default=None,
        help="k_evaporation in 1/s"
    )
    parser.add_argument(
        "--width", type=float, required=True,
        help="physical width represented by GRID_N cells"
    )
    parser.add_argument(
        "--height", type=float, required=True,
        help="physical height represented by GRID_N cells"
    )
    parser.add_argument(
        "--max-concentration", type=float, default=None
    )
    parser.add_argument(
        "--snapshots", type=int, default=6
    )
    parser.add_argument(
        "--animation-stride", type=int, default=1
    )
    parser.add_argument(
        "--no-animation", action="store_true"
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    json_path = data_dir / args.agent_json

    positions, states, raw_json = load_agent_json(json_path)

    counts, count_path, n_count_steps, count_grid_n = load_grid(
        data_dir, "agent_count_grid", np.int32
    )
    phi, phi_path, n_phi_steps, phi_grid_n = load_grid(
        data_dir, "pheromone_grid", np.float32
    )

    if count_grid_n != phi_grid_n:
        raise ValueError(
            f"Grid size mismatch: count={count_grid_n}, phi={phi_grid_n}"
        )

    if n_count_steps != n_phi_steps:
        raise ValueError(
            f"Step mismatch: count={n_count_steps}, phi={n_phi_steps}"
        )

    grid_n = phi_grid_n
    dx = args.width / grid_n
    dy = args.height / grid_n

    print(f"Loaded positions: {json_path}")
    print(f"Loaded counts:    {count_path}")
    print(f"Loaded phi:       {phi_path}")

    print_basic_summary(positions, states, counts, phi)

    print("\n=== FORMAL TESTS ===")
    test_shapes_and_counts(positions, counts, phi)
    test_pheromone_finiteness(
        phi,
        max_concentration=args.max_concentration
    )
    test_mass_budget(
        phi, counts,
        dt=args.dt,
        secretion=args.secretion,
        evaporation=args.evaporation,
        dx=dx, dy=dy
    )

    test_equilibrium_for_stationary_density(
        phi, counts,
        dt=args.dt,
        evaporation=args.evaporation,
        secretion=args.secretion,
        dx=dx, dy=dy
    )

    summarize_spreading(phi, dx, dy)

    plot_snapshots(
        positions, counts, phi, dx, dy,
        data_dir / "pheromone_snapshots.png",
        n_snapshots=args.snapshots
    )

    plot_global_timeseries(
        counts, phi, dx, dy,
        data_dir / "pheromone_timeseries.png"
    )

    if not args.no_animation:
        make_animation(
            positions, phi, dx, dy,
            data_dir / "pheromone_animation.gif",
            stride=args.animation_stride
        )

    print("\nDone.")


if __name__ == "__main__":
    main()