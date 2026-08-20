import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    print("available keys: ", data.keys())
    positions = np.asarray(data["positions"], dtype=float)
    states = np.asarray(data["states"], dtype=int)

    return positions, states


def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_observations(positions, states, dt):
    """
    positions shape:
        (n_agents, n_steps, 2)

    states shape:
        (n_agents, n_steps)

    Returns:
        speeds[state]       -> array of speed observations
        angle_changes[state] -> array of angle-change observations
    """

    n_agents, n_steps, _ = positions.shape

    speeds = {0: [], 1: [], 2: []}
    angle_changes = {0: [], 1: [], 2: []}

    for agent in range(n_agents):

        # ---------------------------------------------------------
        # Calculate displacement between consecutive frames
        # ---------------------------------------------------------
        displacement = np.diff(positions[agent], axis=0)

        # Speed:
        # frame t corresponds to displacement from t-1 -> t
        speed = np.linalg.norm(displacement, axis=1) / dt

        # ---------------------------------------------------------
        # Store speed
        #
        # speed[k] corresponds to frame k+1
        # Therefore:
        #   k=0 -> frame 1
        # ---------------------------------------------------------
        for k in range(len(speed)):
            frame = k + 1
            state = states[agent, frame]

            if state in speeds:
                speeds[state].append(speed[k])

        # ---------------------------------------------------------
        # Calculate heading of every displacement vector
        # ---------------------------------------------------------
        headings = np.arctan2(
            displacement[:, 1],
            displacement[:, 0]
        )

        # ---------------------------------------------------------
        # Angle change
        #
        # headings[k] corresponds to movement:
        #     frame k -> frame k+1
        #
        # So angle change k corresponds to:
        #     heading(k) -> heading(k+1)
        #
        # This requires two displacement vectors, hence starts
        # at frame 2.
        # ---------------------------------------------------------
        for k in range(1, len(headings)):

            angle_change = headings[k] - headings[k - 1]

            # Wrap to [-pi, pi]
            angle_change = wrap_angle(angle_change)

            frame = k + 1
            state = states[agent, frame]

            if state in angle_changes:
                angle_changes[state].append(angle_change)

    # Convert lists to numpy arrays
    for state in range(3):
        speeds[state] = np.asarray(speeds[state])
        angle_changes[state] = np.asarray(angle_changes[state])

    return speeds, angle_changes


def plot_distributions(speeds, angle_changes):
    fig, axes = plt.subplots(
        3, 2,
        figsize=(14, 12)
    )

    for state in range(3):

        # =========================================================
        # SPEED
        # =========================================================

        ax = axes[state, 0]

        values = speeds[state]

        if len(values) > 0:
            ax.hist(
                values,
                bins=100,
                density=True,
                alpha=0.75
            )

            ax.axvline(
                np.mean(values),
                linestyle="--",
                linewidth=2,
                label=f"mean = {np.mean(values):.4g}"
            )

            ax.axvline(
                np.median(values),
                linestyle=":",
                linewidth=2,
                label=f"median = {np.median(values):.4g}"
            )

            ax.legend()

        ax.set_title(f"State {state} — Speed")
        ax.set_xlabel("Speed")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

        # =========================================================
        # ANGLE CHANGE
        # =========================================================

        ax = axes[state, 1]

        values = angle_changes[state]

        if len(values) > 0:
            ax.hist(
                values,
                bins=100,
                density=True,
                range=(-np.pi, np.pi),
                alpha=0.75
            )

            ax.axvline(
                np.mean(values),
                linestyle="--",
                linewidth=2,
                label=f"mean = {np.mean(values):.4g}"
            )

            ax.axvline(
                np.median(values),
                linestyle=":",
                linewidth=2,
                label=f"median = {np.median(values):.4g}"
            )

            ax.legend()

        ax.set_title(f"State {state} — Angle change")
        ax.set_xlabel("Angle change [rad]")
        ax.set_ylabel("Density")
        ax.set_xlim(-np.pi, np.pi)
        ax.grid(alpha=0.2)

        # Make the x-axis easier to interpret
        ax.set_xticks([
            -np.pi,
            -np.pi / 2,
            0,
            np.pi / 2,
            np.pi
        ])

        ax.set_xticklabels([
            r"$-\pi$",
            r"$-\pi/2$",
            "0",
            r"$\pi/2$",
            r"$\pi$"
        ])

    plt.tight_layout()
    plt.show()


def print_statistics(speeds, angle_changes):

    print("\n================ SPEED ================\n")

    for state in range(3):
        values = speeds[state]

        print(f"State {state}:")
        print(f"  N      = {len(values)}")

        if len(values) > 0:
            print(f"  Min    = {np.min(values):.6g}")
            print(f"  Max    = {np.max(values):.6g}")
            print(f"  Mean   = {np.mean(values):.6g}")
            print(f"  Median = {np.median(values):.6g}")

        print()

    print("\n============ ANGLE CHANGE ============\n")

    for state in range(3):
        values = angle_changes[state]

        print(f"State {state}:")
        print(f"  N      = {len(values)}")

        if len(values) > 0:
            print(f"  Min    = {np.min(values):.6g}")
            print(f"  Max    = {np.max(values):.6g}")
            print(f"  Mean   = {np.mean(values):.6g}")
            print(f"  Median = {np.median(values):.6g}")

        print()


def main():

    parser = argparse.ArgumentParser(
        description="Plot speed and angle-change distributions by C. elegans state."
    )

    parser.add_argument(
        "filename",
        help="JSON file produced by saveAllDataToJSON()"
    )

    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="Simulation timestep in seconds"
    )

    args = parser.parse_args()

    positions, states = load_data(args.filename)

    print("Loaded data:")
    print(f"  Agents : {positions.shape[0]}")
    print(f"  Steps  : {positions.shape[1]}")
    print(f"  DT     : {args.dt}")

    speeds, angle_changes = calculate_observations(
        positions,
        states,
        args.dt
    )

    print_statistics(
        speeds,
        angle_changes
    )

    plot_distributions(
        speeds,
        angle_changes
    )


if __name__ == "__main__":
    main()