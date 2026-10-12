import json
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_box_dimensions(params_filename="simulation_parameters.json"):
    with open(params_filename, "r") as f:
        params = json.load(f)
    return params["WIDTH"], params["HEIGHT"]

def load_data(positions_filename, states_filename):
    with open(positions_filename, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        n_steps, n_agents = int(header[0]), int(header[1])
        positions = np.fromfile(f, dtype=np.float32, count=n_steps * n_agents * 2)
        positions = positions.reshape(n_steps, n_agents, 2)

    with open(states_filename, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        states = np.fromfile(f, dtype=np.int32, count=n_steps * n_agents)
        states = states.reshape(n_steps, n_agents)

    # binary logs are written [step][agent], the rest of this script
    # expects [agent][step] like the old JSON did
    positions = positions.transpose(1, 0, 2).astype(float)
    states = states.transpose(1, 0)

    return positions, states


def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def calculate_observations(positions, states, dt, width=None, height=None):
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
    abs_dtheta = {0: [], 1: [], 2: []}

    for agent in range(n_agents):

        # ---------------------------------------------------------
        # Calculate displacement between consecutive frames
        # ---------------------------------------------------------
        displacement = np.diff(positions[agent], axis=0)
        if width is not None and height is not None: #periodic wrap
            displacement[:, 0] -= width  * np.round(displacement[:, 0] / width)
            displacement[:, 1] -= height * np.round(displacement[:, 1] / height)
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
        headings = np.arctan2(
            displacement[:, 1],
            displacement[:, 0]
        )

        # ---------------------------------------------------------
        # 24-frame centered heading change
        # ---------------------------------------------------------
        delta_theta = wrap_angle(
            headings[24:] - headings[:-24]
        )

        window_abs_dtheta = np.full_like(headings, np.nan)
        window_abs_dtheta[12:-12] = np.abs(delta_theta)
        for k in range(12, len(headings) - 12):
            frame = k + 1
            state = states[agent, frame]

            if state in abs_dtheta:
                abs_dtheta[state].append(window_abs_dtheta[k])

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
        abs_dtheta[state] = np.asarray(abs_dtheta[state])

    return speeds, angle_changes, abs_dtheta


def calculate_durations(states, dt):
    """
    states shape:
        (n_agents, n_steps)

    For each agent, collapse consecutive frames that share the same
    state into a single "run" and record its length (in frames).

    Example (single agent):
        states = 0 0 0 1 1 2 1 22 0
        runs    = (0, len=3), (1, len=2), (2, len=1), (1, len=1), (22, len=1), (0, len=1)

    Only runs belonging to states 0, 1, 2 are kept (state 22 above,
    or any other unrecognized state id, is ignored, matching the
    behavior of calculate_observations).

    Returns:
        durations[state] -> array of run lengths, in the same time
                             units as dt (frames * dt)
    """

    n_agents, n_steps = states.shape

    durations = {0: [], 1: [], 2: []}

    for agent in range(n_agents):

        agent_states = states[agent]

        run_state = agent_states[0]
        run_length = 1

        for frame in range(1, n_steps):

            if agent_states[frame] == run_state:
                run_length += 1
            else:
                if run_state in durations:
                    durations[run_state].append(run_length * dt)

                run_state = agent_states[frame]
                run_length = 1

        # Flush the final run
        if run_state in durations:
            durations[run_state].append(run_length * dt)

    for state in range(3):
        durations[state] = np.asarray(durations[state])

    return durations


def plot_distributions(speeds, angle_changes, abs_dtheta, durations):
    fig, axes = plt.subplots(
        3, 4,
        figsize=(24, 12)
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

        # =========================================================
        # 24-FRAME ABSOLUTE HEADING CHANGE
        # =========================================================

        ax = axes[state, 2]

        values = abs_dtheta[state]

        if len(values) > 0:
            ax.hist(
                values,
                bins=100,
                density=True,
                range=(0, np.pi),
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

        ax.set_title(f"State {state} — |Δθ| over 0.48 s")
        ax.set_xlabel("|Δθ| [rad]")
        ax.set_ylabel("Density")
        ax.set_xlim(0, np.pi)
        ax.grid(alpha=0.2)

        ax.set_xticks([
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4,
            np.pi
        ])

        ax.set_xticklabels([
            "0",
            r"$\pi/4$",
            r"$\pi/2$",
            r"$3\pi/4$",
            r"$\pi$"
        ])

        # =========================================================
        # DURATION (state run-length)
        # =========================================================

        ax = axes[state, 3]

        values = durations[state]

        if len(values) > 0:
            n_bins = min(100, max(1, int(values.max() - values.min()) + 1))

            ax.hist(
                values,
                bins=n_bins,
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

        ax.set_title(f"State {state} — Duration")
        ax.set_xlabel("Duration [s]")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


def print_statistics(speeds, angle_changes, durations):

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

    print("\n============== DURATION ==============\n")

    for state in range(3):
        values = durations[state]

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
        description="Plot speed, angle-change, and state-duration distributions by C. elegans state."
    )

    parser.add_argument(
        "--dt",
        type=float,
        required=True,
        help="Simulation timestep in seconds"
    )
    parser.add_argument("positions_filename", help="positions_b.dat")
    parser.add_argument("states_filename", help="states_b.dat")
    parser.add_argument(
        "--periodic",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=periodic boundaries (default), 0=non-periodic"
    )
    args = parser.parse_args()
    positions, states = load_data(args.positions_filename, args.states_filename)

    print("Loaded data:")
    print(f"  Agents : {positions.shape[0]}")
    print(f"  Steps  : {positions.shape[1]}")
    print(f"  DT     : {args.dt}")

    width, height = (None, None)
    if args.periodic == 1:
        width, height = load_box_dimensions()

    speeds, angle_changes, abs_dtheta = calculate_observations(
        positions, states, args.dt, width, height
    )

    durations = calculate_durations(
        states,
        args.dt
    )

    print_statistics(
        speeds,
        angle_changes,
        durations
    )

    plot_distributions(
        speeds,
        angle_changes,
        abs_dtheta,
        durations
    )


if __name__ == "__main__":
    main()