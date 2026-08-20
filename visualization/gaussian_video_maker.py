import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches

def load_and_animate_agents_and_grid2(json_file_path, fps, dest_file_path="animation.mp4"):
    # Load JSON data for agents
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    with open("simulation_parameters.json", 'r') as f:
        params = json.load(f)
    N = params["WORM_COUNT"]
    LOGGING_INTERVAL = 1
    N_STEPS = params["N_STEPS"]
    WIDTH = params["WIDTH"]
    HEIGHT = params["HEIGHT"]
    dt = params["DT"]

    sub_states = [[data["states"][agent][timestep] for timestep in range(int(N_STEPS // LOGGING_INTERVAL))] for agent in range(N)]
    sub_states_map = {
        0: "Reversal",
        1: "Turn",
        2: "Run"
    }

    # Prepare the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)

    # Create a list of scatter plot objects for each agent
    scatters = [ax.plot([], [], 'o', color='white', markersize=1.0)[0] for _ in range(N)]
    traces = [ax.plot([], [], '-', color='white', linewidth=0.5)[0] for _ in range(N)]
    position_matrix = [[data["positions"][agent][timestep] for timestep in range(int(N_STEPS // LOGGING_INTERVAL))] for agent in range(N)]

    DIFFUSION_CONSTANT= 0
    odor_x0 = 0
    odor_y0 = 0
    MAX_CONCENTRATION = 0
    ODOR_T0 = 10# parameters['ODOR_T0']
    ODOR_THRESHOLD = 0


    # Function to calculate the evolving Gaussian density
    def calculate_gaussian_density(t, X, Y):
        dx = X - odor_x0
        dy = Y - odor_y0
        return MAX_CONCENTRATION * np.exp(-((dx**2 + dy**2) / (4 * DIFFUSION_CONSTANT * (t + ODOR_T0))))

    # Create a grid of (x, y) coordinates
    x = np.linspace(0, WIDTH, 128)
    y = np.linspace(0, HEIGHT, 128)
    X, Y = np.meshgrid(x, y)

    # Calculate initial Gaussian density
    if MAX_CONCENTRATION>0:
        Z = calculate_gaussian_density(0, X, Y)
        im = ax.imshow(Z, extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='viridis', norm="log", vmin=ODOR_THRESHOLD)
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Chemical Concentration')
    else:
        # set to a blank image if max concentration is 0
        im = ax.imshow(np.zeros_like(X), extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='viridis', vmin=0, vmax=1)
    # Initialization function to set up the scatter plot and grid
    def init():
        for i, scatter in enumerate(scatters):
            scatter.set_data([position_matrix[i][0][0]], [position_matrix[i][0][1]])
        im.set_data(calculate_gaussian_density(0, X, Y))
        ax.set_title(sub_states_map[sub_states[0][0]])
        return scatters + [im]

    # Animation update function
    def update(frame):
        for i, (scatter, trace) in enumerate(zip(scatters, traces)):
            scatter.set_data([position_matrix[i][frame][0]], [position_matrix[i][frame][1]])

            # Handle trace with periodic boundary conditions
            trace_x = []
            trace_y = []
            start_frame = max(0, frame - 20)  # Limit trace to 20 frames
            for j in range(start_frame, frame + 1):
                x_prev, y_prev = position_matrix[i][j - 1] if j > 0 else position_matrix[i][j]
                x_curr, y_curr = position_matrix[i][j]

                # Check for boundary crossings and adjust coordinates
                if abs(x_curr - x_prev) >= WIDTH / 2 or abs(y_curr - y_prev) >= HEIGHT / 2:
                    break

                trace_x.append(x_curr)
                trace_y.append(y_curr)

            trace.set_data(trace_x, trace_y)

        # Update the Gaussian density function
        if MAX_CONCENTRATION>0:
            Z = calculate_gaussian_density(dt*frame*LOGGING_INTERVAL, X, Y)
            im.set_data(Z)
        ax.set_title(sub_states_map[sub_states[0][frame]])
        return scatters + traces + [im]

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=int(N_STEPS // LOGGING_INTERVAL), blit=False
    )
    anim.save(
        dest_file_path + f"N_{N}_LOGGING_INTERVAL_{LOGGING_INTERVAL}_N_STEPS_{N_STEPS}.mp4",
        writer='ffmpeg', fps=fps
    )

# Main execution
if __name__ == "__main__":
    #base_dir = "auto_agents_100_all_data.json"
    base_dir = "agent_data.json"
    load_and_animate_agents_and_grid2(base_dir, fps=30, dest_file_path="")
