def load_and_animate_agents_and_grid2(json_file_path, heatmap_folder_path, single_file_name, fps, dest_file_path="animation.mp4"):
    # Load JSON data for agents
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    parameters = data['parameters']
    N = parameters['N']
    LOGGING_INTERVAL =parameters['LOGGING_INTERVAL']
    N_STEPS = parameters['N_STEPS']
    WIDTH = parameters['WIDTH']
    HEIGHT = parameters['HEIGHT']
    print(parameters)

    # Prepare the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)

    # Create a list of scatter plot objects for each agent
    scatters = [ax.plot([], [], 'o', color='white', markersize=1.0)[0] for _ in range(N)]
    position_matrix = [[data["positions"][agent][timestep] for timestep in range(int(N_STEPS//LOGGING_INTERVAL))] for agent in range(N)]

    # Parse heatmap data from .txt files
    timesteps = N_STEPS // LOGGING_INTERVAL
    list_of_matrices = []
    for t in range(0, N_STEPS, LOGGING_INTERVAL):
        file_path = os.path.join(heatmap_folder_path, f'{single_file_name}_{t}.txt')
        with open(file_path, 'r') as f:
            matrix = np.loadtxt(f)
            list_of_matrices.append(matrix.T)

    grid = list_of_matrices.copy()
    im = ax.imshow(grid[0], extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='viridis')

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Potential')
    dx = WIDTH / 128
    # Calculate the center of the grid
    center_x = 3* WIDTH / 4 - 10*dx   # 10 is half of the rectangle's width
    center_y = HEIGHT / 2 - 10 *dx   # 10 is half of the rectangle's height

    # Create red rectangle in the middle
    rect = patches.Rectangle((center_x, center_y), 20*dx, 20*dx, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # Plot the trajectory of the agents
    if False:
        for i in range(N):
            x_coords = [position_matrix[i][t][0] for t in range(timesteps)]
            y_coords = [position_matrix[i][t][1] for t in range(timesteps)]
            #periodic boundary conditions are present, so the trajectory can be discontinuous
            for j in range(1, timesteps):
                if abs(x_coords[j] - x_coords[j-1]) >= WIDTH:
                    x_coords[j] += np.sign(x_coords[j-1] - x_coords[j]) * WIDTH
                if abs(y_coords[j] - y_coords[j-1]) >= HEIGHT:
                    y_coords[j] += np.sign(y_coords[j-1] - y_coords[j]) * HEIGHT

            ax.plot(x_coords, y_coords, linestyle='-', linewidth=2.5, color='white')

    # Initialization function to set up the scatter plot and grid
    def init():
        for i, scatter in enumerate(scatters):
            scatter.set_data([position_matrix[i][0][0]], [position_matrix[i][0][1]])
        im.set_data(grid[0])
        return scatters + [im]

    # Animation update function
    def update(frame):
        for i, scatter in enumerate(scatters):
            scatter.set_data([position_matrix[i][frame][0]], [position_matrix[i][frame][1]])
        im.set_data(grid[frame])
        return scatters + [im]

    #plot last frame before making the animation
    for i, scatter in enumerate(scatters):
        scatter.set_data([position_matrix[i][-1][0]], [position_matrix[i][-1][1]])
    im.set_data(grid[-1])
    #set chemotactic index as title
    ci = 0
    for i in range(N):
        if WIDTH/2 - 10 * dx <= position_matrix[i][-1][0] <= WIDTH/2 + 10 * dx and HEIGHT/2 - 10 * dx <= position_matrix[i][-1][1] <= HEIGHT/2 + 10 * dx:
            ci += 1
    ci /= N
    ax.set_title(f'Chemotactic Index: {ci:.2f}')
    plt.savefig(dest_file_path + "N_" + str(N) + "_LOGGING_INTERVAL_" + str(LOGGING_INTERVAL) + "_N_STEPS_" + str(N_STEPS) + ".png", dpi=300)
    #plt.show()

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=timesteps, blit=False
    )
    anim.save(dest_file_path + "N_" + str(N) + "_LOGGING_INTERVAL_" + str(LOGGING_INTERVAL) + "_N_STEPS_" + str(N_STEPS) + ".mp4", writer='ffmpeg', fps=fps)


def load_and_animate_agents_and_multiple_heatmaps(json_file_path, primary_heatmap_folder, additional_heatmap_folder, single_file_name1, single_file_name2):
    # Load JSON data for agents
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract parameters
    parameters = data['parameters']
    N = parameters['N']
    LOGGING_INTERVAL = parameters['LOGGING_INTERVAL']
    N_STEPS = parameters['N_STEPS']
    WIDTH = parameters['WIDTH']
    HEIGHT = parameters['HEIGHT']
    print(parameters)

    # Prepare the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)

    # Create a list of scatter plot objects for each agent
    scatters = [ax.plot([], [], 'o', color='magenta', markersize=2)[0] for _ in range(N)]
    position_matrix = [[data[str(agent)][timestep] for timestep in range(int(N_STEPS//LOGGING_INTERVAL))] for agent in range(N)]

    # Parse primary heatmap data from .txt files
    timesteps = N_STEPS // LOGGING_INTERVAL
    primary_matrices = []
    for t in range(0, N_STEPS, LOGGING_INTERVAL):
        file_path = os.path.join(primary_heatmap_folder, f'{single_file_name1}_{t}.txt')
        with open(file_path, 'r') as f:
            matrix = np.loadtxt(f)
            primary_matrices.append(matrix)

    # Parse additional heatmap data from .txt files
    additional_matrices = []
    for t in range(0, N_STEPS, LOGGING_INTERVAL):
        file_path = os.path.join(additional_heatmap_folder, f'{single_file_name2}_{t}.txt')
        with open(file_path, 'r') as f:
            matrix = np.loadtxt(f)
            additional_matrices.append(matrix)

    primary_grid = primary_matrices.copy()
    additional_grid = additional_matrices.copy()
    primary_im = ax.imshow(primary_grid[0], extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='magma', alpha=0.5)
    additional_im = ax.imshow(additional_grid[0], extent=[0, WIDTH, 0, HEIGHT], origin='lower', cmap='viridis', alpha=0.5)

    # Add colorbars
    primary_cbar = fig.colorbar(primary_im, ax=ax)
    primary_cbar.set_label('Odor')
    additional_cbar = fig.colorbar(additional_im, ax=ax)
    additional_cbar.set_label('Attractive Pheromone')
    dx = WIDTH / 128
    # Calculate the center of the grid
    center_x = WIDTH / 2   # 10 is half of the rectangle's width
    center_y = HEIGHT / 2  # 10 is half of the rectangle's height

    # Create red rectangle in the middle
    rect = patches.Rectangle((center_x, center_y), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Initialization function to set up the scatter plot and grid
    def init():
        for i, scatter in enumerate(scatters):
            scatter.set_data([position_matrix[i][0][0]], [position_matrix[i][0][1]])
        primary_im.set_data(primary_grid[0])
        additional_im.set_data(additional_grid[0])
        return scatters + [primary_im, additional_im]

    # Animation update function
    def update(frame):
        for i, scatter in enumerate(scatters):
            scatter.set_data([position_matrix[i][frame][0]], [position_matrix[i][frame][1]])
        primary_im.set_data(primary_grid[frame])
        additional_im.set_data(additional_grid[frame])
        return scatters + [primary_im, additional_im]

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=timesteps, blit=False
    )
    anim.save('animation.mp4', writer='ffmpeg', fps=15)

