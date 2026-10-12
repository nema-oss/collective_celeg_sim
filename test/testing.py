import numpy as np

WIDTH = 20
HEIGHT = 20
GRID_N = 200

# Load headers
with open("positions_b.dat", "rb") as f:
    h = np.fromfile(f, dtype=np.int32, count=2)
    n_steps, worm_count = map(int, h)

with open("agent_count_grid.dat", "rb") as f:
    h = np.fromfile(f, dtype=np.int32, count=2)
    n_steps_grid, grid_n = map(int, h)

positions = np.memmap(
    "positions_b.dat",
    dtype=np.float32,
    mode="r",
    offset=8,
    shape=(n_steps, worm_count, 2)
)

grids = np.memmap(
    "agent_count_grid.dat",
    dtype=np.int32,
    mode="r",
    offset=8,
    shape=(n_steps_grid, grid_n, grid_n)
)


def reconstruct(pos):
    ix = np.floor(pos[:, 0] / WIDTH * GRID_N).astype(int)
    iy = np.floor(pos[:, 1] / HEIGHT * GRID_N).astype(int)

    ix = np.clip(ix, 0, GRID_N - 1)
    iy = np.clip(iy, 0, GRID_N - 1)

    result = np.zeros((GRID_N, GRID_N), dtype=np.int32)
    np.add.at(result, (iy, ix), 1)

    return result

def reconstruct_cpp(pos):
    dx_grid = WIDTH / GRID_N
    dy_grid = HEIGHT / GRID_N

    ix = (pos[:, 0] / dx_grid).astype(int)
    iy = (pos[:, 1] / dy_grid).astype(int)

    ix = np.clip(ix, 0, GRID_N - 1)
    iy = np.clip(iy, 0, GRID_N - 1)

    grid = np.zeros((GRID_N, GRID_N), dtype=np.int32)

    # EXACTLY the C++ indexing:
    # agent_count_grid[ix * GRID_N + iy]
    np.add.at(grid, (ix, iy), 1)

    return grid


grid_frame = grids[-1]

for t in [0, 1, 2, 10, 100, 1000, 4998, 4999]:
    reconstructed = reconstruct_cpp(positions[t])
    stored = grids[t]

    diff = np.abs(
        stored.astype(np.int64) -
        reconstructed.astype(np.int64)
    )

    print(
        f"{t:4d}: "
        f"error={diff.sum():6d}, "
        f"max={diff.max()}, "
        f"different={np.count_nonzero(diff):6d}"
    )