//
// Created by nema on 31/07/26.
//

#ifndef UNTITLED2_GRID_UPDATE_H
#define UNTITLED2_GRID_UPDATE_H
#include "parameters.h"
#include <cuda_runtime.h>

void init_agent_count_grid(int* agent_count_grid, Agent h_agents[WORM_COUNT], int** d_agent_count_grid){
    for(int i=0; i<GRID_N; i++)
        for(int j=0; j<GRID_N; j++)
            agent_count_grid[i*GRID_N + j] = 0;

    float dx_grid = (float)WIDTH / GRID_N;
    float dy_grid = (float)HEIGHT / GRID_N;

    for(int i=0; i<WORM_COUNT; i++){
        Agent agent = h_agents[i];
        int agent_i = (int)(agent.x / dx_grid);
        int agent_j = (int)(agent.y / dy_grid);
        agent_count_grid[agent_i*GRID_N + agent_j]++;
    }
    cudaMalloc((void**)d_agent_count_grid, sizeof(int) * GRID_N * GRID_N);
    cudaMemcpy(*d_agent_count_grid, agent_count_grid,
               sizeof(int) * GRID_N * GRID_N, cudaMemcpyHostToDevice);
}

//init pheromone grid and upload it to device
void init_pheromone_grid(float* h_phi, int* h_agent_count_grid, float** d_phi)
{
    for (int i = 0; i < GRID_N; i++) {
        for (int j = 0; j < GRID_N; j++) {
            int idx = i * GRID_N + j;
            h_phi[idx] = 0.0f;
        }
    }
    cudaMalloc((void**)d_phi, sizeof(float) * GRID_N * GRID_N);
    cudaMemcpy(*d_phi, h_phi, sizeof(float) * GRID_N * GRID_N, cudaMemcpyHostToDevice);
}

// Function to compute the Laplacian (second derivative)
__device__ float laplacian(const float* grid, int i, int j)
{
    int leftIndex  = (i == 0) ? GRID_N - 1 : i - 1;
    int rightIndex = (i == GRID_N - 1) ? 0 : i + 1;

    int downIndex = (j == 0) ? GRID_N - 1 : j - 1;
    int upIndex   = (j == GRID_N - 1) ? 0 : j + 1;

    float center = grid[i * GRID_N + j];
    float left   = grid[leftIndex  * GRID_N + j];
    float right  = grid[rightIndex * GRID_N + j];
    float down   = grid[i * GRID_N + downIndex];
    float up     = grid[i * GRID_N + upIndex];

    float dx = WIDTH  / (float)GRID_N;
    float dy = HEIGHT / (float)GRID_N;

    return (left + right - 2.0f * center) / (dx * dx)
           + (down + up - 2.0f * center) / (dy * dy);
}

__device__ float zero_flux_laplacian(const float* grid, int i, int j)
{
    // clamp instead of wrap: zero-flux (Neumann) boundaries
    int leftIndex  = (i == 0) ? 0 : i - 1;
    int rightIndex = (i == GRID_N - 1) ? GRID_N - 1 : i + 1;

    int downIndex = (j == 0) ? 0 : j - 1;
    int upIndex   = (j == GRID_N - 1) ? GRID_N - 1 : j + 1;

    float center = grid[i * GRID_N + j];
    float left   = grid[leftIndex  * GRID_N + j];
    float right  = grid[rightIndex * GRID_N + j];
    float down   = grid[i * GRID_N + downIndex];
    float up     = grid[i * GRID_N + upIndex];

    float dx = WIDTH  / (float)GRID_N;
    float dy = HEIGHT / (float)GRID_N;

    return (left + right - 2.0f * center) / (dx * dx)
           + (down + up - 2.0f * center) / (dy * dy);
}


//CUDA kernel to update the grid of the chemical concentration using a reaction-diffusion equation
__global__ void updateGrid(float* phi_grid, int* count_grid, float* new_phi_grid) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < GRID_N && j < GRID_N) {
        float laplacian_value = zero_flux_laplacian(phi_grid, i, j);
        float dx = WIDTH  / (float)GRID_N;
        float dy = HEIGHT / (float)GRID_N;
        float rho = (float)count_grid[i * GRID_N + j] / (dx * dy);
        float new_concentration = phi_grid[i *GRID_N + j] + DT * (PHI_DIFFUSION_CONSTANT * laplacian_value - PHI_EVAPORATION_RATE * phi_grid[i * GRID_N + j] + PHI_SECRETION_RATE * rho);
        if (new_concentration < 0) new_concentration = 0.0f;
        if (new_concentration > MAX_CONCENTRATION) new_concentration = MAX_CONCENTRATION;
        new_phi_grid[i * GRID_N + j] = new_concentration;
    }
}


#endif //UNTITLED2_GRID_UPDATE_H
