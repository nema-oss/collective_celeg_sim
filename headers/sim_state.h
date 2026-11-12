//
// Created by nema on 10/11/26.
//

#ifndef UNTITLED2_SIM_STATES_H
#define UNTITLED2_SIM_STATES_H
#pragma once
#include "deps.h"

inline void get_last_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));
}

static int blocks_for(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

static void sync_and_check() {
    cudaDeviceSynchronize();
    get_last_error();
}

static bool launch_ok(const char* kernel, int step) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed at step %d: %s\n", kernel, step, cudaGetErrorString(err));
        return false;
    }
    sync_and_check();
    return true;
}

struct DeviceState {
    curandState* curand           = nullptr;
    Agent*       agents           = nullptr;
    StateParams* params           = nullptr;
    StateParams* params_on_food   = nullptr;
    int*         agent_count_grid = nullptr;
    float*       pheromone_grid   = nullptr;
    float*       new_pheromone    = nullptr;
    float*       food_grid        = nullptr;
    float*       new_food         = nullptr;
    int*         leader_of        = nullptr;
    float*       angle_prev       = nullptr;
    float        kappa            = 0.0f;

    void release() {
        for (void* p : {(void*)curand, (void*)agents, (void*)agent_count_grid,
                        (void*)pheromone_grid, (void*)new_pheromone, (void*)food_grid,
                        (void*)new_food, (void*)leader_of, (void*)angle_prev})
            cudaFree(p);
    }
};

struct HostState {
    std::vector<Agent>  agents;
    std::vector<float>  positions_step;
    std::vector<int>    states_step;
    std::vector<float>  dphi_step;
    std::vector<int>    neighbors_step;
    std::vector<double> mean_neighbors;
    std::vector<int>    agent_count_grid;
    std::vector<float>  pheromone_grid;
    std::vector<float>  food_grid;

    explicit HostState(int worm_count)
            : agents(worm_count),
              positions_step(worm_count * 3),
              states_step(worm_count),
              dphi_step(worm_count),
              neighbors_step(worm_count),
              agent_count_grid(GRID_N * GRID_N),
              pheromone_grid(GRID_N * GRID_N),
              food_grid(GRID_N * GRID_N) {}
};

struct Streams {
    FILE* positions  = nullptr;
    FILE* neighbors  = nullptr;
    FILE* food_grid  = nullptr;
    FILE* states     = nullptr;
    FILE* agent_grid = nullptr;
    FILE* phi_grid   = nullptr;


    void close() {
        for (FILE* f : {positions, states, neighbors, agent_grid, phi_grid, food_grid})
            if (f) fclose(f);
    }
};
#endif //UNTITLED2_SIM_STATES_H
