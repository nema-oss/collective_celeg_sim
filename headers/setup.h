//
// Created by nema on 10/11/26.
//

#ifndef UNTITLED2_SETUP_H
#define UNTITLED2_SETUP_H
#pragma once
#include "paths.h"
#include "sim_state.h"

__global__ void initialize_rng(curandState* states, unsigned long seed, int worm_count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < worm_count)
        curand_init(seed, id, 0, &states[id]);
}

static void load_state_models(DeviceState& dev) {
    static bool loaded = false;
    static StateParams* params = nullptr;
    static StateParams* params_on_food = nullptr;

    if (!loaded) {
        std::vector<DurationDistributionHost> off_food(N_STATES), on_food(N_STATES);

        load_state_duration(off_food.data(), path::DURATION_OFF_FOOD);
        upload_duration_distributions(off_food.data());
        load_state_duration(on_food.data(), path::DURATION_ON_FOOD);
        upload_duration_distributions_on_food(on_food.data());

        load_distributions(path::JOINT_OFF_FOOD, N_STATES, &params);
        load_distributions(path::JOINT_ON_FOOD, N_STATES, &params_on_food);
        sync_and_check();
        loaded = true;
    }

    dev.params         = params;
    dev.params_on_food = params_on_food;
}

static void init_agents(DeviceState& dev, HostState& host, const Options& opt) {
    const int n = opt.worm_count;

    initialize_rng<<<blocks_for(n), BLOCK_SIZE>>>(dev.curand, opt.seed, n);
    sync_and_check();

    initAgents<<<blocks_for(n), BLOCK_SIZE>>>(dev.agents, dev.curand, opt.seed, n);
    //printf("Initializing agents\n");
    sync_and_check();

    cudaMemcpy(host.agents.data(), dev.agents, n * sizeof(Agent), cudaMemcpyDeviceToHost);
}

static void init_grids(DeviceState& dev, HostState& host, int worm_count, float food_amount) {
    init_agent_count_grid(host.agent_count_grid.data(), host.agents.data(), worm_count, &dev.agent_count_grid);
    get_last_error();
    init_food_grid(host.food_grid.data(), &dev.food_grid, food_amount);
    get_last_error();
    init_pheromone_grid(host.pheromone_grid.data(), host.agent_count_grid.data(), &dev.pheromone_grid);
    get_last_error();

    cudaMalloc(&dev.new_pheromone, sizeof(float) * GRID_N * GRID_N);
    cudaMalloc(&dev.new_food,      sizeof(float) * GRID_N * GRID_N);
}

static void load_transition_models(const Options& opt) {
    const char* l1 = opt.l1_path.empty() ? path::EXIT_PARAMS       : opt.l1_path.c_str();
    const char* l2 = opt.l2_path.empty() ? path::TRANSITION_PARAMS : opt.l2_path.c_str();
    if (TRANSITION_TYPE == 0) {
        std::vector<TransitionModelHost> exit_models(N_STATES);
        std::vector<TransitionModelHost> transitions(N_STATES * N_STATES);

        load_exit_data(exit_models.data(), l1);
        upload_exit_models(exit_models.data());
        sync_and_check();

        //printf("Loading transition data from file...\n");
        load_transition_data(transitions.data(), l2);
        //printf("Uploading transition data to device...\n");
        upload_transition_models(transitions.data());
        sync_and_check();
    } else {
        std::vector<CompoundExitModelHost> exit_models(N_STATES);
        std::vector<CompoundTransitionModelHost> transitions(N_STATES * N_STATES);

        load_compound_exit_data(exit_models[0], opt.lambda);
        upload_compound_exit_model(exit_models.data());
        sync_and_check();

        //printf("Loading compound transition data from file...\n");
        load_compound_transition_data(transitions[0], opt.lambda);
        //printf("Uploading compound transition data from file...\n");
        upload_compound_transition_model(transitions.data());
        sync_and_check();
    }
}
#endif //UNTITLED2_SETUP_H
