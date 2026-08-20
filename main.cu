#include <stdio.h>
#include <curand_kernel.h>
#include "include/json.hpp"
#include "headers/init_joint.h"
#include "headers/init_agents.h"
#include "headers/parameters.h"
#include "headers/init_interaction.h"
#include "headers/agent_update.h"
#include "headers/grid_update.h"
#include "headers/logging.h"
#include <fstream>
#include <iostream>

__global__ void initialize_rng(curandState* states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < WORM_COUNT) {
        // Use a combination of seed, agent ID, and time to ensure unique seeds
        curand_init(seed, id, 0, &states[id]);
    }
}

void get_last_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("CP1: start\n");
    //INPUTS
    const char* joint_distribution_file_name = "/state_estimations/joint_distributions_off_food.json";
    const char *exit_params_filename = "/state_estimations/l1.json";
    const char *transition_params_filename = "/state_estimations/l2.json";
    const char *extracted_params_filename = "/state_estimations/behavior_distributions_off_food.json";
    //OUTPUTS
    const char *agent_log = "/sim/agent_data.json";
    const char *agent_count_grid_log = "/sim/agent_count_grid.dat";
    const char *pheromone_grid_log = "/sim/pheromone_grid.dat";
    const char *param_log = "/sim/simulation_parameters.json";

    auto* positions = new float[WORM_COUNT * N_STEPS * 2]; // Matrix to store positions (x, y) for each agent at each timestep
    auto* states = new int[WORM_COUNT * N_STEPS]; // Matrix to store the state for each agent at each timestep
    int* agent_count_grid, *h_agent_count_grid = new int[GRID_N * GRID_N];
    float* pheromone_grid, *h_pheromone_grid = new float[GRID_N * GRID_N], *new_pheromone;
    StateParams* d_params = nullptr;
    DurationDistributionHost *h_states = new DurationDistributionHost[N_STATES];
    size_t size = WORM_COUNT * sizeof(Agent);
    Agent* d_agents, *h_agents = new Agent[WORM_COUNT];
    TransitionModelHost* h_exit = new TransitionModelHost[N_STATES];
    TransitionModelHost* h_transitions = new TransitionModelHost[N_STATES*N_STATES];
    curandState* d_curand_states, *d_states_grids;
    int* agent_count_grids = new int[N_STEPS * GRID_N * GRID_N];
    float* phi_grids = new float[N_STEPS * GRID_N * GRID_N];
    try {
        cudaMalloc(&d_curand_states, WORM_COUNT * sizeof(curandState));
        cudaMalloc(&d_agents, WORM_COUNT * sizeof(Agent));
        //STATES
        //-duration
        load_state_duration(h_states, extracted_params_filename);
        upload_duration_distributions(h_states);
        cudaDeviceSynchronize();
        get_last_error();
        //-joint speed angle change
        printf("loading distributions:\n");
        load_distributions(joint_distribution_file_name, N_STATES, &d_params);
        cudaDeviceSynchronize();
        get_last_error();
        printf("loading joint distribution: %s\n", joint_distribution_file_name);
        //AGENTS
        initialize_rng<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_curand_states, SEED);
        get_last_error();
        cudaDeviceSynchronize();
        initAgents<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states, SEED,
                                                                               WORM_COUNT);
        printf("Initializing agents\n");
        get_last_error();
        cudaDeviceSynchronize();
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);

        //INIT AGENT GRID
        init_agent_count_grid(h_agent_count_grid, h_agents, &agent_count_grid);
        get_last_error();

        //INIT PHEROMONE GRID
        init_pheromone_grid(h_pheromone_grid, h_agent_count_grid, &pheromone_grid);
        get_last_error();
        cudaMalloc(&new_pheromone, sizeof(float) * GRID_N * GRID_N);

        //L1-L2 FUNCTIONS
        load_exit_data(h_exit, exit_params_filename);
        upload_exit_models(h_exit);
        cudaDeviceSynchronize();
        get_last_error();
        printf("Loading transition data from file...\n");
        load_transition_data(h_transitions, transition_params_filename);
        printf("Uploading transition data to device...\n");
        upload_transition_models(h_transitions);
        cudaDeviceSynchronize();
        get_last_error();

        dim3 gridSize((GRID_N + BLOCK_SIZE - 1) / BLOCK_SIZE, (GRID_N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        for (int i = 0; i < N_STEPS; ++i) {
            //printf("step %d\n", i);
            //printf("logging:\n");
            //LOG POSITIONS AND STATES
            for (int j = 0; j < WORM_COUNT; ++j) {
                positions[(i * WORM_COUNT + j) * 2] = h_agents[j].x;
                positions[(i * WORM_COUNT + j) * 2 + 1] = h_agents[j].y;
                states[i * WORM_COUNT + j] = h_agents[j].state;
            }
            //printf("moving\n");
            //MOVE AGENTS
            moveAgents<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states,
                                                                                   WORM_COUNT, i, d_params,
                                                                                   agent_count_grid);
            get_last_error();
            cudaDeviceSynchronize();
            cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
            //PHEROMONE UPDATE
            //printf("updating grids\n");
            updateGrid<<<gridSize, blockSize>>>(pheromone_grid, agent_count_grid, new_pheromone);
            get_last_error();
            cudaDeviceSynchronize();
            //printf("swapping grids\n");
            std::swap(pheromone_grid, new_pheromone);

            //COPY GRIDS FROM DEVICE TO HOST
            cudaMemcpy(h_agent_count_grid, agent_count_grid, GRID_N * GRID_N * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_pheromone_grid, pheromone_grid, GRID_N * GRID_N * sizeof(float), cudaMemcpyDeviceToHost);

            //LOG PHEROMONE AND COUNT GRIDS
            for (int j = 0; j < GRID_N; j++) {
                for (int k = 0; k < GRID_N; k++) {
                    agent_count_grids[(i * GRID_N + j) * GRID_N + k] = h_agent_count_grid[j * GRID_N + k];
                    phi_grids[(i * GRID_N + j) * GRID_N + k] = h_pheromone_grid[j * GRID_N + k];
                }
            }

            //UPDATE STATES -- ~collective
            updateAgentStateCollective<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents,
                                                                                                   d_curand_states, i,
                                                                                                   WORM_COUNT, d_params,pheromone_grid);
            get_last_error();
            cudaDeviceSynchronize();

        }

        //FINAL LOG WITH POSITIONS AND STATES
        printf("logging\n");
        saveAllDataToJSON(agent_log, positions, states);
        saveSimulationParameters(param_log);
        //LOG GRIDS, this should be sensible as it's O(2000 * 100 * 100) ~ 20 000 000
        //perhaps a sparse implementation? (i,j):x IFF x>0
        log_matrices(agent_count_grids, phi_grids, agent_count_grid_log, pheromone_grid_log, true);

    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    } catch (...) {
        fprintf(stderr, "Fatal error: unknown exception (non-std::exception type)\n");
        return 1;
    }
    return 0;

    free(positions);
    free(agent_count_grid);
    free(h_agent_count_grid);
    delete[] positions;
    delete[] states;
    delete[] agent_count_grids;
    delete[] phi_grids;
    return 0;
}
