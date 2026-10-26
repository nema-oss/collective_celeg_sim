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

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    //printf("CP1: start\n");
    //INPUTS
    const char* joint_distribution_file_name = "/state_estimations/fish_data/joint_distributions.json";
    const char *extracted_params_filename = "/state_estimations/fish_data/behavior_distributions.json";
    const char *exit_params_filename = "/state_estimations/l1.json";
    const char *transition_params_filename = "/state_estimations/l2.json";
    //OUTPUTS
    const char *agent_log = "/sim/agent_data.json";
    const char *agent_count_grid_log = "/sim/agent_count_grid.dat";
    const char *pheromone_grid_log = "/sim/pheromone_grid.dat";
    const char *param_log = "/sim/simulation_parameters.json";

    //auto* positions = new float[WORM_COUNT * N_STEPS * 2]; // Matrix to store positions (x, y) for each agent at each timestep
    //auto* states = new int[WORM_COUNT * N_STEPS]; // Matrix to store the state for each agent at each timestep
    int* agent_count_grid, *h_agent_count_grid = new int[GRID_N * GRID_N];
    float* pheromone_grid, *h_pheromone_grid = new float[GRID_N * GRID_N], *new_pheromone;
    int worm_count = WORM_COUNT;
    float lambda = LAMBDA;
    int seed = SEED, agent_id = -1;;
    char target_json[256], label_sequence_filename[256]; //target_json will be where positions are saved
    WormLabelSequence seq;


    //check if seed is provided as argument
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = atoi(argv[++i]);
        if (strcmp(argv[i], "--lambda")==0 && i+1<argc)
            lambda = atof(argv[++i]);
        if (strcmp(argv[i], "--worm-count")==0 & i+1<argc)
            worm_count = atoi(argv[++i]);
        if (strcmp(argv[i], "--agent") == 0 && i+1 < argc)
            agent_id = atoi(argv[++i]);
    }
    if (agent_id < 0 && ENABLE_DETERMINISTIC_REPLAY){
        fprintf(stderr, "agent_id<0 and ENABLE_DETERMINISTIC_REPLAY=true. Label sequence imposed without specified agent id.\n");
        return 1;
    }
    if (agent_id >= 0 && ENABLE_DETERMINISTIC_REPLAY) {
        printf("agent id: %d\n", agent_id);
        snprintf(target_json, sizeof(target_json),
                 "/sim/agent_%d_positions_b.dat", agent_id);
        snprintf(label_sequence_filename, sizeof(label_sequence_filename),
                 "/state_estimations/fish_data/label_sequences/fish_%d_labels.json", agent_id);
        seq = load_worm_labels_to_device(label_sequence_filename);
        if (!seq.d_labels) {
            fprintf(stderr, "Failed to load labels for agent %d\n", agent_id);
            return 1;
        }

    } else {
        snprintf(target_json, sizeof(target_json), "/sim/positions_b.dat");
    }
    float* h_positions_step = new float[worm_count * 3]; //x,y,angle
    int*   h_states_step    = new int[worm_count];
    const int n_logged = (N_STEPS + LOG_EVERY - 1) / LOG_EVERY;
    FILE* f_positions  = open_agent_stream(target_json, n_logged, worm_count);
    FILE* f_states     = open_agent_stream("/sim/states_b.dat",    n_logged, worm_count);
    FILE* f_agent_grid = open_grid_stream(agent_count_grid_log, N_STEPS, GRID_N);
    FILE* f_phi_grid   = open_grid_stream(pheromone_grid_log,   N_STEPS, GRID_N);

    StateParams* d_params = nullptr;
    DurationDistributionHost *h_states = new DurationDistributionHost[N_STATES];
    size_t size = worm_count * sizeof(Agent);
    Agent* d_agents, *h_agents = new Agent[worm_count];
    TransitionModelHost* h_exit = new TransitionModelHost[N_STATES];
    TransitionModelHost* h_transitions = new TransitionModelHost[N_STATES*N_STATES];
    CompoundExitModelHost* h_exit_compound = new CompoundExitModelHost[N_STATES];
    CompoundTransitionModelHost* h_transitions_compound = new CompoundTransitionModelHost[N_STATES*N_STATES];
    curandState* d_curand_states, *d_states_grids;

    try {
        cudaMalloc(&d_curand_states, worm_count * sizeof(curandState));
        cudaMalloc(&d_agents, worm_count * sizeof(Agent));
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

        initialize_rng<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_curand_states, seed);
        get_last_error();
        cudaDeviceSynchronize();
        initAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states, seed,
                                                                               worm_count);
        printf("Initializing agents\n");
        get_last_error();
        cudaDeviceSynchronize();
        cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);

        //INIT AGENT GRID
        init_agent_count_grid(h_agent_count_grid, h_agents, worm_count, &agent_count_grid);
        get_last_error();

        //INIT PHEROMONE GRID
        init_pheromone_grid(h_pheromone_grid, h_agent_count_grid, &pheromone_grid);
        get_last_error();
        cudaMalloc(&new_pheromone, sizeof(float) * GRID_N * GRID_N);

        //L1-L2 FUNCTIONS
        if(TRANSITION_TYPE==0) {
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
        } else {
            load_compound_exit_data(*h_exit_compound, lambda);
            upload_compound_exit_model(h_exit_compound);
            cudaDeviceSynchronize();
            get_last_error();
            printf("Loading compound transition data from file...\n");
            load_compound_transition_data(*h_transitions_compound, lambda);
            printf("Uploading compound transition data from file...\n");
            upload_compound_transition_model(h_transitions_compound);
            cudaDeviceSynchronize();
            get_last_error();
        }

        dim3 gridSize((GRID_N + BLOCK_SIZE - 1) / BLOCK_SIZE, (GRID_N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        // host, allocated once before the timestep loop
        /*float* d_dphi_log;
        cudaMalloc(&d_dphi_log, (size_t)WORM_COUNT * N_STEPS * sizeof(float));
        */
        if(agent_id >= 0 && ENABLE_DETERMINISTIC_REPLAY){
            updateAgentStateDeterministic<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                    d_agents, seq.d_labels, seq.length, 0);
            cudaDeviceSynchronize();
        }
        for (int i = 0; i < N_STEPS; ++i) {
            //printf("step %d\n", i);
            //printf("logging:\n");
            //LOG POSITIONS AND STATES
            if(i%LOG_EVERY ==0){
                for (int j = 0; j < worm_count; ++j) {
                    h_positions_step[j * 3]     = h_agents[j].x;
                    h_positions_step[j * 3 + 1] = h_agents[j].y;
                    h_positions_step[j * 3 + 2] = h_agents[j].angle;
                    h_states_step[j]            = h_agents[j].state;
                }
                fwrite(h_positions_step, sizeof(float), worm_count * 3, f_positions);
                fwrite(h_states_step,    sizeof(int),   worm_count,     f_states);
                if(ANIMAL_TYPE==0) {
                    write_grid_step(f_agent_grid, h_agent_count_grid, GRID_N);
                    write_grid_step(f_phi_grid, h_pheromone_grid, GRID_N);
                }
            }
            //printf("moving\n");
            //MOVE AGENTS

            if(ANIMAL_TYPE==0){
                    moveAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states,worm_count, i, d_params,
                                                                                       agent_count_grid);

            }
            else{
                moveFishAgents<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states,worm_count, i, d_params,
                                                                                       agent_count_grid);

            }
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "moveAgents failed at step %d: %s\n", i, cudaGetErrorString(err));
                return 1;
            }
            get_last_error();
            cudaDeviceSynchronize();

            if(i%LOG_EVERY==0){
                cudaMemcpy(h_agents, d_agents, size, cudaMemcpyDeviceToHost);
            }

            if(ANIMAL_TYPE==0){

                //PHEROMONE UPDATE
                //printf("updating grids\n");

                updateGrid<<<gridSize, blockSize>>>(pheromone_grid, agent_count_grid, new_pheromone);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    fprintf(stderr, "Update grid failed at step %d: %s\n", i, cudaGetErrorString(err));
                    return 1;
                }
                get_last_error();
                cudaDeviceSynchronize();
                //printf("swapping grids\n");
                std::swap(pheromone_grid, new_pheromone);
            }


            //COPY GRIDS FROM DEVICE TO HOST
            if(ANIMAL_TYPE==0) {
                cudaMemcpy(h_agent_count_grid, agent_count_grid, GRID_N * GRID_N * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_pheromone_grid, pheromone_grid, GRID_N * GRID_N * sizeof(float), cudaMemcpyDeviceToHost);
            }
            //LOG PHEROMONE AND COUNT GRIDS
            /*for (int j = 0; j < GRID_N; j++) {
                for (int k = 0; k < GRID_N; k++) {
                    agent_count_grids[(i * GRID_N + j) * GRID_N + k] = h_agent_count_grid[j * GRID_N + k];
                    phi_grids[(i * GRID_N + j) * GRID_N + k] = h_pheromone_grid[j * GRID_N + k];
                }
            }*/

            //printf("updating states...\n");
            //UPDATE STATES -- ~collective
            if(agent_id>=0 && ENABLE_DETERMINISTIC_REPLAY){
                updateAgentStateDeterministic<<<(WORM_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                        d_agents, seq.d_labels, seq.length, i+1);
            } else {
                if(ANIMAL_TYPE==0) {
                    updateAgentStateCollective<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents,
                                                                                                           d_curand_states,
                                                                                                           i,
                                                                                                           worm_count,
                                                                                                           d_params,
                                                                                                           pheromone_grid);
                }
                else{
                    if(TRANSITION_TYPE==-1)
                        updateFishState<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states, worm_count);
                    else
                        updateFishStateCollective<<<(worm_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_agents, d_curand_states,i, worm_count);

                }
            }
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "updateAgent failed at step %d: %s\n", i, cudaGetErrorString(err));
                return 1;
            }
            get_last_error();
            cudaDeviceSynchronize();

        }

        //FINAL LOG WITH POSITIONS AND STATES
        printf("logging\n");
        //saveAllDataToJSON(agent_log, positions, states);
        saveSimulationParameters(param_log, worm_count);
        //LOG GRIDS, this should be sensible as it's O(2000 * 100 * 100) ~ 20 000 000
        //perhaps a sparse implementation? (i,j):x IFF x>0
        //log_matrices(agent_count_grids, phi_grids, agent_count_grid_log, pheromone_grid_log, true);
        /*std::vector<float> h_dphi_log((size_t)WORM_COUNT * N_STEPS);
        cudaMemcpy(h_dphi_log.data(), d_dphi_log, h_dphi_log.size() * sizeof(float), cudaMemcpyDeviceToHost);

        FILE* f = fopen("/sim/dphi_log.bin", "wb");
        fwrite(h_dphi_log.data(), sizeof(float), h_dphi_log.size(), f);
        fclose(f);*/
        fclose(f_positions);
        fclose(f_states);
        fclose(f_agent_grid);
        fclose(f_phi_grid);
        delete[] h_positions_step;
        delete[] h_states_step;
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    } catch (...) {
        fprintf(stderr, "Fatal error: unknown exception (non-std::exception type)\n");
        return 1;
    }

    return 0;
}
