//
// Created by nema on 10/11/26.
//

#ifndef UNTITLED2_STEP_H
#define UNTITLED2_STEP_H
#pragma once
#include "options.h"
#include "sim_state.h"

static void log_step(HostState& host, Streams& out, int worm_count) {
    double neighbor_sum = 0.0;
    for (int j = 0; j < worm_count; ++j) {
        const Agent& a = host.agents[j];
        host.positions_step[j * 3]     = a.x;
        host.positions_step[j * 3 + 1] = a.y;
        host.positions_step[j * 3 + 2] = a.angle;
        host.states_step[j]            = a.state;
        host.neighbors_step[j]         = a.n_neighbors;
        neighbor_sum += a.n_neighbors;
    }

    if (!LEAN_LOG) {
        fwrite(host.states_step.data(),    sizeof(int),   worm_count,     out.states);
        fwrite(host.positions_step.data(), sizeof(float), worm_count * 3, out.positions);
    }
    fwrite(host.neighbors_step.data(), sizeof(int),   worm_count,     out.neighbors);
    host.mean_neighbors.push_back(neighbor_sum / worm_count);

    if (ANIMAL_TYPE == 0) {
        if(!LEAN_LOG){
            write_grid_step(out.agent_grid, host.agent_count_grid.data(), GRID_N);
            write_grid_step(out.phi_grid,   host.pheromone_grid.data(),   GRID_N);
        }
        write_grid_step(out.food_grid,  host.food_grid.data(),        GRID_N);
    }
}

static void move_agents(DeviceState& dev, int worm_count, int step) {
    const int blocks = blocks_for(worm_count);

    if (ANIMAL_TYPE == 0) {
        moveAgents<<<blocks, BLOCK_SIZE>>>(dev.agents, dev.curand, worm_count, step, dev.params,
                dev.agent_count_grid, dev.food_grid, dev.params_on_food);
    } else {
        computeLeaders<<<blocks, BLOCK_SIZE>>>(dev.agents, dev.leader_of, worm_count);
        snapshotAngles<<<blocks, BLOCK_SIZE>>>(dev.agents, dev.angle_prev, worm_count);
        moveFishAgents<<<blocks, BLOCK_SIZE>>>(dev.agents, dev.curand, worm_count, step, dev.params,
                dev.agent_count_grid, dev.leader_of);
        synchronizeFollower<<<blocks, BLOCK_SIZE>>>(dev.agents, dev.angle_prev, dev.leader_of,
                dev.curand, worm_count, dev.kappa);
    }
}

static bool update_grids(DeviceState& dev, int step) {
    dim3 grid(blocks_for(GRID_N), blocks_for(GRID_N));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    updateGrid<<<grid, block>>>(dev.pheromone_grid, dev.agent_count_grid, dev.new_pheromone);
    //updateFoodGrid<<<grid, block>>>(dev.food_grid, dev.agent_count_grid, dev.new_food);
    if (!launch_ok("Update grid", step)) return false;

    std::swap(dev.pheromone_grid, dev.new_pheromone);
    //std::swap(dev.food_grid, dev.new_food);
    return true;
}

static void copy_grids_to_host(const DeviceState& dev, HostState& host) {
    const size_t cells = GRID_N * GRID_N;
    cudaMemcpy(host.food_grid.data(), dev.food_grid, cells * sizeof(float), cudaMemcpyDeviceToHost);
    if (!LEAN_LOG) {
        cudaMemcpy(host.agent_count_grid.data(), dev.agent_count_grid, cells * sizeof(int),   cudaMemcpyDeviceToHost);
        cudaMemcpy(host.pheromone_grid.data(),   dev.pheromone_grid,   cells * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

static void update_states(DeviceState& dev, const Options& opt, const WormLabelSequence& seq, int step) {
    const int n      = opt.worm_count;
    const int blocks = blocks_for(n);

    if (opt.replay()) {
        updateAgentStateDeterministic<<<blocks, BLOCK_SIZE>>>(dev.agents, seq.d_labels, seq.length, step + 1);
    } else if (ANIMAL_TYPE == 0) {
        if (TRANSITION_TYPE == -1)
            updateAgentStateConstant<<<blocks, BLOCK_SIZE>>>(
                dev.agents, dev.curand, dev.food_grid, dev.params, dev.params_on_food, n, dev.pheromone_grid);
        else
        updateAgentStateCollective<<<blocks, BLOCK_SIZE>>>(
                dev.agents, dev.curand, step, n, dev.params, dev.pheromone_grid, dev.food_grid, d_integration_model);
    } else {
        if (TRANSITION_TYPE == -1)
            updateFishState<<<blocks, BLOCK_SIZE>>>(dev.agents, dev.curand, n);
        else
        updateFishStateCollective<<<blocks, BLOCK_SIZE>>>(
                dev.agents, dev.curand, step, n, opt.lambda, dev.leader_of);
    }
}

static bool food_exhausted(const HostState& host, int t) {
    double sum = 0.0;
    for (float v : host.food_grid)
        sum += v;
    if (sum < 1e-3) {
        printf("FOOD EXHAUSTED AT T=%d\n\tmean food: %f\n\ttotal food remaining: %f (initial: %d; delta: %f)\n",
               t, sum / (GRID_N * GRID_N), sum, GRID_N * GRID_N, sum - GRID_N * GRID_N);
        return true;
    }
    return false;
}

static FILE* open_dphi_run(int run_idx) {
    mkdir("/sim/dphi_baseline_runs", 0755);
    std::string p = "/sim/dphi_baseline_runs/run_" + std::to_string(run_idx) + ".bin";
    FILE* f = fopen(p.c_str(), "wb");
    printf("open: %s\n", p.c_str());
    if (!f) throw std::runtime_error("Cannot open " + p);
    setvbuf(f, NULL, _IOFBF, 1 << 20);
    return f;
}

static void write_dphi_step(FILE* f, HostState& host) {
    const size_t n = host.agents.size();
    for (size_t i = 0; i < n; ++i)
        host.dphi_step[i] = host.agents[i].dphi;
    fwrite(host.dphi_step.data(), sizeof(float), n, f);
}
#endif //UNTITLED2_STEP_H
