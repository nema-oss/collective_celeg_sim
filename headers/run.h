//
// Created by nema on 10/11/26.
//

#ifndef UNTITLED2_RUN_H
#define UNTITLED2_RUN_H
#pragma once
#include "deps.h"
#include "options.h"
#include "paths.h"
#include "baselines.h"
#include "sim_state.h"
#include "setup.h"
#include "step.h"

static double sum_of(const std::vector<float>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0);
}

static bool run_simulation(DeviceState& dev, HostState& host, Streams& out,
                           const Options& opt, const WormLabelSequence& seq) {
    const int    n            = opt.worm_count;
    const size_t agents_bytes = n * sizeof(Agent);
    const bool   dphi_log     = opt.run_idx >= 0;

    if (opt.replay()) {
        updateAgentStateDeterministic<<<blocks_for(n), BLOCK_SIZE>>>(dev.agents, seq.d_labels, seq.length, 0);
        cudaDeviceSynchronize();
    }

    FILE* f_dphi = dphi_log ? open_dphi_run(opt.run_idx) : nullptr;
    bool exhausted = opt.quiet;

    for (int i = 0; i < N_STEPS; ++i) {
        const bool log_now = (i % LOG_EVERY == 0);

        if (log_now) log_step(host, out, n);

        move_agents(dev, n, i);
        if (!launch_ok("moveAgents", i)) return false;

        if (log_now)
            cudaMemcpy(host.agents.data(), dev.agents, agents_bytes, cudaMemcpyDeviceToHost);
        if (dphi_log)
            write_dphi_step(f_dphi, host);

        if (ANIMAL_TYPE == 0) {
            if (!update_grids(dev, i)) return false;
            copy_grids_to_host(dev, host);
        }

        update_states(dev, opt, seq, i);
        if (!launch_ok("updateAgent", i)) return false;

        if (!exhausted)
            exhausted = food_exhausted(host, i);
    }

    if (f_dphi) fclose(f_dphi);
    return true;
}

static size_t free_mb() {
    size_t f, t;
    cudaMemGetInfo(&f, &t);
    return f >> 20;
}

static bool run_once(const Options& opt, double& mean_neighbors,
                     double* food_fraction = nullptr, double* mean_intake = nullptr) {
    //printf("[START] free GPU mem: %zu MB\n", free_mb());
    IntegrationModel m;
    if (!lookup_baseline(opt, m)) return false;

    if (!opt.quiet) make_dirs(opt.out_dir);

    WormLabelSequence seq;
    if (!load_replay_labels(opt, seq)) return false;

    const int n_logged = (N_STEPS + LOG_EVERY - 1) / LOG_EVERY;

    Streams out;
    out.neighbors = open_agent_stream(sink(opt, outfile::NEIGHBORS).c_str(), n_logged, opt.worm_count);
    out.food_grid = open_grid_stream(sink(opt, outfile::FOOD_GRID).c_str(), N_STEPS, GRID_N);
    if (!LEAN_LOG) {
        out.agent_grid = open_grid_stream(sink(opt, outfile::AGENT_COUNT_GRID).c_str(), N_STEPS, GRID_N);
        out.phi_grid   = open_grid_stream(sink(opt, outfile::PHEROMONE_GRID).c_str(), N_STEPS, GRID_N);
        out.states     = open_agent_stream(sink(opt, outfile::STATES).c_str(), n_logged, opt.worm_count);
    }

    HostState host(opt.worm_count);
    DeviceState dev;

    const SimConfig cfg = make_config(opt);
    upload_config(cfg);

    cudaMalloc(&dev.curand, opt.worm_count * sizeof(curandState));
    cudaMalloc(&dev.agents, opt.worm_count * sizeof(Agent));

    load_state_models(dev);
    init_agents(dev, host, opt);
    init_grids(dev, host, opt.worm_count, cfg.food_amount);
    load_transition_models(opt);
    allocate_integration_model(m.mean_dphi, m.std_dphi, m.w_food, m.w_phero, m.min_dphi, m.max_dphi);

    const double food0 = sum_of(host.food_grid);

    const bool ok = run_simulation(dev, host, out, opt, seq);

    cudaMemcpy(host.agents.data(), dev.agents, opt.worm_count * sizeof(Agent), cudaMemcpyDeviceToHost);
    double intake = 0.0;
    for (const Agent& a : host.agents) intake += a.total_food_consumed;

    const double food1 = sum_of(host.food_grid);

    if (!opt.quiet) {
        printf("logging\n");
        saveSimulationParameters(out_path(opt, outfile::PARAMS).c_str(), opt.worm_count);
    }
    free_integration_model();
    out.close();
    dev.release();

    if (!ok) return false;

    const auto& mn = host.mean_neighbors;
    mean_neighbors = std::accumulate(mn.begin() + mn.size() / 2, mn.end(), 0.0) / (mn.size() - mn.size() / 2);
    if (food_fraction)
        *food_fraction = food0 > 0.0 ? (food0 - food1) / food0 : 0.0;
    if (mean_intake)
        *mean_intake = intake / opt.worm_count;
    //printf("[END] free GPU mem: %zu MB\n", free_mb());
    return true;
}
#endif //UNTITLED2_RUN_H
