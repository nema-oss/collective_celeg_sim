//
// Created by nema on 10/11/26.
//

#ifndef UNTITLED2_PATHS_H
#define UNTITLED2_PATHS_H
#pragma once
#include "options.h"

namespace path {
    constexpr const char* JOINT_OFF_FOOD    = "/state_estimations/joint_distributions_off_food.json";
    constexpr const char* DURATION_OFF_FOOD = "/state_estimations/behavior_distributions_off_food.json";
    constexpr const char* JOINT_ON_FOOD     = "/state_estimations/on_food_joint_distributions.json";
    constexpr const char* DURATION_ON_FOOD  = "/state_estimations/on_food_duration_distributions.json";
    constexpr const char* EXIT_PARAMS       = "/state_estimations/l1.json";
    constexpr const char* TRANSITION_PARAMS = "/state_estimations/l2.json";
}

namespace outfile {
    constexpr const char* AGENT_COUNT_GRID = "agent_count_grid.dat";
    constexpr const char* PHEROMONE_GRID   = "pheromone_grid.dat";
    constexpr const char* FOOD_GRID        = "food_grid.dat";
    constexpr const char* STATES           = "states_b.dat";
    constexpr const char* NEIGHBORS        = "n_neighbors.dat";
    constexpr const char* FOOD_EATEN       = "food_eaten.dat";
    constexpr const char* POSITIONS        = "positions_b.dat";
    constexpr const char* PARAMS           = "simulation_parameters.json";
}

static std::string out_path(const Options& opt, const std::string& name) {
    return opt.out_dir + "/" + name;
}

static std::string sink(const Options& opt, const char* name) {
    return opt.quiet ? "/dev/null" : out_path(opt, name);
}

static std::string run_dir(const Options& opt, int run) {
    char w[16], l[16];
    snprintf(w, sizeof(w), "%.2f", opt.w_phero);
    snprintf(l, sizeof(l), "%.2f", opt.lambda);
    return opt.out_dir + "/im" + std::to_string(opt.integration_model) + "/" +
           condition_name(opt.condition) + "/" + std::to_string(opt.worm_count) +
           "w/lambda_" + l + "/wphero_" + w + "/run_" + std::to_string(run);
}

static std::string positions_path(const Options& opt) {
    if (opt.replay())
        return out_path(opt, "agent_" + std::to_string(opt.agent_id) + "_positions_b.dat");
    return out_path(opt, outfile::POSITIONS);
}

static bool load_replay_labels(const Options& opt, WormLabelSequence& seq) {
    if (!opt.replay()) return true;

    printf("agent id: %d\n", opt.agent_id);
    char label_file[256];
    snprintf(label_file, sizeof(label_file),
             "/state_estimations/fish_data/label_sequences/fish_%d_labels.json", opt.agent_id);

    seq = load_worm_labels_to_device(label_file);
    if (!seq.d_labels) {
        fprintf(stderr, "Failed to load labels for agent %d\n", opt.agent_id);
        return false;
    }
    return true;
}

static void make_dirs(const std::string& path) {
    for (size_t pos = 1; pos <= path.size(); ++pos)
        if (pos == path.size() || path[pos] == '/')
            mkdir(path.substr(0, pos).c_str(), 0755);
}

#endif //UNTITLED2_PATHS_H
