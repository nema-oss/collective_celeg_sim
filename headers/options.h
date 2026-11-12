//
// Created by nema on 10/11/26.
//

#ifndef UNTITLED2_OPTIONS_H
#define UNTITLED2_OPTIONS_H
#pragma once
#include "deps.h"

enum class Condition { Off, Slow, Fast };

struct Options {
    int         seed              = SEED;
    float       lambda            = LAMBDA;
    int         worm_count        = WORM_COUNT;
    int         agent_id          = -1;
    int         run_idx           = -1;
    int         integration_model = 0;
    Condition   condition         = Condition::Off;
    float       w_phero           = 0.5f;
    std::string out_dir           = "/sim/integration_comparison";

    bool replay() const { return agent_id >= 0 && ENABLE_DETERMINISTIC_REPLAY; }

    bool        quiet       = false;
    bool        nsga        = false;
    int         pop         = 80;
    int         gens        = 200;
    int         eval_seeds  = 5;
    std::string l1_path;
    std::string l2_path;

    double tol_nu   = 0.01;
    double tol_fc   = 0.005;
    int    patience = 50;
};

static const char* condition_name(Condition c) {
    switch (c) {
        case Condition::Off:  return "off";
        case Condition::Slow: return "slow";
        case Condition::Fast: return "fast";
    }
    return "";
}

static bool parse_condition(const char* s, Condition& c) {
    for (Condition x : {Condition::Off, Condition::Slow, Condition::Fast})
        if (strcmp(s, condition_name(x)) == 0) { c = x; return true; }
    return false;
}

static bool is_flag(int argc, char* argv[], int i, const char* name) {
    return strcmp(argv[i], name) == 0 && i + 1 < argc;
}

static Options parse_options(int argc, char* argv[]) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--nsga") == 0)         opt.nsga       = true;
        else if (is_flag(argc, argv, i, "--pop"))        opt.pop        = atoi(argv[++i]);
        else if (is_flag(argc, argv, i, "--gens"))       opt.gens       = atoi(argv[++i]);
        else if (is_flag(argc, argv, i, "--eval-seeds")) opt.eval_seeds = atoi(argv[++i]);
        else if      (is_flag(argc, argv, i, "--seed"))       opt.seed              = atoi(argv[++i]);
        else if (is_flag(argc, argv, i, "--lambda"))     opt.lambda            = atof(argv[++i]);
        else if (is_flag(argc, argv, i, "--worm-count")) opt.worm_count        = atoi(argv[++i]);
        else if (is_flag(argc, argv, i, "--agent"))      opt.agent_id          = atoi(argv[++i]);
        else if (is_flag(argc, argv, i, "--run-idx"))    opt.run_idx           = atoi(argv[++i]);
        else if (is_flag(argc, argv, i, "--im"))         opt.integration_model = atoi(argv[++i]);
        else if (is_flag(argc, argv, i, "--w-phero"))    opt.w_phero           = atof(argv[++i]);
        else if (is_flag(argc, argv, i, "--cond")) {
            if (!parse_condition(argv[++i], opt.condition)) {
                fprintf(stderr, "unknown condition: %s\n", argv[i]);
                exit(1);
            }
        }
        else if (is_flag(argc, argv, i, "--out"))        opt.out_dir           = argv[++i];
        else if (is_flag(argc, argv, i, "--tol-nu"))   opt.tol_nu   = atof(argv[++i]);
        else if (is_flag(argc, argv, i, "--tol-fc"))   opt.tol_fc   = atof(argv[++i]);
        else if (is_flag(argc, argv, i, "--patience")) opt.patience = atoi(argv[++i]);
    }
    return opt;
}

static SimConfig make_config(const Options& opt) {
    SimConfig c;
    c.integration_model = opt.integration_model;
    switch (opt.condition) {
        case Condition::Off:  c.food_amount = 0.0f; c.food_consumption = 0.0f;          break;
        case Condition::Slow: c.food_amount = 1.0f; c.food_consumption = 0.0055555556f; break;
        case Condition::Fast: c.food_amount = 1.0f; c.food_consumption = 0.0111111111f; break;
    }
    return c;
}

static void upload_config(const SimConfig& c) {
    cudaMemcpyToSymbol(c_cfg, &c, sizeof(SimConfig));
}
#endif //UNTITLED2_OPTIONS_H
