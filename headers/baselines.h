//
// Created by nema on 10/11/26.
//

#ifndef UNTITLED2_BASELINES_H
#define UNTITLED2_BASELINES_H
#pragma once
#include "options.h"

using BaselineKey = std::tuple<int, Condition, int>;  // integration model, condition, worm count

static const std::map<BaselineKey, IntegrationModel> BASELINES = {
        // {w_food, w_phero, mean, std, min, max}; weights are filled in by lookup_baseline
        {{0, Condition::Off, 1000}, {0, 0, 0.260312f, 66.462908f, -654.408203f, 653.531799f}},
        {{0, Condition::Fast, 1000}, {0, 0, 0.259333f, 64.710754f, -734.395203f, 672.425476f}},
        {{0, Condition::Slow, 1000}, {0, 0, 0.261199f, 62.852702f, -889.302979f, 701.860718f}},
        {{0, Condition::Off, 2000}, {0, 0, 0.427543f, 85.592568f, -883.017395f, 893.260864f}},
        {{0, Condition::Fast, 2000}, {0, 0, 0.428530f, 84.300467f, -987.277161f, 802.399597f}},
        {{0, Condition::Slow, 2000}, {0, 0, 0.429557f, 83.012358f, -976.493225f, 878.324097f}},
        {{0, Condition::Off, 4000}, {0, 0, 0.767997f, 115.574179f, -1181.509033f, 1260.167114f}},
        {{0, Condition::Slow, 4000}, {0, 0, 0.768762f, 115.571753f, -1207.787842f, 1147.388672f}},
        {{0, Condition::Fast, 4000}, {0, 0, 0.768205f, 115.597904f, -1202.908813f, 1136.895386f}},
};

static bool lookup_baseline(const Options& opt, IntegrationModel& m) {
    auto it = BASELINES.find({opt.integration_model, opt.condition, opt.worm_count});
    if (it == BASELINES.end()) {
        fprintf(stderr, "no baseline for im=%d cond=%s worms=%d\n",
                opt.integration_model, condition_name(opt.condition), opt.worm_count);
        return false;
    }
    m = it->second;
    m.w_phero = opt.w_phero;
    m.w_food  = 1.0f - opt.w_phero;
    return true;
}
#endif //UNTITLED2_BASELINES_H
