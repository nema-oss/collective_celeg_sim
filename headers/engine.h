#ifndef UNTITLED2_ENGINE_H
#define UNTITLED2_ENGINE_H

#include "deps.h"
#include "options.h"
#include "paths.h"
#include "baselines.h"
#include "sim_state.h"
#include "setup.h"
#include "step.h"
#include "run.h"
#include "nsga2.h"

int engine(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const Options opt = parse_options(argc, argv);

    if (opt.agent_id < 0 && ENABLE_DETERMINISTIC_REPLAY) {
        fprintf(stderr, "agent_id<0 and ENABLE_DETERMINISTIC_REPLAY=true. Label sequence imposed without specified agent id.\n");
        return 1;
    }
    if (opt.w_phero < 0.0f || opt.w_phero > 1.0f) {
        fprintf(stderr, "--w-phero must be in [0, 1], got %f\n", opt.w_phero);
        return 1;
    }
    if (opt.w_phero < W_MIN || opt.w_phero > W_MAX ||
        opt.w_food  < W_MIN || opt.w_food  > W_MAX) {
        fprintf(stderr, "weights must be in [%g, %g], got w_phero=%f w_food=%f\n",
                W_MIN, W_MAX, opt.w_phero, opt.w_food);
        return 1;
    }
    IntegrationModel probe;
    if (!lookup_baseline(opt, probe)) return 1;

    const int N_ENSEMBLE = opt.run_idx >= 0 ? 1 : 5;
    std::vector<double> run_means, run_fracs, run_intakes;

    try {
        if (opt.nsga) return nsga::run(opt);

        for (int n = 0; n < N_ENSEMBLE; ++n) {
            Options run_opt = opt;
            run_opt.seed   += n;
            run_opt.out_dir = run_dir(opt, n);

            double mean_neighbors = 0.0, food_fraction = 0.0, mean_intake = 0.0;
            if (!run_once(run_opt, mean_neighbors, &food_fraction, &mean_intake))
                return 1;

            printf("run %d: mean n_neighbors = %.4f, food eaten = %.4f, intake/agent = %.4f\n",
                   n, mean_neighbors, food_fraction, mean_intake);
            run_means.push_back(mean_neighbors);
            run_fracs.push_back(food_fraction);
            run_intakes.push_back(mean_intake);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    } catch (...) {
        fprintf(stderr, "Fatal error: unknown exception (non-std::exception type)\n");
        return 1;
    }

    auto mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    printf("ensemble: n_neighbors = %.4f, food eaten = %.4f, intake/agent = %.4f\n",
           mean(run_means), mean(run_fracs), mean(run_intakes));
    return 0;
}

#endif