#ifndef UNTITLED_INIT_ENV_H
#define UNTITLED_INIT_ENV_H
#include <cuda_runtime.h>
#include "parameters.h"
#include <random>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "../include/json.hpp"
#include <limits>
using json = nlohmann::json;


struct DurationDistribution{
    int n_duration_bins;
    int* duration_bins;
    float* duration_prob;
};

struct DurationDistributionHost{
    std::vector<int> duration_bins;
    std::vector<float> duration_prob;
};

void load_state_duration(DurationDistributionHost* states, const char* filename)
{

    std::ifstream file(filename);
    if (!file.is_open())
    {
        printf("Could not open %s\n", filename);
        exit(1);
    }

    json data = json::parse(file);
    printf("Loading state data from %s\n", filename);
    for (int i = 0; i < N_STATES; i++)
    {
        auto& s = states[i];

        auto duration = data[state_ids[i]]["duration"];
        s.duration_bins  = duration["values"].get<std::vector<int>>();
        s.duration_prob  = duration["cumprobs"].get<std::vector<float>>();
    }
}

__constant__ DurationDistribution d_duration_distributions[N_STATES];


void upload_duration_distributions(DurationDistributionHost* h_states)
{
    DurationDistribution h_gpu_states[N_STATES];

    for (int i = 0; i < N_STATES; i++)
    {
        auto& src = h_states[i];
        auto& dst = h_gpu_states[i];

        dst.n_duration_bins = src.duration_bins.size();
        cudaMalloc(&dst.duration_bins,  dst.n_duration_bins * sizeof(int));
        cudaMalloc(&dst.duration_prob,  dst.n_duration_bins * sizeof(float));
        cudaMemcpy(dst.duration_bins,  src.duration_bins.data(),  dst.n_duration_bins*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dst.duration_prob,  src.duration_prob.data(),  dst.n_duration_bins*sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpyToSymbol(d_duration_distributions,
                       h_gpu_states,
                       sizeof(DurationDistribution)*N_STATES);
}

struct JointTable {
    int    n;         // number of observations
    float* obs;       // device ptr: interleaved [speed0, angle0, speed1, angle1, ...]
    float* prob;      // device ptr: alias prob array, length n
    int*   alias;     // device ptr: alias index array, length n
};

struct StateParams {
    // Conditional joint tables (one per duration level)
    int         n_durations;
    int*        durations;   // device ptr: sorted int array, length n_durations
    JointTable* tables;      // device ptr: JointTable array, length n_durations
};

// Host-side helper (mirrors device layout, owns host memory)
struct StateParamsHost {
    int                       n_durations;
    std::vector<int>          durations;
    std::vector<JointTable>   tables;      // each table's ptrs are HOST ptrs here
    // flat storage for obs/prob/alias per table
    std::vector<std::vector<float>> obs_data;
    std::vector<std::vector<float>> prob_data;
    std::vector<std::vector<int>>   alias_data;
};

// ---- helpers ---------------------------------------------------

static void cuda_check(cudaError_t err, const char* ctx) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(ctx) + ": " + cudaGetErrorString(err));
}

template<typename T>
static T* device_alloc_copy(const std::vector<T>& src) {
    T* d_ptr = nullptr;
    cuda_check(cudaMalloc(&d_ptr, src.size() * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(d_ptr, src.data(),
                          src.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return d_ptr;
}

// ---- load joint distributions from JSON ------------------------

static void load_joint(const char* path, std::vector<StateParamsHost>& host_params) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("Cannot open ") + path);

    json j = json::parse(f);
    float global_min_speed = std::numeric_limits<float>::max();
    float global_max_speed = std::numeric_limits<float>::lowest();
    for (auto it = j.items().begin(); it != j.items().end(); ++it) {
        const std::string& s_key = it.key();
        const json& dur_map = it.value();

        int state = std::stoi(s_key);

        StateParamsHost& sp = host_params[state];

        // collect and sort duration levels
        std::vector<int> dur_keys;
        for (auto it2 = dur_map.items().begin(); it2 != dur_map.items().end(); ++it2) {
            const std::string& d_key = it2.key();
            dur_keys.push_back(std::stoi(d_key));
        }
        std::sort(dur_keys.begin(), dur_keys.end());
        sp.n_durations = (int)dur_keys.size();
        sp.durations   = dur_keys;
        sp.tables.resize(sp.n_durations);
        sp.obs_data.resize(sp.n_durations);
        sp.prob_data.resize(sp.n_durations);
        sp.alias_data.resize(sp.n_durations);
        for (int i = 0; i < sp.n_durations; ++i) {
            std::string d_key = std::to_string(dur_keys[i]);
            const json& entry = dur_map[d_key];
            int n = entry["n"].get<int>();
            int actual_obs = entry["obs"].size();


            // obs: array of [speed, angle] pairs → flatten to float*
            sp.obs_data[i].reserve(n * 2);

            for (auto it3 = entry["obs"].begin(); it3 != entry["obs"].end(); ++it3) {
                const auto& pair = *it3;

                // pair[0] = speed
                // pair[1] = angle
                float speed = pair[0].get<float>();
                float angle = pair[1].get<float>();

                sp.obs_data[i].push_back(speed);
                sp.obs_data[i].push_back(angle);

                global_min_speed = std::min(global_min_speed, speed);
                global_max_speed = std::max(global_max_speed, speed);
            }

            sp.prob_data[i]  = entry["prob"].get<std::vector<float>>();
            sp.alias_data[i] = entry["alias"].get<std::vector<int>>();
            if ((int)sp.obs_data[i].size() != 2 * n) {
                throw std::runtime_error(
                        "obs size mismatch for state " +
                        std::to_string(state) +
                        ", duration " +
                        std::to_string(dur_keys[i])
                );
            }

            if ((int)sp.prob_data[i].size() != n) {
                throw std::runtime_error(
                        "prob size mismatch for state " +
                        std::to_string(state) +
                        ", duration " +
                        std::to_string(dur_keys[i])
                );
            }

            if ((int)sp.alias_data[i].size() != n) {
                throw std::runtime_error(
                        "alias size mismatch for state " +
                        std::to_string(state) +
                        ", duration " +
                        std::to_string(dur_keys[i])
                );
            }

            for (int k : sp.alias_data[i]) {
                if (k < 0 || k >= n) {
                    throw std::runtime_error(
                            "INVALID ALIAS INDEX: state=" +
                            std::to_string(state) +
                            " duration=" +
                            std::to_string(dur_keys[i]) +
                            " n=" +
                            std::to_string(n) +
                            " alias=" +
                            std::to_string(k)
                    );
                }
            }
            sp.tables[i].n     = n;
            sp.tables[i].obs   = nullptr;  // filled in upload step
            sp.tables[i].prob  = nullptr;
            sp.tables[i].alias = nullptr;

        }
    }
    printf("\nSpeed observations:\n");
    printf("  min = %.6f\n", global_min_speed);
    printf("  max = %.6f\n", global_max_speed);
}

// ---- upload one state to device --------------------------------

static void upload_state(const StateParamsHost& sp, StateParams& out_d) {
    out_d.n_durations = sp.n_durations;

    // 1. leaf arrays: durations
    out_d.durations = device_alloc_copy(sp.durations);

    // 2. for each table: upload obs/prob/alias, build a host-side JointTable
    //    with device pointers, then upload the array of those structs
    std::vector<JointTable> tables_with_dptrs(sp.n_durations);
    for (int i = 0; i < sp.n_durations; ++i) {
        tables_with_dptrs[i].n     = sp.tables[i].n;
        printf("state table %d: n=%d\n", i, sp.tables[i].n);

        for (int k = 0; k < sp.tables[i].n; ++k) {
            if (sp.obs_data[i][2*k] == 0.0f) {
                printf("HOST ZERO: table=%d k=%d speed=%f angle=%f\n",
                       i,
                       k,
                       sp.obs_data[i][2*k],
                       sp.obs_data[i][2*k+1]);
            }
        }
        tables_with_dptrs[i].obs   = device_alloc_copy(sp.obs_data[i]);
        tables_with_dptrs[i].prob  = device_alloc_copy(sp.prob_data[i]);
        tables_with_dptrs[i].alias = device_alloc_copy(sp.alias_data[i]);
    }

    // 3. now copy the JointTable structs (which contain device ptrs) to device
    out_d.tables = device_alloc_copy(tables_with_dptrs);
}


void load_distributions(const char* joint_path, int n_states, StateParams** d_params_out)   // device array
{
    std::vector<StateParamsHost> host_params(N_STATES);
    load_joint(joint_path, host_params);
    // build device StateParams array
    std::vector<StateParams> h_state_params(n_states);
    for (int s = 0; s < n_states; ++s)
        upload_state(host_params[s], h_state_params[s]);

    *d_params_out = device_alloc_copy(h_state_params);
}
#endif //UNTITLED_INIT_ENV_H
