//
// Created by nema on 31/07/26.
//

#ifndef UNTITLED2_INIT_INTERACTION_H
#define UNTITLED2_INIT_INTERACTION_H
#include <cuda_runtime.h>
#include "parameters.h"
#include <random>
#include <fstream>
#include "../include/json.hpp"
using json = nlohmann::json;

struct TransitionModelHost
{
    float p_off_food;
    int tau;
    float coeff;
    float intercept;
    float mean, std; //scale params
    int sign;
    float height;
};

struct TransitionModel
{
    float p_off_food;
    int tau;
    float coeff;
    float intercept;
    float mean, std; //scale params
    int sign;
    float height;
};

__constant__ TransitionModel d_transition_models[N_STATES*N_STATES], d_transition_models_b[N_STATES*N_STATES];
__constant__ TransitionModel d_exit_models[N_STATES];

void load_exit_data(TransitionModelHost* exit_models, const char* filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        printf("Could not open %s\n", filename);
        exit(1);
    }
    json data = json::parse(file);
    for (int i = 0; i < N_STATES; i++)
    {
        if (i!=2) continue;
        auto& src = data[state_ids[i]];

        exit_models[i].p_off_food = src["p_off_food"].get<float>();
        exit_models[i].coeff      = src["model_coeff"].get<float>();
        exit_models[i].intercept  = src["model_intercept"].get<float>();
        exit_models[i].height     = src["model_height"].get<float>();
    }
}

void upload_exit_models(TransitionModelHost* h_exit_models)
{
    cudaMemcpyToSymbol(
            d_exit_models,
            h_exit_models,
            sizeof(TransitionModel) * N_STATES
    );
}

void load_transition_data(TransitionModelHost* models, const char* filename)
{
    printf("ENTER load_transition_data, filename ptr = %p\n", (void*)filename);
    fflush(stdout);
    printf("filename = %s\n", filename);   // separate line - if THIS crashes, filename is garbage
    fflush(stdout);
    std::ifstream file(filename);
    printf("OPENING FILE:\n");
    if (!file.is_open())
    {
        printf("Could not open %s\n", filename);
        exit(1);
    }

    json data = json::parse(file);

    for (int i = 0; i < N_STATES; i++)
    {
        auto& src_from = data[state_ids[i]];

        for (int j = 0; j < N_STATES; j++)
        {
            auto& src_to = src_from[state_ids[j]];

            int idx = i * N_STATES + j;
            printf("state %d to state %d\n", i, j);
            models[idx].p_off_food = src_to["p_off_food"].get<float>();
            models[idx].tau        = src_to["tau"].get<int>();
            models[idx].coeff      = src_to["model_coeff"].get<float>();
            models[idx].intercept  = src_to["model_intercept"].get<float>();
            models[idx].mean       = src_to["mean"].get<float>();
            models[idx].std        = src_to["std"].get<float>();
            models[idx].sign       = src_to["sign"].get<int>();
            models[idx].height     = src_to["model_height"].get<float>();
        }
    }
}

void upload_transition_models(TransitionModelHost* h_models)
{
    cudaMemcpyToSymbol(
            d_transition_models,
            h_models,
            sizeof(TransitionModel) * N_STATES * N_STATES
    );
}

#endif //UNTITLED2_INIT_INTERACTION_H
