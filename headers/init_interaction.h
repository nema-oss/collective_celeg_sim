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

// Compound TRANSITION model: one full N_STATES x N_STATES grid per leg
struct CompoundTransitionModelHost{
    TransitionModelHost aggregation_model[N_STATES*N_STATES];
    TransitionModelHost dispersal_model[N_STATES*N_STATES];
    float lambda; //0: dispersal, 1: aggregation
};

struct CompoundTransitionModel{
    TransitionModel aggregation_model[N_STATES*N_STATES];
    TransitionModel dispersal_model[N_STATES*N_STATES];
    float lambda;
};

// Compound EXIT model: one entry per state per leg
struct CompoundExitModelHost{
    TransitionModelHost aggregation_model[N_STATES];
    TransitionModelHost dispersal_model[N_STATES];
    float lambda;
};

struct CompoundExitModel{
    TransitionModel aggregation_model[N_STATES];
    TransitionModel dispersal_model[N_STATES];
    float lambda;
};

__constant__ TransitionModel d_transition_models[N_STATES*N_STATES], d_transition_models_b[N_STATES*N_STATES];
__constant__ TransitionModel d_exit_models[N_STATES];
__constant__ CompoundTransitionModel d_compound_transition_models;
__constant__ CompoundExitModel d_compound_exit_models;

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
    std::ifstream file(filename);
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

// ---------------- Compound transition model (aggregation/dispersal) ----------------

// Loads /sim/l1_aggregation.json and /sim/l1_diffusion.json into the two
// full N_STATES x N_STATES grids of a CompoundTransitionModelHost.
void load_compound_transition_data(CompoundTransitionModelHost& compound_model, float lambda)
{
    char agg_path[256], disp_path[256];
    if(METRIC_TYPE==0){
        snprintf(agg_path,  sizeof(agg_path),  "/state_estimations/best_dphi_cluster_metric/l2_%s.json", "aggregation");
        snprintf(disp_path, sizeof(disp_path), "/state_estimations/best_dphi_cluster_metric/l2_%s.json", "diffusion");

    } else {
        snprintf(agg_path,  sizeof(agg_path),  "/state_estimations/best_dphi_compound_metric/l2_%s.json", "aggregation");
        snprintf(disp_path, sizeof(disp_path), "/state_estimations/best_dphi_compound_metric/l2_%s.json", "diffusion");
    }
    load_transition_data(compound_model.aggregation_model, agg_path);
    load_transition_data(compound_model.dispersal_model,   disp_path);

    compound_model.lambda = lambda;
}

void upload_compound_transition_model(CompoundTransitionModelHost* h_model)
{
    cudaMemcpyToSymbol(
            d_compound_transition_models,
            h_model,
            sizeof(CompoundTransitionModel)
    );
}

// ---------------- Compound exit model (aggregation/dispersal) ----------------

// Loads /sim/l1_aggregation.json and /sim/l1_diffusion.json into the two
// per-state arrays of a CompoundExitModelHost (only state index 2 populated,
// same convention as load_exit_data).
void load_compound_exit_data(CompoundExitModelHost& compound_model, float lambda)
{
    char agg_path[256], disp_path[256];
    if(METRIC_TYPE==0){
        snprintf(agg_path,  sizeof(agg_path),  "/state_estimations/best_dphi_cluster_metric/l1_%s.json", "aggregation");
        snprintf(disp_path, sizeof(disp_path), "/state_estimations/best_dphi_cluster_metric/l1_%s.json", "diffusion");

    } else {
        snprintf(agg_path,  sizeof(agg_path),  "/state_estimations/best_dphi_compound_metric/l1_%s.json", "aggregation");
        snprintf(disp_path, sizeof(disp_path), "/state_estimations/best_dphi_compound_metric/l1_%s.json", "diffusion");
    }

    load_exit_data(compound_model.aggregation_model, agg_path);
    load_exit_data(compound_model.dispersal_model,   disp_path);

    compound_model.lambda = lambda;
}

void upload_compound_exit_model(CompoundExitModelHost* h_model)
{
    cudaMemcpyToSymbol(
            d_compound_exit_models,
            h_model,
            sizeof(CompoundExitModel)
    );
}

#endif //UNTITLED2_INIT_INTERACTION_H