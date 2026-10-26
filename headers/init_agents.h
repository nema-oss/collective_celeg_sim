//
// Created by nema on 31/07/26.
//

#ifndef UNTITLED2_INIT_AGENTS_H
#define UNTITLED2_INIT_AGENTS_H
#include <cuda_runtime.h>
#include "parameters.h"
#include <random>
#include <fstream>
#include <stdio.h>

struct Agent {
    float x, y, angle, speed;  // Position in 2D space
    float angle_change;
    float previous_angle_change;
    float previous_angle, previous_speed;
    int state;  // State of the agent
    int previous_state;
    int state_duration;
    int initial_state_duration;
    float kappa;
    float phi;
    float run_omega = 0.0f;
    float run_amp = 0.0f;
    float run_bias;
    float previous_phi;
    float prev_raw_sign;
    float bout_sign;
};

// ── host struct to hold the label array on device ────────────────────────────
struct WormLabelSequence {
    int*  d_labels;     // device pointer
    int   length;       // total number of timesteps
};
// ── minimal JSON parser (no external deps) ───────────────────────────────────
// Expects exactly the format above: a JSON array of integers on one line.
static int* load_json_labels(const char* path, int* out_length) {
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    // read whole file
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    char* buf = (char*)malloc(fsize + 1);
    fread(buf, 1, fsize, f);
    buf[fsize] = '\0';
    fclose(f);

    // count commas to pre-allocate
    int n = 1;
    for (long i = 0; i < fsize; i++) if (buf[i] == ',') n++;

    int* arr = (int*)malloc(n * sizeof(int));
    int  idx = 0;
    char* p  = buf;
    while (*p) {
        // skip until digit or minus
        while (*p && (*p < '0' || *p > '9') && *p != '-') p++;
        if (!*p) break;
        arr[idx++] = (int)strtol(p, &p, 10);
    }
    free(buf);

    *out_length = idx;
    return arr;
}

WormLabelSequence load_worm_labels_to_device(char* json_path) {
    WormLabelSequence seq = {NULL, 0};

    int   n_labels  = 0;
    int*  h_labels  = load_json_labels(json_path, &n_labels);
    if (!h_labels) return seq;

    cudaMalloc(&seq.d_labels, n_labels * sizeof(int));
    cudaMemcpy(seq.d_labels, h_labels, n_labels * sizeof(int),
               cudaMemcpyHostToDevice);
    free(h_labels);

    seq.length = n_labels;
    printf("Loaded %d labels from %s\n", n_labels, json_path);
    return seq;
}


// CUDA kernel to initialize the position of each agent
__global__ void initAgents(Agent* agents, curandState* states, unsigned long seed, int worm_count) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < worm_count) {
        curand_init(seed, id, 0, &states[id]);
        agents[id].x = WIDTH / 2;
        agents[id].y = HEIGHT / 2;
        //add random offset of 1mm -- DIFFUSION
        agents[id].x += (curand_uniform(&states[id]) - 0.5f) * sqrt(10.0f);
        agents[id].y += (curand_uniform(&states[id]) - 0.5f) * sqrt(10.0f);
        if(TASK == "aggregation"){
            //initialise in a circle of radius 15 in the center -- AGGREGATION
            //agents[id].x = WIDTH / 2 + cos(2.0f * M_PI * curand_uniform(&states[id])) * 15.0f;
            //agents[id].y = HEIGHT / 2 + sin(2.0f * M_PI * curand_uniform(&states[id])) * 15.0f;
            agents[id].x = curand_uniform(&states[id]) * WIDTH;
            agents[id].y = curand_uniform(&states[id]) * HEIGHT;
        }


        //generate angle in the range [-pi, pi]
        agents[id].angle =(2.0f * curand_uniform(&states[id]) - 1.0f) * M_PI;
        agents[id].speed = 0.0f;
        agents[id].angle_change = 0.0f;
        agents[id].previous_angle = agents[id].angle;
        agents[id].previous_speed = agents[id].speed;
        agents[id].state_duration = 1;
        agents[id].initial_state_duration = 1;
        agents[id].phi = 0.0f;
        agents[id].run_omega = 0.0f;
        agents[id].run_amp = 0.0f;
        agents[id].previous_angle_change = 0.0f;
        agents[id].run_bias =  curand_normal(&states[id]) * 0.005f;//0.1828f ;
        agents[id].kappa =5.0f + (16.0f - 5.0f) * curand_normal(&states[id]); // 3.0f;//
        while(agents[id].kappa<8.0f){
            agents[id].kappa =5.0f + (16.0f - 5.0f) * curand_normal(&states[id]);
        }
        float u =  curand_uniform(&states[id]);
        float period = 6 + (9-6) * u;
        agents[id].run_omega = 2.0f* M_PI / period;//2.0f * M_PI / ((float)d_agent_periods[agent_id - 37]);
        float n = curand_normal(&states[id]);
        agents[id].run_amp =0.02f + (0.5f - 0.02f) * n;//0.22f;// d_agent_amplitudes[agent_id - 37];
        while(agents[id].run_amp<0.02f && agents[id].run_amp>0.5f){
            printf("reinitialising amplitude\n");
            n = curand_normal(&states[id]);
            agents[id].run_amp =0.02f + (0.5f - 0.02f) * n;
        }

        float generated_value = curand_uniform(&states[id]);
        agents[id].state =2;// static_cast<int>(generated_value*(N_STATES));
        agents[id].previous_state = agents[id].state;
        agents[id].prev_raw_sign = 1.0f;
        agents[id].bout_sign = 1.0f;

        if(ANIMAL_TYPE==1){
            agents[id].x = WIDTH  / 2.0f + (id == 0 ? -0.5f : 0.5f) * 5.0f;
            agents[id].y = HEIGHT / 2.0f;
            agents[id].angle          = 0.0f;
            agents[id].previous_angle = 0.0f;

            agents[id].speed          = 0.0f;
            agents[id].previous_speed = 0.0f;
            agents[id].angle_change   = 0.0f;
            agents[id].previous_angle_change = 0.0f;

            // duration 1 -> first update immediately samples a state via L2
            agents[id].state_duration         = 1;
            agents[id].initial_state_duration = 1;

            // independent random initial state per fish (worms always started in state 2).
            // Use `= 2;` instead for the old behaviour.
            int init_state = min((int)(curand_uniform(&states[id]) * N_STATES), N_STATES - 1);
            agents[id].state          = init_state;
            agents[id].previous_state = init_state;
        }


    }
}
#endif //UNTITLED2_INIT_AGENTS_H
