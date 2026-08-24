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
};

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
        agents[id].state = static_cast<int>(generated_value*(N_STATES));
        agents[id].previous_state = agents[id].state;

    }
}
#endif //UNTITLED2_INIT_AGENTS_H
