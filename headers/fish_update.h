//
// Created by nema on 02/11/26.
//
#include "math.h"
#ifndef UNTITLED2_FISH_UPDATE_H
#define UNTITLED2_FISH_UPDATE_H

__host__ __device__ float kappaFromR(float R)
{
    R = fminf(fmaxf(R, 0.0f), 0.999f);   // clamp: R=1 means kappa=inf
    if (R < 0.53f)
        return 2.0f*R + R*R*R + 5.0f*powf(R, 5.0f)/6.0f;
    if (R < 0.85f)
        return -0.4f + 1.39f*R + 0.43f/(1.0f - R);
    return 1.0f / (R*R*R - 4.0f*R*R + 3.0f*R);
}

__device__ int getLeader(const Agent* agents, int id, int n)
{
    const Agent me = agents[id];
    int best = -1;
    float best_dist2 = 1e30f;

    for (int j = 0; j < n; ++j) {
        if (j == id) continue;

        float dx = agents[j].x - me.x;
        float dy = agents[j].y - me.y;
        dx -= WIDTH  * nearbyintf(dx / WIDTH);
        dy -= HEIGHT * nearbyintf(dy / HEIGHT);

        // Is j in front of me?
        const float q_me = dx * cosf(me.angle) + dy * sinf(me.angle);
        if (q_me <= 0.0f) continue;

        // Is j NOT treating me as being in front of it? (mutual-ahead tie)
        const float q_j = -dx * cosf(agents[j].angle) - dy * sinf(agents[j].angle);
        if (q_j > 0.0f) continue;

        const float d2 = dx*dx + dy*dy;
        if (d2 < best_dist2) { best_dist2 = d2; best = j; }
    }
    return best;   // -1 => no leader, behave as baseline
}

__global__ void computeLeaders(const Agent* agents, int* leader_of, int n)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;
    leader_of[id] = getLeader(agents, id, n);
}

__global__ void snapshotAngles(const Agent* agents, float* angle_prev, int n)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) angle_prev[id] = agents[id].angle;
}

__global__ void synchronizeFollower(
        Agent* agents, const float* angle_prev, const int* leader_of,
        curandState* rng_states, int n, float kappa)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;

    const int leader = leader_of[id];
    if (leader < 0) return;

    curandState rng = rng_states[id];
    agents[id].angle = sampleVonMises(angle_prev[leader], kappa, &rng);
    agents[id].angle_change = 0.0f;
    rng_states[id] = rng;
}

#endif //UNTITLED2_FISH_UPDATE_H
