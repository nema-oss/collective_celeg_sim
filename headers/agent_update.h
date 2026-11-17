//
// Created by nema on 31/07/26.
//

#ifndef UNTITLED2_AGENT_UPDATE_H
#define UNTITLED2_AGENT_UPDATE_H
#include "parameters.h"
#include <cuda_runtime.h>
#include <random>
#include <limits>
#include <cmath>
#include <stdio.h>

__device__ float wrapAngle(float angle)
{
    angle = fmodf(angle + M_PI, 2.0f * M_PI);
    if (angle < 0.0f)
        angle += 2.0f * M_PI;
    return angle - M_PI;
}


// Assumes states 0, 1, 2 and exactly two fish.
// Returns follower ID: 0 or 1; -1 if no unique follower exists.
__device__ int getFollower(const Agent* agents)
{
    const Agent a0 = agents[0];
    const Agent a1 = agents[1];

    float dx = a1.x - a0.x;
    float dy = a1.y - a0.y;

    // Replace WIDTH and HEIGHT with your actual periodic box dimensions.
    dx -= WIDTH  * nearbyintf(dx / WIDTH);
    dy -= HEIGHT * nearbyintf(dy / HEIGHT);

    // Is fish 1 ahead of fish 0?
    const float q0 =
            dx * cosf(a0.angle) + dy * sinf(a0.angle);

    // Is fish 0 ahead of fish 1?
    const float q1 =
            -dx * cosf(a1.angle) - dy * sinf(a1.angle);

    const bool fish1_ahead = q0 > 0.0f;
    const bool fish0_ahead = q1 > 0.0f;

    if (fish1_ahead && !fish0_ahead) return 0;
    if (fish0_ahead && !fish1_ahead) return 1;

    return -1;
}

__device__ inline float sign(float x)
{
    return (x > 0.0f) - (x < 0.0f);
}

__device__ int select_next_state(
        float* probabilities,
        curandState* rng,
        int num_states)
{
    float r = curand_uniform(rng);

    float cumulative = 0.0f;

    for (int i = 0; i < num_states; i++)
    {
        cumulative += probabilities[i];

        if (r <= cumulative)
            return i;
    }

    // fallback if rounding errors occur
    return num_states - 1;
}

// ---- Alias draw from a JointTable ------------------------------------
__device__ void alias_draw(
        const JointTable* table,
        curandState* rng,
        float* out_speed,
        float* out_angle
) {
    assert(table != nullptr);
    assert(table->n > 0);

    float r = curand_uniform(rng);
    int i = (int)(r * table->n);

    // Protect against the upper endpoint.
    if (i >= table->n)
        i = table->n - 1;

    assert(i >= 0 && i < table->n);

    float u = curand_uniform(rng);

    int idx = (u < table->prob[i])
              ? i
              : table->alias[i];

    assert(idx >= 0 && idx < table->n);

    float speed = table->obs[idx * 2];
    float angle = table->obs[idx * 2 + 1];

    /*if (angle == 0.0f) {
        printf("ZERO SPEED: n=%d i=%d idx=%d prob=%f alias=%d\n",
               table->n,
               i,
               idx,
               table->prob[i],
               table->alias[i]);
    }*/

    *out_speed = speed;
    *out_angle = angle;
}

// ---- Interpolated draw (point iv) ------------------------------------
__device__ void draw_speed_angle(const StateParams* sp, int t_star,
                                 curandState* rng,
                                 float* out_speed, float* out_angle) {
    // Binary search for t0, t1 bracketing t_star
    assert(sp->n_durations > 0);
    int lo = 0, hi = sp->n_durations - 1;

    // exact match
    // (linear scan is fine for small n_durations; replace with bsearch if needed)
    for (int i = 0; i < sp->n_durations; i++) {
        if (sp->durations[i] == t_star) {
            alias_draw(&sp->tables[i], rng, out_speed, out_angle);
            return;
        }
    }

    // find bracketing t0, t1
    int idx0 = 0;
    while (idx0 < sp->n_durations - 1 && sp->durations[idx0 + 1] < t_star)
        idx0++;
    int idx1 = idx0 + 1;

    // clamp to edges (extrapolation → nearest)
    if (t_star < sp->durations[0]) {
        alias_draw(&sp->tables[0], rng, out_speed, out_angle);
        return;
    }
    if (t_star > sp->durations[sp->n_durations - 1]) {
        alias_draw(&sp->tables[sp->n_durations - 1], rng, out_speed, out_angle);
        return;
    }

    float t0 = sp->durations[idx0];
    float t1 = sp->durations[idx1];
    float lambda = (t_star - t0) / (t1 - t0);   // weight toward t1

    // stochastic interpolation: draw from t0 or t1 with prob (1-l, l)
    if (curand_uniform(rng) > lambda)
        alias_draw(&sp->tables[idx0], rng, out_speed, out_angle);
    else
        alias_draw(&sp->tables[idx1], rng, out_speed, out_angle);

}

__global__ void test_sampling_kernel(const StateParams* params, int state, int t_star,
                                     int n_samples, unsigned long long seed,
                                     int* near_zero, int* exact_zero, float* sum_abs_angle) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples) return;
    curandState rng;
    curand_init(seed, tid, 0, &rng);

    float speed, angle;
    draw_speed_angle(&params[state], t_star, &rng, &speed, &angle);

    if (fabsf(angle) < 0.5f) atomicAdd(near_zero, 1);
    if (angle == 0.0f)       atomicAdd(exact_zero, 1);
    atomicAdd(sum_abs_angle, fabsf(angle));
}

void test_sampling(const StateParams* d_params, int state, int t_star, int n_samples = 100000) {
    int *d_nz, *d_ez; float* d_sum;
    cudaMalloc(&d_nz, sizeof(int)); cudaMalloc(&d_ez, sizeof(int)); cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_nz, 0, sizeof(int)); cudaMemset(d_ez, 0, sizeof(int)); cudaMemset(d_sum, 0, sizeof(float));

    test_sampling_kernel<<<(n_samples + 255) / 256, 256>>>(d_params, state, t_star, n_samples, 1234ULL, d_nz, d_ez, d_sum);
    cudaDeviceSynchronize();

    int nz, ez; float sum;
    cudaMemcpy(&nz, d_nz, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ez, d_ez, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  sample test state %d t=%d: mean|angle|=%.3f frac|a|<0.5=%.4f frac a==0=%.4f\n",
           state, t_star, sum / n_samples, (float)nz / n_samples, (float)ez / n_samples);
    cudaFree(d_nz); cudaFree(d_ez); cudaFree(d_sum);
}

__device__ float sampleVonMises(float mu, float kappa, curandState* rng)
{
    if (kappa < 1e-6f)
        return wrapAngle(mu + (2.0f * curand_uniform(rng) - 1.0f) * M_PI);

    const float tau = 1.0f + sqrtf(1.0f + 4.0f * kappa * kappa);
    const float rho = (tau - sqrtf(2.0f * tau)) / (2.0f * kappa);
    const float r   = (1.0f + rho * rho) / (2.0f * rho);

    float f;
    while (true) {
        const float u1 = curand_uniform(rng);
        const float u2 = curand_uniform(rng);
        const float z  = cosf(M_PI * u1);
        f = (1.0f + r * z) / (r + z);
        const float c = kappa * (r - f);
        if (c * (2.0f - c) - u2 > 0.0f || logf(c / u2) + 1.0f - c >= 0.0f)
            break;
    }
    const float u3 = curand_uniform(rng);
    const float theta = (u3 > 0.5f ? 1.0f : -1.0f) * acosf(fminf(fmaxf(f, -1.0f), 1.0f));
    return wrapAngle(mu + theta);
}

__global__ void moveFishAgents(Agent* agents, curandState* local_state, int fish_count, int timestep, StateParams* params, int* agent_count_grid, int* leader_of){
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id<fish_count) {
        int agent_state = agents[agent_id].state;
        StateParams* sp = &params[agent_state];
        float speed, angle_change;
        curandState local_rng = local_state[agent_id];
        draw_speed_angle(sp, agents[agent_id].initial_state_duration, &local_rng, &speed, &angle_change);
        // Enforce the social turning direction for the follower.
        if(fabsf(angle_change)>1.0f)
            angle_change = sign(angle_change) * 1.0f;
        if (agents[agent_id].turn_direction != 0) {
            angle_change =
                    agents[agent_id].turn_direction * fabsf(angle_change);
            //printf("t=%d fish=%d change direction, new angle change: %f\n", timestep, agent_id, angle_change);
        }
        float raw_angle = angle_change;

        if (leader_of[agent_id] >= 0) {
            angle_change = 0.0f;
        }
        float new_angle =agents[agent_id].angle+angle_change;
        //keep between -pi and pi
        new_angle = fmodf(new_angle + M_PI, 2 * M_PI);
        if (new_angle < 0) new_angle += 2 * M_PI;
        new_angle -= M_PI;

        //clip speed to 0-MAXIMUM_ALLOWED_SPEED
        if(speed<0.0f) speed=0.0f;
        if(speed>MAX_ALLOWED_SPEED) speed=MAX_ALLOWED_SPEED;
        //apply linear slowdown
        int n_neighbours = 0;
        float dx_grid = (float)WIDTH / GRID_N;   // 0.1mm per cell
        float dy_grid = (float) HEIGHT / GRID_N;
        // convert agent position -> grid cell indices
        int cell_x = (int)(agents[agent_id].x / dx_grid);
        int cell_y = (int)(agents[agent_id].y / dy_grid);
        // sum agent counts in the 3x3 neighbourhood (clamped to grid bounds)
        for (int dy_ = -1; dy_ <= 1; dy_++) {
            for (int dx_ = -1; dx_ <= 1; dx_++) {
                int nx = cell_x + dx_;
                int ny = cell_y + dy_;
                if (nx >= 0 && nx < GRID_N && ny >= 0 && ny < GRID_N) {
                    n_neighbours += agent_count_grid[nx * GRID_N + ny];
                }
            }
        }
        //printf("n_neighbors: %d\n", n_neighbours);
        //speed *= fmaxf(0.1f, 1.0f - SLOWDOWN_FACTOR * n_neighbours);
        //float factor = fmaxf(0.1f, 1.0f - SLOWDOWN_FACTOR * n_neighbours);
        //printf("factor: %f\n", factor);
        //find dx and dy
        float dx = speed * cosf(new_angle) * DT;
        float dy = speed * sinf(new_angle) * DT;
        int agent_old_i =  (int)(agents[agent_id].x / dx_grid);
        int agent_old_j =  (int)(agents[agent_id].y / dy_grid);
        // float rounding can push x/dx_grid to exactly GRID_N even though x < WIDTH
        if (agent_old_i >= GRID_N) agent_old_i = GRID_N - 1;
        if (agent_old_i < 0) agent_old_i = 0;
        if (agent_old_j >= GRID_N) agent_old_j = GRID_N - 1;
        if (agent_old_j < 0) agent_old_j = 0;
        agents[agent_id].x += dx;
        agents[agent_id].y += dy;
        //just keep them within the boundaries for now
        /*if (agents[agent_id].x < 0) agents[agent_id].x = 0;
        if (agents[agent_id].x >= WIDTH) agents[agent_id].x = WIDTH - 0.001f;
        if (agents[agent_id].y < 0) agents[agent_id].y = 0;
        if (agents[agent_id].y >= HEIGHT) agents[agent_id].y = HEIGHT - 0.001f;*/ //bad idea -- worms get stuck there

        // Reflect at boundaries
        if(BOUNDARY_TYPE==1) {
            if (agents[agent_id].x < 0.0f) {
                agents[agent_id].x = -agents[agent_id].x;
            } else if (agents[agent_id].x >= WIDTH) {
                agents[agent_id].x = 2.0f * WIDTH - agents[agent_id].x;
            }

            if (agents[agent_id].y < 0.0f) {
                agents[agent_id].y = -agents[agent_id].y;
            } else if (agents[agent_id].y >= HEIGHT) {
                agents[agent_id].y = 2.0f * HEIGHT - agents[agent_id].y;
            }
        }
        else {

            // Periodic boundary conditions
            if (agents[agent_id].x < 0.0f)
                agents[agent_id].x += WIDTH;
            else if (agents[agent_id].x >= WIDTH)
                agents[agent_id].x -= WIDTH;

            if (agents[agent_id].y < 0.0f)
                agents[agent_id].y += HEIGHT;
            else if (agents[agent_id].y >= HEIGHT)
                agents[agent_id].y -= HEIGHT;

        }
        assert(isfinite(agents[agent_id].x) && isfinite(agents[agent_id].y));

        int agent_new_i =  (int)(agents[agent_id].x / dx_grid);
        int agent_new_j =  (int)(agents[agent_id].y / dy_grid);
        // float rounding can push x/dx_grid to exactly GRID_N even though x < WIDTH
        if (agent_new_i >= GRID_N) agent_new_i = GRID_N - 1;
        if (agent_new_i < 0) agent_new_i = 0;
        if (agent_new_j >= GRID_N) agent_new_j = GRID_N - 1;
        if (agent_new_j < 0) agent_new_j = 0;
        assert(agent_new_i >= 0 && agent_new_i < GRID_N);
        assert(agent_new_j >= 0 && agent_new_j < GRID_N);
        if(agent_old_i!=agent_new_i || agent_old_j!=agent_new_j){
            //atomic add to grid
            atomicAdd(&agent_count_grid[agent_old_i * GRID_N + agent_old_j], -1);
            atomicAdd(&agent_count_grid[agent_new_i * GRID_N + agent_new_j], 1);
        }
        agents[agent_id].previous_speed = agents[agent_id].speed;
        agents[agent_id].speed = speed;
        agents[agent_id].angle = new_angle;
        agents[agent_id].previous_angle_change = agents[agent_id].angle_change;
        agents[agent_id].angle_change = angle_change;


        local_state[agent_id] = local_rng;

    }
}

__global__ void updateFishState(Agent* agents, curandState* local_rng, int fish_count){
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (agent_id >= fish_count)
        return;

    Agent agent = agents[agent_id];
    //printf("Agent %d duration: %d\n", agent_id, agent.state_duration);
    if(agent.state_duration==0){
        int cur_s = agent.state;
        //printf("cur_s : %d\n", cur_s);

        float mean_transition_matrix[9] = {0.0f, 0.049972f, 0.950028f, 0.044170f, 0.000000f, 0.955830f, 0.467582f,
                                           0.532418f, 0.000000f};
        float p[3] = {0.0f, 0.0f, 0.0f};
        //printf("Agent %d transitions with probs ", agent_id);
        for (int i = 0; i < 3; i++) {
            p[i] = mean_transition_matrix[cur_s * N_STATES + i];
            //printf("p[%d->%d]=%f | ", cur_s, i, p[i]);
        }

        curandState rng = local_rng[agent_id];
        int next_state = select_next_state(p, &rng, N_STATES);
        //printf("| chosen: %d->%d\n" , cur_s, next_state);

        agents[agent_id].previous_state = agents[agent_id].state;
        agents[agent_id].state = next_state;
        //sample duration for the new state
        const DurationDistribution& new_state_dist = d_duration_distributions[next_state];
        float u = curand_uniform(&rng);
        int idx = 0;
        for (int j = 0; j < new_state_dist.n_duration_bins - 1; j++) {
            if (u <= new_state_dist.duration_prob[j]) {
                idx = j;
                break;
            }
            idx = j + 1;  // fallback to last bin if u > all but last cumprob
        }
        int new_duration = new_state_dist.duration_bins[idx];
        agents[agent_id].state_duration = max(new_duration, 1); //at least 1 timestep in the new state

        DurationDistribution state = d_duration_distributions[next_state];

        agents[agent_id].initial_state_duration = agents[agent_id].state_duration;
        //printf("Agent %d chose duration %d for new state %d\n", agent_id, agents[agent_id].state_duration , next_state);

        local_rng[agent_id] = rng;
    } else{
        //agent.state_duration--;
        agents[agent_id].state_duration--;
        //printf("->reduced duration to %d\n", agents[agent_id].state_duration);
    }

}


// partner alignment, a in [-1, 1]
__device__ __forceinline__ float get_alignment(const Agent* agents, int agent_id, int n_agents)
{
    // 2-fish experiment: the partner is the other agent.
    // For >2 fish, replace with nearest-neighbour (or mean-field) alignment.
    int partner = (n_agents == 2) ? (1 - agent_id) : agent_id;
    return cosf(agents[agent_id].angle - agents[partner].angle);
}


__device__ void sampleNewDuration(
        Agent* agents, int id, int next_state, curandState* rng)
{
    const DurationDistribution& dist =
            d_duration_distributions[next_state];

    const float u = curand_uniform(rng);
    int idx = 0;

    for (int j = 0; j < dist.n_duration_bins - 1; ++j) {
        if (u <= dist.duration_prob[j]) {
            idx = j;
            break;
        }
        idx = j + 1;
    }

    const int duration = max(dist.duration_bins[idx], 1);

    agents[id].previous_state = agents[id].state;
    agents[id].state = next_state;
    agents[id].state_duration = duration;
    agents[id].initial_state_duration = duration;
}

__device__ void getSocialCommand(
        const Agent* agents,
        int follower,
        int leader,
        int* desired_state,
        int* turn_direction, curandState* rng, float lambda)
{
    const float e = wrapAngle(
            agents[leader].angle - agents[follower].angle
    );

    const float abs_e = fabsf(e);

    // Direction: positive heading error means turn left.
    if (abs_e > 1e-6f)
        *turn_direction = (e > 0.0f) ? 1 : -1;
    else
        *turn_direction = agents[follower].turn_direction;

    int social_state;
    int antisocial_state;

    if (abs_e > 2.0f * M_PI / 3.0f) {
        social_state = 0;
        antisocial_state = 2;
    }
    else if (abs_e > M_PI / 3.0f) {
        social_state = 1;
        antisocial_state = 1;
    }
    else {
        social_state = 2;
        antisocial_state = 0;

    }

    // Select between the two
    *desired_state = (curand_uniform(rng) < lambda)
                     ? social_state
                     : antisocial_state;
}

__device__ void updateHandcraftedFollower(
        Agent* agents,
        int id,
        int leader,
        curandState* rng, float lambda)
{
    int desired_state;
    int desired_direction;

    getSocialCommand(
            agents,
            id,
            leader,
            &desired_state,
            &desired_direction, rng, lambda
    );

    // Update turning direction continuously.
    agents[id].turn_direction = desired_direction;

    // Preserve the current empirical bout duration.
    if (agents[id].state_duration > 0) {
        agents[id].state_duration--;
        return;
    }

    // Enter the state dictated by the current heading error.
    sampleNewDuration(agents, id, desired_state, rng);
}

__device__ void updateBaselineAgent(
        Agent* agents, int id, curandState* rng)
{
    if (agents[id].state_duration > 0) {
        agents[id].state_duration--;
        return;
    }

    const float mean_transition_matrix[9] = {
            0.0f,      0.049972f, 0.950028f,
            0.044170f, 0.0f,      0.955830f,
            0.467582f, 0.532418f, 0.0f
    };

    const int cur_s = agents[id].state;
    float p[N_STATES];

    for (int j = 0; j < N_STATES; ++j)
        p[j] = mean_transition_matrix[cur_s * N_STATES + j];

    const int next_state = select_next_state(p, rng, N_STATES);

    if (next_state >= 0 && next_state < N_STATES)
        sampleNewDuration(agents, id, next_state, rng);
}

__device__ void updateSocialAgent(
        Agent* agents, int id, curandState* rng, int fish_count)
{
    const int agent_state = agents[id].state;

    // Keep the original alignment-based social input.
    const float a = get_alignment(agents, id, fish_count);

    // L1: state-2 exit probability.
    if (agents[id].state_duration > 1 && agent_state == 2) {
        const TransitionModel exit_model = d_exit_models[agent_state];

        const float z = exit_model.coeff * a + exit_model.intercept;
        const float p_exit =
                exit_model.height / (1.0f + expf(-z));

        if (curand_uniform(rng) < p_exit)
            agents[id].state_duration = 0;
    }

    if (agents[id].state_duration > 1) {
        agents[id].previous_state = agent_state;
        agents[id].state_duration--;
        return;
    }

    // L2: alignment-dependent transition probabilities.
    float p[N_STATES];
    float sum_p = 0.0f;

    for (int j = 0; j < N_STATES; ++j) {
        const TransitionModel model =
                d_transition_models[agent_state * N_STATES + j];

        const float z = model.coeff * a + model.intercept;
        p[j] = model.height / (1.0f + expf(-z));
        sum_p += p[j];
    }

    if (sum_p > 0.0f) {
        for (int j = 0; j < N_STATES; ++j)
            p[j] /= sum_p;
    } else {
        for (int j = 0; j < N_STATES; ++j)
            p[j] = (j == agent_state) ? 1.0f : 0.0f;
    }

    const int next_state = select_next_state(p, rng, N_STATES);

    if (next_state >= 0 && next_state < N_STATES)
        sampleNewDuration(agents, id, next_state, rng);
}

__global__ void updateFishStateCollective(
        Agent* agents,
        curandState* rng_states,
        int timestep,
        int worm_count, float lambda, int* leader_of)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= worm_count)
        return;

    curandState rng = rng_states[id];

    if (worm_count == 2) {
        const int follower = getFollower(agents);

        if (id == follower) {
            const int leader = 1 - follower;
            updateHandcraftedFollower(
                    agents, id, leader, &rng, lambda
            );
        }
        else {
            updateBaselineAgent(agents, id, &rng);
        }
    }
    else {
        const int leader = leader_of[id];
        if (leader >= 0)
            updateHandcraftedFollower(agents, id, leader, &rng, lambda);
        else
            updateBaselineAgent(agents, id, &rng);
    }
    /*float heading_diff = atan2f(
            sinf(agents[0].angle - agents[1].angle),
            cosf(agents[0].angle - agents[1].angle)
    );

    printf("t=%d | states=(%d,%d) | headings=(%.3f,%.3f) "
           "| diff=%.3f | alignment=%.3f | follower=%d\n",
           timestep,
           agents[0].state, agents[1].state,
           agents[0].angle, agents[1].angle,
           heading_diff, cosf(heading_diff),
           getFollower(agents));*/
    rng_states[id] = rng;
}

__global__ void moveAgents(Agent* agents, curandState* local_state, int worm_count, int timestep, StateParams* params, int* agent_count_grid, float* food_grid, StateParams* food_params) {
    //logic:
    //1) get local #agents by summing agent_count_grid in agent cell + 8 surrounding cells
    //2) local #agents influences speed linearly
    //3) bias = LOGISTIC_BIAS(phi) -- for the future
    //4) update position + atomic sum/diff on agent_count_grid

    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id<worm_count) {
        float dx_grid = (float)WIDTH / GRID_N;   // 0.1mm per cell
        float dy_grid = (float) HEIGHT / GRID_N;
        // convert agent position -> grid cell indices
        int cell_x = (int)(agents[agent_id].x / dx_grid);
        int cell_y = (int)(agents[agent_id].y / dy_grid);
        int agent_state = agents[agent_id].state;
        assert(agent_state >= 0 && agent_state < N_STATES);
        StateParams* sp = &params[agent_state];
        float speed, angle_change;
        float speed_on_food, angle_change_on_food;
        curandState local_rng = local_state[agent_id];
        draw_speed_angle(sp, agents[agent_id].initial_state_duration, &local_rng, &speed, &angle_change);
        StateParams* sp_on_food = &food_params[agent_state];
        draw_speed_angle(sp_on_food, agents[agent_id].initial_state_duration, &local_rng, &speed_on_food, &angle_change_on_food);
        //if(agent_state==1 && (fabsf(angle_change)<1.5f || fabsf(angle_change_on_food)<1.5f)) printf("agent=%d t=%d STATE %d [OFF] dtheta=%f, speed=%f; [ON] dtheta=%f, speed=%f\n", agent_id, timestep, agent_state, angle_change, speed, angle_change_on_food, speed_on_food);

        float food_level = food_grid[cell_x * GRID_N + cell_y];
        float* cell = &food_grid[cell_x * GRID_N + cell_y];
        float c = c_cfg.food_consumption;
        float old = atomicAdd(cell, -c);
        float eaten = fminf(fmaxf(old, 0.0f), c);
        if (eaten < c)
            atomicAdd(cell, c - eaten);   // give back what wasn't there
        agents[agent_id].food_consumed = eaten;
        agents[agent_id].total_food_consumed += eaten;
        speed = food_level * speed_on_food + (1.0f-food_level) * speed;
        //speed *= 0.001f;
        float mu_score = 0.2f, std_score = 0.14f;//0.547f; -> all off-food worms
        float mu_period = 2.363f, sigma_period = 0.57f;//0.581f; -> all off-food worms
        agents[agent_id].phi += agents[agent_id].run_omega;
        if (agent_state == 2) {
            // initialize once when entering run
            if (agents[agent_id].previous_state != 2 || timestep==0) {
                agents[agent_id].run_bias =  0.005f * curand_normal(&local_rng);
                while(fabsf(agents[agent_id].run_bias)>0.7f){
                    agents[agent_id].run_bias = 0.005f * curand_normal(&local_rng);
                }
            }
            float sigma_theta = sampleVonMises(0.0f, agents[agent_id].kappa, &local_rng); // tune from residuals of real data
            angle_change = agents[agent_id].run_amp *sign(sinf(agents[agent_id].phi))
                           + agents[agent_id].run_bias;// + sigma_theta;
            while(fabsf(angle_change)>1.5f){
                sigma_theta = sampleVonMises(0.0f, agents[agent_id].kappa, &local_rng); // tune from residuals of real data
                angle_change = agents[agent_id].run_amp * sign(sinf(agents[agent_id].phi))
                               + sigma_theta+ agents[agent_id].run_bias ;
            }
        }
        if (curand_uniform(&local_rng) < food_level && agent_state!=2) {
            angle_change = angle_change_on_food;
        }

        float new_angle =agents[agent_id].angle+angle_change;
        //keep between -pi and pi
        new_angle = fmodf(new_angle + M_PI, 2 * M_PI);
        if (new_angle < 0) new_angle += 2 * M_PI;
        new_angle -= M_PI;
        //clip speed to 0-MAXIMUM_ALLOWED_SPEED
        if(speed<0.0f) speed=0.0f;
        if(speed>MAX_ALLOWED_SPEED) speed=MAX_ALLOWED_SPEED;
        //apply linear slowdown
        int n_neighbours = 0;

        // sum agent counts in the 3x3 neighbourhood (clamped to grid bounds)
        for (int dy_ = -1; dy_ <= 1; dy_++) {
            for (int dx_ = -1; dx_ <= 1; dx_++) {
                int nx = cell_x + dx_;
                int ny = cell_y + dy_;
                if (nx >= 0 && nx < GRID_N && ny >= 0 && ny < GRID_N) {
                    n_neighbours += agent_count_grid[nx * GRID_N + ny];
                }
            }
        }
        agents[agent_id].n_neighbors = n_neighbours;
        //printf("n_neighbors: %d\n", n_neighbours);
        speed *= fmaxf(0.1f, 1.0f - SLOWDOWN_FACTOR * n_neighbours);
        //float factor = fmaxf(0.1f, 1.0f - SLOWDOWN_FACTOR * n_neighbours);
        //printf("factor: %f\n", factor);
        //find dx and dy
        float dx = speed * cosf(new_angle) * DT;
        float dy = speed * sinf(new_angle) * DT;
        int agent_old_i =  (int)(agents[agent_id].x / dx_grid);
        int agent_old_j =  (int)(agents[agent_id].y / dy_grid);
        // float rounding can push x/dx_grid to exactly GRID_N even though x < WIDTH
        if (agent_old_i >= GRID_N) agent_old_i = GRID_N - 1;
        if (agent_old_i < 0) agent_old_i = 0;
        if (agent_old_j >= GRID_N) agent_old_j = GRID_N - 1;
        if (agent_old_j < 0) agent_old_j = 0;
        agents[agent_id].x += dx;
        agents[agent_id].y += dy;

        // Reflect at boundaries
        if(BOUNDARY_TYPE==1) {
            if (agents[agent_id].x < 0.0f) {
                agents[agent_id].x = -agents[agent_id].x;
            } else if (agents[agent_id].x >= WIDTH) {
                agents[agent_id].x = 2.0f * WIDTH - agents[agent_id].x;
            }

            if (agents[agent_id].y < 0.0f) {
                agents[agent_id].y = -agents[agent_id].y;
            } else if (agents[agent_id].y >= HEIGHT) {
                agents[agent_id].y = 2.0f * HEIGHT - agents[agent_id].y;
            }
        }
        else {

            // Periodic boundary conditions
            if (agents[agent_id].x < 0.0f)
                agents[agent_id].x += WIDTH;
            else if (agents[agent_id].x >= WIDTH)
                agents[agent_id].x -= WIDTH;

            if (agents[agent_id].y < 0.0f)
                agents[agent_id].y += HEIGHT;
            else if (agents[agent_id].y >= HEIGHT)
                agents[agent_id].y -= HEIGHT;

        }
        assert(isfinite(agents[agent_id].x) && isfinite(agents[agent_id].y));

        int agent_new_i =  (int)(agents[agent_id].x / dx_grid);
        int agent_new_j =  (int)(agents[agent_id].y / dy_grid);
        // float rounding can push x/dx_grid to exactly GRID_N even though x < WIDTH
        if (agent_new_i >= GRID_N) agent_new_i = GRID_N - 1;
        if (agent_new_i < 0) agent_new_i = 0;
        if (agent_new_j >= GRID_N) agent_new_j = GRID_N - 1;
        if (agent_new_j < 0) agent_new_j = 0;
        assert(agent_new_i >= 0 && agent_new_i < GRID_N);
        assert(agent_new_j >= 0 && agent_new_j < GRID_N);
        if(agent_old_i!=agent_new_i || agent_old_j!=agent_new_j){
            //atomic add to grid
            atomicAdd(&agent_count_grid[agent_old_i * GRID_N + agent_old_j], -1);
            atomicAdd(&agent_count_grid[agent_new_i * GRID_N + agent_new_j], 1);
        }

        agents[agent_id].speed = speed;
        agents[agent_id].angle = new_angle;
        agents[agent_id].angle_change = angle_change;


        local_state[agent_id] = local_rng;

    }
}


__device__ float get_local_pheromone(const Agent& agent, const float* phi_grid)
{
    int i = (int)(agent.x / WIDTH  * GRID_N);
    int j = (int)(agent.y / HEIGHT * GRID_N);

    // Safety clamp
    i = max(0, min(i, GRID_N - 1));
    j = max(0, min(j, GRID_N - 1));

    return phi_grid[i * GRID_N + j];
}


__device__ int sample_duration(
        const DurationDistribution& dist,
        curandState* rng
) {
    float u = curand_uniform(rng);
    int idx = 0;

    for (int j = 0; j < dist.n_duration_bins - 1; j++) {
        if (u <= dist.duration_prob[j]) {
            idx = j;
            break;
        }
        idx = j + 1;
    }

    return max(dist.duration_bins[idx], 1);
}

__global__ void updateAgentStateConstant(Agent* agents, curandState* rng_states, float* food_grid, StateParams* params, StateParams* params_on_food,
                                         int worm_count, float* phi_grid) {
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (agent_id >= worm_count)
        return;

    float phi = get_local_pheromone(agents[agent_id], phi_grid);
    float dphi = (phi - agents[agent_id].previous_phi) / DT;

    agents[agent_id].previous_phi = phi;
    agents[agent_id].dphi = dphi;

    if(agents[agent_id].state_duration > 1){

        agents[agent_id].previous_state = agents[agent_id].state;
        agents[agent_id].state = agents[agent_id].state; //keep the same state
        agents[agent_id].state_duration -= 1;
        //printf("agent %d state duration: %d\n", agent_id, agents[agent_id].state_duration);
        return; //don't update state if duration not over
    }

    int i = (int)(agents[agent_id].x / WIDTH  * GRID_N);
    int j = (int)(agents[agent_id].y / HEIGHT * GRID_N);
    float food_level = food_grid[i*GRID_N+j];
    float p_on[N_STATES*N_STATES] = {0.0, 0.3448275862068966, 0.6551724137931034,
                                      0.08762057877813505, 0.0, 0.912379421221865,
                                      0.23136427566807313, 0.7686357243319268, 0.0};
    float p_off[N_STATES*N_STATES] = {0.0, 0.3951612903225806, 0.6048387096774194,
                                      0.06286836935166994, 0.0, 0.93713163064833,
                                      0.16787003610108303, 0.8321299638989169, 0.0};

    int cur_s = agents[agent_id].state;
    float p[N_STATES];
    for(int i=0; i<N_STATES; i++){
        int next_s = cur_s*N_STATES+i;
        p[i] = food_level * p_on[next_s] + (1-food_level) * p_off[next_s];
        //printf("[TRANSITION]: p(%d -> %d) = %f\n", cur_s, cur_s*N_STATES+i, p[i]);
    }
    curandState local_rng = rng_states[agent_id];
    int next_state = select_next_state(p, &local_rng, N_STATES);

    agents[agent_id].previous_state = agents[agent_id].state;
    agents[agent_id].state = next_state;
    //sample duration for the new state
    int new_duration;

    if (curand_uniform(&local_rng) < food_level) {
        new_duration = sample_duration(
                d_duration_distributions_on_food[next_state],
                &local_rng
        );
    } else {
        new_duration = sample_duration(
                d_duration_distributions[next_state],
                &local_rng
        );
    }

    agents[agent_id].state_duration = new_duration;
    agents[agent_id].initial_state_duration = new_duration;

    rng_states[agent_id] = local_rng;

}

__global__ void updateAgentStateCollective(
        Agent* agents,
        curandState* rng_states,
        int timestep,
        int worm_count, StateParams* params,
        const float* phi_grid, float* food_grid, IntegrationModel* integration_model)
{
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (agent_id >= worm_count)
        return;
    if (timestep==0){
        agents[agent_id].previous_phi = get_local_pheromone(agents[agent_id], phi_grid);
    }
    float phi = get_local_pheromone(agents[agent_id], phi_grid);
    float dphi = (phi - agents[agent_id].previous_phi) / DT;
    agents[agent_id].previous_phi = phi;
    //float dphi_star  = (dphi - m->mean_dphi) / m->std_dphi;
    float integrated_signal;
    if(c_cfg.integration_model==0 && c_cfg.food_amount>0.0f){
        //normalise dphi
        float dphi_star = (dphi - integration_model->mean_dphi) / integration_model->std_dphi;
        float food_level = get_local_pheromone(agents[agent_id], food_grid);
        //project food [0,1] -> dphi [R]
        //food_level = integration_model->min_dphi + food_level * (integration_model->max_dphi-integration_model->min_dphi);
        //normalise food
        //float food_norm = (food_level - integration_model->mean_dphi)/integration_model->std_dphi;
        //weight
        //integrated_signal = integration_model->w_food * food_norm + integration_model->w_phero * dphi_star;
        //de-normalize for L1/L2(dphi)
        integrated_signal = integration_model->w_food * food_level + integration_model->w_phero * dphi_star;
        //integrated_signal = dphi;
    } else{
        integrated_signal = dphi;
    }

    if(agents[agent_id].state_duration>1 && agents[agent_id].state==2){

        float p_exit;

        if(TRANSITION_TYPE == 1){
                const TransitionModel& exit_agg  = d_compound_exit_models.aggregation_model[agents[agent_id].state];
                const TransitionModel& exit_disp = d_compound_exit_models.dispersal_model[agents[agent_id].state];

                float z_exit_agg  = exit_agg.coeff  * integrated_signal + exit_agg.intercept;
                float z_exit_disp = exit_disp.coeff * integrated_signal + exit_disp.intercept;

                float p_exit_agg  = exit_agg.height  / (1.0f + expf(-z_exit_agg));
                float p_exit_disp = exit_disp.height / (1.0f + expf(-z_exit_disp));

                p_exit = d_compound_exit_models.lambda * p_exit_agg + (1.0f - d_compound_exit_models.lambda) * p_exit_disp;
        }
        else{
                TransitionModel exit_model = d_exit_models[agents[agent_id].state];

            float z_exit = exit_model.coeff * integrated_signal + exit_model.intercept;
            p_exit = exit_model.height / (1.0f + expf(-z_exit));
        }

        float u = curand_uniform(&rng_states[agent_id]);
        if (u < p_exit) {
            agents[agent_id].state_duration = 0;
        }
    }

    if(agents[agent_id].state_duration > 1){

        agents[agent_id].previous_state = agents[agent_id].state;
        agents[agent_id].state = agents[agent_id].state; //keep the same state
        agents[agent_id].state_duration -= 1;
        return; //don't update state if duration not over
    }

    curandState local_rng = rng_states[agent_id];

    int agent_state = agents[agent_id].state;

    float p[N_STATES] = {0.0f};

    float p_irr = 0.0f;
    float p_r_raw[N_STATES];
    float sum_r = 0.0f;

// PASS 1: compute raw values
    for (int i = 0; i < N_STATES; i++)
    {
        int idx = agent_state * N_STATES + i;

        if(TRANSITION_TYPE == 1)
        {
            const TransitionModel &model_agg = d_compound_transition_models.aggregation_model[idx];
            const TransitionModel &model_disp = d_compound_transition_models.dispersal_model[idx];

            // assume aggregation/dispersal models agree on which transitions are
            // "off-food" categorical entries (coeff==-1 && intercept==-1)
            if (model_agg.coeff == -1 && model_agg.intercept == -1) {
                float p_off = d_compound_transition_models.lambda * model_agg.p_off_food +
                              (1.0f - d_compound_transition_models.lambda) * model_disp.p_off_food;
                p[i] = p_off;
                p_irr += p[i];
                p_r_raw[i] = 0.0f;
            } else {
                float z_agg = model_agg.coeff * integrated_signal + model_agg.intercept;
                float z_disp = model_disp.coeff * integrated_signal + model_disp.intercept;

                float val_agg = model_agg.height / (1.0f + expf(-z_agg));
                float val_disp = model_disp.height / (1.0f + expf(-z_disp));

                float val = d_compound_transition_models.lambda * val_agg +
                            (1.0f - d_compound_transition_models.lambda) * val_disp;

                p_r_raw[i] = val;
                sum_r += val;
            }
        }
        else{
            const TransitionModel &model = d_transition_models[idx];

            if (model.coeff == -1 && model.intercept == -1) {
                p[i] = model.p_off_food;
                p_irr += p[i];
                p_r_raw[i] = 0.0f;
            } else {
                float z = model.coeff * integrated_signal + model.intercept;
                float height = model.height;
                float val = height / (1.0f + expf(-z));

                p_r_raw[i] = val;
                sum_r += val;
            }
        }
    }

// PASS 2: normalize ONLY relevant transitions
    float remaining_mass = 1.0f - p_irr;

    if (sum_r > 0.0f && remaining_mass > 0.0f)
    {
        for (int i = 0; i < N_STATES; i++)
        {
            int idx = agent_state * N_STATES + i;
            bool is_off_food;
            if(TRANSITION_TYPE == 1) {
                const TransitionModel &model_agg = d_compound_transition_models.aggregation_model[idx];
                is_off_food = (model_agg.coeff == -1 && model_agg.intercept == -1);
            }
            else {
                const TransitionModel &model = d_transition_models[idx];
                is_off_food = (model.coeff == -1 && model.intercept == -1);
                //printf("is_off_food is defined and %d\n", is_off_food);
            }

            if (!is_off_food)
            {
                p[i] = (p_r_raw[i] / sum_r) * remaining_mass;
            }
        }
    }

    int next_state = select_next_state(p, &local_rng, N_STATES);
    if (next_state < 0 || next_state >= N_STATES) {
        printf("ERROR next_state=%d (agent %d)\n", next_state, agent_id);
        return;
    }

    agents[agent_id].previous_state = agents[agent_id].state;
    agents[agent_id].state = next_state;
    //sample duration for the new state
    const DurationDistribution& new_state_dist = d_duration_distributions[next_state];
    float u = curand_uniform(&local_rng);
    int idx = 0;
    for (int j = 0; j < new_state_dist.n_duration_bins - 1; j++) {
        if (u <= new_state_dist.duration_prob[j]) {
            idx = j;
            break;
        }
        idx = j + 1;  // fallback to last bin if u > all but last cumprob
    }
    int new_duration = new_state_dist.duration_bins[idx];
    agents[agent_id].state_duration = max(new_duration, 1); //at least 1 timestep in the new state

    DurationDistribution state = d_duration_distributions[next_state];

    agents[agent_id].initial_state_duration = agents[agent_id].state_duration;

    rng_states[agent_id] = local_rng;
}



//DETERMINISTIC
__global__ void updateAgentStateDeterministic(
        Agent* agents,
        const int* __restrict__ d_labels,
        int   n_labels,
        int   t)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= WORM_COUNT) return;

    int t_cur  = min(t,     n_labels - 1);
    int t_prev = max(0, t - 1);

    int current_state  = d_labels[t_cur];
    int previous_state = d_labels[t_prev];

    agents[id].previous_state = agents[id].state;  // for moveAgents to read
    agents[id].state          = current_state;

    // transition detection: use label sequence only, not agents[id].state
    bool state_changed = (t == 0) || (previous_state != current_state);

    if (state_changed) {
        int run_length = 0;
        for (int tau = t_cur; tau < n_labels; tau++) {
            if (d_labels[tau] == current_state) run_length++;
            else                                break;
        }
        agents[id].initial_state_duration = run_length;
        agents[id].state_duration         = run_length;
    } else {
        agents[id].state_duration -= 1;
    }
}

#endif //UNTITLED2_AGENT_UPDATE_H
