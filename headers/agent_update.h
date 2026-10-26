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

    /*if (speed == 0.0f) {
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

__device__ float sample_von_mises(curandState* rng, float kappa) {
    // Best & Fisher (1979) algorithm
    // Returns a sample in (-pi, pi) with concentration kappa around 0

    float tau  = 1.0f + sqrtf(1.0f + 4.0f * kappa * kappa);
    float rho  = (tau - sqrtf(2.0f * tau)) / (2.0f * kappa);
    float r    = (1.0f + rho * rho) / (2.0f * rho);

    float z, f, c, u1, u2, u3;
    while (true) {
        u1 = curand_uniform(rng);
        u2 = curand_uniform(rng);
        u3 = curand_uniform(rng);

        z  = cosf(3.14159265f * u1);
        f  = (1.0f + r * z) / (r + z);
        c  = kappa * (r - f);

        if (c * (2.0f - c) > u2) break;           // acceptance condition 1
        if (logf(c / u2) + 1.0f - c >= 0.0f) break;  // acceptance condition 2
    }

    return (u3 > 0.5f ? 1.0f : -1.0f) * acosf(f);
}

__global__ void moveFishAgents(Agent* agents, curandState* local_state, int fish_count, int timestep, StateParams* params, int* agent_count_grid){
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id<fish_count) {
        int agent_state = agents[agent_id].state;
        StateParams* sp = &params[agent_state];
        float speed, angle_change;
        curandState local_rng = local_state[agent_id];
        draw_speed_angle(sp, agents[agent_id].initial_state_duration, &local_rng, &speed, &angle_change);
        float raw_angle = angle_change;
        //simple autocorrelation rule: if state =0|1, sign(angle_change(t-1)) = sign(angle_change(t))
        /*if ((agent_state == 0 || agent_state == 1) &&
            agent_state != agents[agent_id].previous_state)
        {
            agents[agent_id].previous_angle_change = sign(angle_change);
        }
        if(agent_state==0 || agent_state == 1) {
            float abs_dtheta = fabsf(angle_change);
            float prev_sign = sign(agents[agent_id].previous_angle_change);
            angle_change = prev_sign * abs_dtheta;
        }*/
        float Q = 0.9f;
        bool new_bout = (agents[agent_id].initial_state_duration == agents[agent_id].state_duration);
        if (agent_state == 0 || agent_state == 1) {
            if (new_bout)             // new bout: draw its sign
                agents[agent_id].bout_sign = (curand_uniform(&local_rng) < 0.5f) ? -1.f : 1.f;
            float s = (curand_uniform(&local_rng) < Q) ? agents[agent_id].bout_sign
                                                 : ((curand_uniform(&local_rng) < 0.5f) ? -1.f : 1.f);
            //angle_change = s * fabsf(angle_change);
        }

        //simple state 2 rule: decelerate with a = 0.995
        /*if(agent_state==2 && agents[agent_id].initial_state_duration!=agents[agent_id].state_duration) { //NOT first timestep in 2, otherwise speed is taken from prev state
            speed= agents[agent_id].previous_speed * 0.9995f;
        }*/
        // angle_change = raw i.i.d. draw from the joint table
        float raw_sign = copysignf(1.f, angle_change);
        float a        = fabsf(angle_change);
        /*float R = 0.5f;
        if (agent_state == 2) {
            int elapsed = agents[agent_id].initial_state_duration - agents[agent_id].state_duration;
            if (elapsed >= 1 && curand_uniform(&local_rng) < R)
                angle_change = -agents[agent_id].prev_raw_sign * a;   // anti-correlate with previous RAW draw
            agents[agent_id].prev_raw_sign = raw_sign;                // store the raw sign, not the modified one
        }*/
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

__global__ void updateFishStateCollective(
        Agent* agents,
        curandState* rng_states,
        int timestep,
        int worm_count)
{
    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id >= worm_count)
        return;

    curandState local_rng = rng_states[agent_id];

    // L1/L2 input: alignment with the partner
    const float a = get_alignment(agents, agent_id, worm_count);

    const int agent_state = agents[agent_id].state;

    // ---------------- L1: exit probability, given a (state 2 ONLY) ----------------
    if (agents[agent_id].state_duration > 1 && agent_state == 2) {
        const TransitionModel exit_model = d_exit_models[agent_state];

        float z_exit  = exit_model.coeff * a + exit_model.intercept;
        float p_exit  = exit_model.height / (1.0f + expf(-z_exit));

        if (curand_uniform(&local_rng) < p_exit) {
            agents[agent_id].state_duration = 0;     // force a transition now
        }
    }

    // duration not over -> stay in the same state
    if (agents[agent_id].state_duration > 1) {
        agents[agent_id].previous_state = agents[agent_id].state;
        agents[agent_id].state_duration -= 1;
        rng_states[agent_id] = local_rng;            // persist RNG on early exit
        return;
    }

    // ---------------- L2: transition probabilities i -> j, given a ----------------
    float p[N_STATES];
    float sum_r = 0.0f;

    for (int j = 0; j < N_STATES; j++) {
        const TransitionModel model = d_transition_models[agent_state * N_STATES + j];
        float z   = model.coeff * a + model.intercept;
        p[j]      = model.height / (1.0f + expf(-z));
        sum_r    += p[j];
    }

    // No food / off-food categorical entries for fish: every transition is
    // "relevant", so just normalise. NOTE: only the RELATIVE heights matter
    // after normalisation.
    if (sum_r > 0.0f) {
        for (int j = 0; j < N_STATES; j++)
            p[j] /= sum_r;
    } else {
        // all heights ~0 (possible during optimisation): stay in current state
        for (int j = 0; j < N_STATES; j++)
            p[j] = (j == agent_state) ? 1.0f : 0.0f;
    }

    int next_state = select_next_state(p, &local_rng, N_STATES);
    //printf("agent %d t=%d transitions %d->%d\n", agent_id, timestep, agent_state, next_state);
    if (next_state < 0 || next_state >= N_STATES) {
        printf("ERROR next_state=%d (agent %d)\n", next_state, agent_id);
        rng_states[agent_id] = local_rng;
        return;
    }

    agents[agent_id].previous_state = agent_state;
    agents[agent_id].state          = next_state;

    // ---------------- sample duration for the new state ----------------
    const DurationDistribution& dist = d_duration_distributions[next_state];
    float u = curand_uniform(&local_rng);
    int idx = 0;
    for (int j = 0; j < dist.n_duration_bins - 1; j++) {
        if (u <= dist.duration_prob[j]) { idx = j; break; }
        idx = j + 1;
    }
    agents[agent_id].state_duration         = max(dist.duration_bins[idx], 1);
    agents[agent_id].initial_state_duration = agents[agent_id].state_duration;

    rng_states[agent_id] = local_rng;
}

__global__ void moveAgents(Agent* agents, curandState* local_state, int worm_count, int timestep, StateParams* params, int* agent_count_grid) {
    //logic:
    //1) get local #agents by summing agent_count_grid in agent cell + 8 surrounding cells
    //2) local #agents influences speed linearly
    //3) bias = LOGISTIC_BIAS(phi) -- for the future
    //4) update position + atomic sum/diff on agent_count_grid

    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id<worm_count) {

        int agent_state = agents[agent_id].state;
        assert(agent_state >= 0 && agent_state < N_STATES);
        StateParams* sp = &params[agent_state];
        float speed, angle_change;
        curandState local_rng = local_state[agent_id];
        draw_speed_angle(sp, agents[agent_id].initial_state_duration, &local_rng, &speed, &angle_change);
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
            float sigma_theta = sample_von_mises(&local_rng, agents[agent_id].kappa); // tune from residuals of real data
            angle_change = agents[agent_id].run_amp *sign(sinf(agents[agent_id].phi))
                           + agents[agent_id].run_bias;// + sigma_theta;
            while(fabsf(angle_change)>1.5f){
                sigma_theta = sample_von_mises(&local_rng, agents[agent_id].kappa); // tune from residuals of real data
                angle_change = agents[agent_id].run_amp * sign(sinf(agents[agent_id].phi))
                               + sigma_theta+ agents[agent_id].run_bias ;
            }
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

__global__ void updateAgentStateCollective(
        Agent* agents,
        curandState* rng_states,
        int timestep,
        int worm_count, StateParams* params,
        const float* phi_grid)
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

    if(agents[agent_id].state_duration>1 && agents[agent_id].state==2 ){

        float p_exit;

    #if TRANSITION_TYPE == 1
            const TransitionModel& exit_agg  = d_compound_exit_models.aggregation_model[agents[agent_id].state];
            const TransitionModel& exit_disp = d_compound_exit_models.dispersal_model[agents[agent_id].state];

            float z_exit_agg  = exit_agg.coeff  * dphi + exit_agg.intercept;
            float z_exit_disp = exit_disp.coeff * dphi + exit_disp.intercept;

            float p_exit_agg  = exit_agg.height  / (1.0f + expf(-z_exit_agg));
            float p_exit_disp = exit_disp.height / (1.0f + expf(-z_exit_disp));

            p_exit = d_compound_exit_models.lambda * p_exit_agg + (1.0f - d_compound_exit_models.lambda) * p_exit_disp;
    #else
            TransitionModel exit_model = d_exit_models[agents[agent_id].state];

        float z_exit = exit_model.coeff * dphi + exit_model.intercept;
        p_exit = exit_model.height / (1.0f + expf(-z_exit));
    #endif

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

        #if TRANSITION_TYPE == 1
                const TransitionModel& model_agg  = d_compound_transition_models.aggregation_model[idx];
                const TransitionModel& model_disp = d_compound_transition_models.dispersal_model[idx];

                // assume aggregation/dispersal models agree on which transitions are
                // "off-food" categorical entries (coeff==-1 && intercept==-1)
                if (model_agg.coeff == -1 && model_agg.intercept == -1)
                {
                    float p_off = d_compound_transition_models.lambda * model_agg.p_off_food + (1.0f - d_compound_transition_models.lambda) * model_disp.p_off_food;
                    p[i] = p_off;
                    p_irr += p[i];
                    p_r_raw[i] = 0.0f;
                }
                else
                {
                    float z_agg  = model_agg.coeff  * dphi + model_agg.intercept;
                    float z_disp = model_disp.coeff * dphi + model_disp.intercept;

                    float val_agg  = model_agg.height  / (1.0f + expf(-z_agg));
                    float val_disp = model_disp.height / (1.0f + expf(-z_disp));

                    float val = d_compound_transition_models.lambda * val_agg + (1.0f - d_compound_transition_models.lambda) * val_disp;

                    p_r_raw[i] = val;
                    sum_r += val;
                }
        #else
                const TransitionModel& model = d_transition_models[idx];

            if (model.coeff == -1 && model.intercept == -1)
            {
                p[i] = model.p_off_food;
                p_irr += p[i];
                p_r_raw[i] = 0.0f;
            }
            else
            {
                float z = model.coeff * dphi + model.intercept;
                float height = model.height;
                float val = height / (1.0f + expf(-z));

                p_r_raw[i] = val;
                sum_r += val;
            }
        #endif
            }

// PASS 2: normalize ONLY relevant transitions
    float remaining_mass = 1.0f - p_irr;

    if (sum_r > 0.0f && remaining_mass > 0.0f)
    {
        for (int i = 0; i < N_STATES; i++)
        {
            int idx = agent_state * N_STATES + i;

            #if TRANSITION_TYPE == 1
                        const TransitionModel& model_agg = d_compound_transition_models.aggregation_model[idx];
                        bool is_off_food = (model_agg.coeff == -1 && model_agg.intercept == -1);
            #else
                        const TransitionModel& model = d_transition_models[idx];
                    bool is_off_food = (model.coeff == -1 && model.intercept == -1);
            #endif

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
