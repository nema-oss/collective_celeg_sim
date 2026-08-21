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
__device__ void alias_draw(const JointTable* table, curandState* rng,
                           float* out_speed, float* out_angle) {
    int   i = (int)(curand_uniform(rng) * table->n);   // uniform bin
    float u =       curand_uniform(rng);
    int   idx = (u < table->prob[i]) ? i : table->alias[i];
    *out_speed = table->obs[idx * 2];
    *out_angle = table->obs[idx * 2 + 1];
}

// ---- Interpolated draw (point iv) ------------------------------------
__device__ void draw_speed_angle(const StateParams* sp, int t_star,
                                 curandState* rng,
                                 float* out_speed, float* out_angle) {
    // Binary search for t0, t1 bracketing t_star
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

__global__ void moveAgents(Agent* agents, curandState* local_state, int worm_count, int timestep, StateParams* params, int* agent_count_grid) {
    //logic:
    //1) get local #agents by summing agent_count_grid in agent cell + 8 surrounding cells
    //2) local #agents influences speed linearly
    //3) bias = LOGISTIC_BIAS(phi) -- for the future
    //4) update position + atomic sum/diff on agent_count_grid

    int agent_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (agent_id<worm_count) {

        int agent_state = agents[agent_id].state;

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

        speed *= fmaxf(0.0f, 1.0f - SLOWDOWN_FACTOR * n_neighbours);
        //find dx and dy
        float dx = speed * cosf(new_angle) * DT;
        float dy = speed * sinf(new_angle) * DT;
        int agent_old_i =  (int)(agents[agent_id].x / dx_grid);
        int agent_old_j =  (int)(agents[agent_id].y / dy_grid);
        agents[agent_id].x += dx;
        agents[agent_id].y += dy;
        //just keep them within the boundaries for now
        if (agents[agent_id].x < 0) agents[agent_id].x = 0;
        if (agents[agent_id].x >= WIDTH) agents[agent_id].x = WIDTH - 0.001f;
        if (agents[agent_id].y < 0) agents[agent_id].y = 0;
        if (agents[agent_id].y >= HEIGHT) agents[agent_id].y = HEIGHT - 0.001f;

        int agent_new_i =  (int)(agents[agent_id].x / dx_grid);
        int agent_new_j =  (int)(agents[agent_id].y / dy_grid);
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
    float phi = get_local_pheromone(agents[agent_id], phi_grid);
    if(agents[agent_id].state_duration>1 && agents[agent_id].state==2 ){//&& agents[agent_id].neighbor_count>0){ //only consider early exit for run state
        TransitionModel exit_model = d_exit_models[agents[agent_id].state];
        //use exit model to determine if the agent should exit the state early -- it's a logistic function on the number of neighbors
        float z_exit =
                exit_model.coeff * phi
                + exit_model.intercept;

        float p_exit =
                exit_model.height /
                (1.0f + expf(-z_exit));
        //float p_exit = exit_model.height / (1.0f + expf(-exit_model.coeff * (float)agents[agent_id].delta_neighbor_count + exit_model.intercept));
        //float p_exit = exit_model.height / (1.0f + expf(-exit_model.coeff * (float)agents[agent_id].neighbor_count + exit_model.intercept));

        float u = curand_uniform(&rng_states[agent_id]);
        if (u < p_exit) {
            //set duration to 0
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
        const TransitionModel& model =
                d_transition_models[agent_state * N_STATES + i];

        if (model.coeff == -1 && model.intercept == -1)
        {
            p[i] = model.p_off_food;
            p_irr += p[i];
            p_r_raw[i] = 0.0f;
        }
        else
        {
            float z =
                    model.coeff * phi
                    + model.intercept;

            float height = model.height;

            float val =
                    height / (1.0f + expf(-z));

            p_r_raw[i] = val;
            sum_r += val;
        }
    }

    // PASS 2: normalize ONLY relevant transitions
    float remaining_mass = 1.0f - p_irr;

    if (sum_r > 0.0f && remaining_mass > 0.0f)
    {
        for (int i = 0; i < N_STATES; i++)
        {
            const TransitionModel& model =
                    d_transition_models[agent_state * N_STATES + i];

            if (!(model.coeff==-1 && model.intercept==-1 ))// || agents[agent_id].neighbor_count>0)//|| fabsf(agents[agent_id].accumulated_dc_tot) < ODOR_THRESHOLD))
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


#endif //UNTITLED2_AGENT_UPDATE_H
