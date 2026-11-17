//
// Created by nema on 31/07/26.
//

#ifndef UNTITLED2_PARAMETERS_H
#define UNTITLED2_PARAMETERS_H
#define N_STATES 3
#define WORM_COUNT 1000
#define WIDTH 10 //mm
#define HEIGHT 10  //mm
#define GRID_N 100 //dx = WIDTH/GRID_N [mm]
#define N_STEPS 1800
#define TASK "aggregation"
#define BLOCK_SIZE 32
#define SEED 1333
std::string* state_ids = new std::string[N_STATES]{"0", "1", "2"};
#define MAX_ALLOWED_SPEED 0.5f
#define DT 0.33f//0.33
#define PHI_SECRETION_RATE 1.0f // chosen s.t. secretion>evaporation
#define PHI_EVAPORATION_RATE 0.1f //chosen for the half life to be 10s
#define PHI_DIFFUSION_CONSTANT 0.004f // chosen s.t. the typical radius is 0.2mm: d = (0.2mm)^2 * k_evap = 0.004 mm^2/s.
//for phi to admit solution, time and space resolution must respect (missing source)
//D * dt (1/dx^2+1/dy^2) <= 1/2
// 0.004 * 0.33 (200) <=? 1/2
// 0.264 < 1/2
#define SLOWDOWN_FACTOR 0.01f//0.01f
#define MAX_CONCENTRATION 30000000000.0f
#define BOUNDARY_TYPE 0 //or "PERIODIC":0,"WALL":1
#define TRANSITION_TYPE 0 //0: simple L1L2, 1: compound L1L2, -1: constant mean matrix
#define LAMBDA 1.0f //0->dispersal/antisocial, 1->aggregation/social
#define METRIC_TYPE 1 // 0->cluster metric, 1->compound metric
#define ANIMAL_TYPE 0 // 0->worm, 1-> fish
#define FOOD_AMOUNT 1.0f
#define FOOD_CONSUMPTION 0.0055555556f//0.0111111111f//
#define ENABLE_DETERMINISTIC_REPLAY 0
#define LOG_EVERY 1
#define INTEGRATION_MODEL 0// 0: input integration (z=phi+food); 1: output integration (z=L(phi)+T(food))
#define LEAN_LOG 1 //if set to 1, saves only positions, neighbor count and food grid.

struct SimConfig {
    int   integration_model  = 0;
    float food_amount        = 0.0f;
    float food_consumption   = 0.0f;
};

__constant__ SimConfig c_cfg;


#endif //UNTITLED2_PARAMETERS_H
