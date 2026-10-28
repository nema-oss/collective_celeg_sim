//
// Created by nema on 31/07/26.
//

#ifndef UNTITLED2_PARAMETERS_H
#define UNTITLED2_PARAMETERS_H
#define N_STATES 3
#define WORM_COUNT 2
#define WIDTH 3500 //10mm
#define HEIGHT 3500  //10mm
#define GRID_N 1 //dx = 0.1mm
#define N_STEPS 300
#define TASK "aggregation"
#define BLOCK_SIZE 32
#define SEED 1333
std::string* state_ids = new std::string[N_STATES]{"0", "1", "2"};
#define MAX_ALLOWED_SPEED 300000.0f
#define DT 1.0f//0.33
#define PHI_SECRETION_RATE 1.0f // chosen s.t. secretion>evaporation
#define PHI_EVAPORATION_RATE 0.1f //chosen for the half life to be 10s
#define PHI_DIFFUSION_CONSTANT 0.004f // chosen s.t. the typical radius is 0.2mm: d = (0.2mm)^2 * k_evap = 0.004 mm^2/s.
//for phi to admit solution, time and space resolution must respect (missing source)
//D * dt (1/dx^2+1/dy^2) <= 1/2
// 0.004 * 0.33 (200) <=? 1/2
// 0.264 < 1/2
#define SLOWDOWN_FACTOR 0.01f
#define MAX_CONCENTRATION 30000000000.0f
#define BOUNDARY_TYPE 0 //or "PERIODIC":0,"WALL":1
#define TRANSITION_TYPE 0 //0: simple L1L2, 1: compound L1L2, -1: (fish) constant mean matrix
#define LAMBDA 0.0f //0->dispersal, 1->aggregation
#define METRIC_TYPE 0 // 0->cluster metric, 1->compound metric
#define ANIMAL_TYPE 1 // 0->worm, 1-> fish
#define ENABLE_DETERMINISTIC_REPLAY 0
#define LOG_EVERY 1
#endif //UNTITLED2_PARAMETERS_H
