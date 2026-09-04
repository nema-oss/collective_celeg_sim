//
// Created by nema on 17/08/26.
//

#ifndef UNTITLED2_LOGGING_H
#define UNTITLED2_LOGGING_H
#include <cuda_runtime.h>
#include "parameters.h"

using json = nlohmann::json;

// --- Streaming binary writer helpers -----------------------------------

FILE* open_grid_stream(const char* filename, int n_steps, int grid_n) {
    FILE* f = fopen(filename, "wb");
    if (!f) throw std::runtime_error(std::string("Cannot open ") + filename);
    setvbuf(f, NULL, _IOFBF, 1 << 20); // 1MB buffer, same as log_matrix_stack
    int32_t ns = n_steps, gn = grid_n;
    fwrite(&ns, sizeof(int32_t), 1, f);
    fwrite(&gn, sizeof(int32_t), 1, f);
    return f;
}

template <typename T>
void write_grid_step(FILE* f, const T* grid, int grid_n) {
    fwrite(grid, sizeof(T), (size_t)grid_n * grid_n, f);
}

FILE* open_agent_stream(const char* filename, int n_steps, int worm_count) {
    FILE* f = fopen(filename, "wb");
    if (!f) throw std::runtime_error(std::string("Cannot open ") + filename);
    setvbuf(f, NULL, _IOFBF, 1 << 20);
    int32_t ns = n_steps, wc = worm_count;
    fwrite(&ns, sizeof(int32_t), 1, f);
    fwrite(&wc, sizeof(int32_t), 1, f);
    return f;
}


void saveAllDataToJSON(const char* filename, float* positions, int* sub_states) {
    nlohmann::json json_data;
    json_data["positions"] = nlohmann::json::array();
    json_data["states"] = nlohmann::json::array();
    for (int i = 0; i < WORM_COUNT; ++i) {
        nlohmann::json agent_data;
        agent_data["positions"] = nlohmann::json::array();
        agent_data["states"] = nlohmann::json::array();

        for (int j = 0; j < N_STEPS; ++j) {
            agent_data["positions"].push_back({positions[(j * WORM_COUNT + i) * 2], positions[(j * WORM_COUNT + i) * 2 + 1]});
            agent_data["states"].push_back(sub_states[j * WORM_COUNT + i]);
        }

        json_data["positions"].push_back(agent_data["positions"]);
        json_data["states"].push_back(agent_data["states"]);
    }

    std::ofstream file(filename);
    file << json_data.dump(4);
    file.close();
}

void saveSimulationParameters(const char* filename)
{
    nlohmann::json p;

    p["N_STATES"] = N_STATES;
    p["WORM_COUNT"] = WORM_COUNT;
    p["WIDTH"] = WIDTH;
    p["HEIGHT"] = HEIGHT;
    p["GRID_N"] = GRID_N;
    p["N_STEPS"] = N_STEPS;

    p["TASK"] = TASK;
    p["BLOCK_SIZE"] = BLOCK_SIZE;
    p["SEED"] = SEED;

    p["MAX_ALLOWED_SPEED"] = MAX_ALLOWED_SPEED;
    p["DT"] = DT;

    p["PHI_SECRETION_RATE"] = PHI_SECRETION_RATE;
    p["PHI_EVAPORATION_RATE"] = PHI_EVAPORATION_RATE;
    p["PHI_DIFFUSION_CONSTANT"] = PHI_DIFFUSION_CONSTANT;
    p["SLOWDOWN_FACTOR"] = SLOWDOWN_FACTOR;
    p["MAX_CONCENTRATION"] = MAX_CONCENTRATION;

    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error(
                std::string("Cannot open ") + filename
        );
    }

    file << p.dump(4);
}

#include <cstdio>
#include <cstdint>
#include <string>
#include <type_traits>

// Build output filename: insert "_sparse" and/or "_b" before the extension.
static std::string build_filename(const char* filename, bool sparse, bool binary) {
    std::string fname(filename);
    size_t dot = fname.find('.');
    std::string base = (dot == std::string::npos) ? fname : fname.substr(0, dot);
    std::string ext  = (dot == std::string::npos) ? ""    : fname.substr(dot); // includes '.'

    if (sparse) base += "_sparse";
    if (binary) base += "_b";

    return base + ext;
}

template <typename T>
static void log_matrix_stack(const T* grids, const char* filename, bool write_bits) {
    static_assert(std::is_same<T,int>::value || std::is_same<T,float>::value,
                  "log_matrix_stack only supports int or float grids");

    const int cell_count = GRID_N * GRID_N;

    // 1) determine sparsity: count how many timesteps are "mostly zero"
    int sparse_timestep_count = 0;
    for (int t = 0; t < N_STEPS; t++) {
        const T* mat = grids + (size_t)t * cell_count;
        int zero_count = 0;
        for (int k = 0; k < cell_count; k++)
            if (mat[k] == 0) zero_count++;
        if (zero_count > cell_count / 2) sparse_timestep_count++;
    }
    bool use_sparse = sparse_timestep_count > N_STEPS / 2;

    std::string outname = build_filename(filename, use_sparse, write_bits);
    printf("saving phi to %s\n", outname.c_str());
    FILE* f = fopen(outname.c_str(), write_bits ? "wb" : "w");
    if (!f) {
        fprintf(stderr, "log_matrix_stack: failed to open %s\n", outname.c_str());
        return;
    }
    setvbuf(f, NULL, _IOFBF, 1 << 20); // 1MB buffer

    if (use_sparse) {
        if (write_bits) {
            // binary sparse: header(n_steps, grid_n), then per timestep:
            // int32 t, int32 nnz, then nnz * (int32 i, int32 j, T value)
            int32_t n_steps = N_STEPS, grid_n = GRID_N;
            fwrite(&n_steps, sizeof(int32_t), 1, f);
            fwrite(&grid_n,  sizeof(int32_t), 1, f);

            for (int t = 0; t < N_STEPS; t++) {
                const T* mat = grids + (size_t)t * cell_count;
                int32_t nnz = 0;
                for (int k = 0; k < cell_count; k++) if (mat[k] != 0) nnz++;

                int32_t t32 = t;
                fwrite(&t32, sizeof(int32_t), 1, f);
                fwrite(&nnz, sizeof(int32_t), 1, f);

                for (int i = 0; i < GRID_N; i++) {
                    for (int j = 0; j < GRID_N; j++) {
                        T val = mat[i * GRID_N + j];
                        if (val != 0) {
                            int32_t ii = i, jj = j;
                            fwrite(&ii,  sizeof(int32_t), 1, f);
                            fwrite(&jj,  sizeof(int32_t), 1, f);
                            fwrite(&val, sizeof(T),       1, f);
                        }
                    }
                }
            }
        } else {
            // text sparse: one line per timestep: "t: (i,j):x (i,j):x ..."
            for (int t = 0; t < N_STEPS; t++) {
                const T* mat = grids + (size_t)t * cell_count;
                fprintf(f, "%d:", t);
                for (int i = 0; i < GRID_N; i++) {
                    for (int j = 0; j < GRID_N; j++) {
                        T val = mat[i * GRID_N + j];
                        if (val != 0) {
                            if constexpr (std::is_same<T,int>::value)
                            fprintf(f, " (%d,%d):%d", i, j, val);
                            else
                            fprintf(f, " (%d,%d):%g", i, j, val);
                        }
                    }
                }
                fprintf(f, "\n");
            }
        }
    } else {
        if (write_bits) {
            // binary dense: header(n_steps, grid_n), then raw N_STEPS*GRID_N*GRID_N values
            int32_t n_steps = N_STEPS, grid_n = GRID_N;
            fwrite(&n_steps, sizeof(int32_t), 1, f);
            fwrite(&grid_n,  sizeof(int32_t), 1, f);
            fwrite(grids, sizeof(T), (size_t)N_STEPS * cell_count, f);
        } else {
            // text dense: CSV rows per timestep
            for (int t = 0; t < N_STEPS; t++) {
                const T* mat = grids + (size_t)t * cell_count;
                fprintf(f, "# t=%d\n", t);
                for (int i = 0; i < GRID_N; i++) {
                    for (int j = 0; j < GRID_N; j++) {
                        if constexpr (std::is_same<T,int>::value)
                        fprintf(f, "%d%s", mat[i * GRID_N + j], (j < GRID_N - 1) ? "," : "");
                        else
                        fprintf(f, "%g%s", mat[i * GRID_N + j], (j < GRID_N - 1) ? "," : "");
                    }
                    fprintf(f, "\n");
                }
                fprintf(f, "\n");
            }
        }
    }

    fclose(f);
}

void log_matrices(int *agent_count_grids, float *pheromone_grids,
                  const char *agent_count_filename, const char *pheromone_filename,
                  bool write_bits = false) {
    log_matrix_stack<int>(agent_count_grids, agent_count_filename, write_bits);
    log_matrix_stack<float>(pheromone_grids, pheromone_filename, write_bits);
}

#endif //UNTITLED2_LOGGING_H
