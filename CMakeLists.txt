cmake_minimum_required(VERSION 3.28)
project(untitled2 CUDA C CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
add_executable(untitled2 main.cu
            headers/init_joint.h
        headers/init_agents.h
        headers/parameters.h
        headers/init_interaction.h
        headers/agent_update.h
        headers/grid_update.h
        headers/logging.h
        headers/fish_update.h
        headers/engine.h
        headers/deps.h
        headers/options.h
        headers/paths.h
        headers/baselines.h
        headers/sim_state.h
        headers/setup.h
        headers/step.h
        headers/run.h
        headers/nsga2.h)

set_target_properties(untitled2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(untitled2 stdc++)

target_link_libraries(untitled2 m)
