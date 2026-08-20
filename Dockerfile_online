FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    nano \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    curl \
    g++ \
    make \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3.tar.gz | tar zx && \
    cd cmake-3.28.3 && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf cmake-3.28.3


# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Make a build directory
RUN cmake -S . -B build && cmake --build build

# Set entrypoint to run the compiled binary
CMD ["./build/untitled"]
