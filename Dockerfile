FROM cuda-env:12.2

WORKDIR /app
RUN rm -rf /app/*
# Copy only what CMake and the compiler actually need
COPY CMakeLists.txt .
COPY headers/ headers/
COPY include/ include/
COPY json/ json/
COPY main.cu .

# Configure and build in one layer to avoid intermediate cache bloat
RUN cmake -S . -B build && cmake --build build --parallel $(nproc)

ENTRYPOINT ["/app/build/untitled2"]