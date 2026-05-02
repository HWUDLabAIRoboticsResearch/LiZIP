#ifndef TRT_PREDICTOR_HPP
#define TRT_PREDICTOR_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>
#include <string>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

static TRTLogger gLogger;

class TRTPredictor {
public:
    TRTPredictor() : engine(nullptr), context(nullptr), runtime(nullptr), d_input(nullptr), d_output(nullptr), input_dim(0), max_batch(0) {}

    ~TRTPredictor() {
        if (d_input) CHECK_CUDA(cudaFree(d_input));
        if (d_output) CHECK_CUDA(cudaFree(d_output));
    }

    bool load(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) return false;

        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);

        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);

        runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) return false;

        engine = runtime->deserializeCudaEngine(engine_data.data(), size);
        if (!engine) return false;

        context = engine->createExecutionContext();
        if (!context) return false;

        auto input_shape = engine->getTensorShape("input");
        input_dim = input_shape.d[input_shape.nbDims - 1]; 
        max_batch = 0;

        return true;
    }

    void predict_batch(const float* input, float* output, int batch_size) {
        if (batch_size > max_batch) {
            if (d_input) CHECK_CUDA(cudaFree(d_input));
            if (d_output) CHECK_CUDA(cudaFree(d_output));
            CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_dim * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_output, batch_size * 3 * sizeof(float)));
            max_batch = batch_size;
        }

        CHECK_CUDA(cudaMemcpy(d_input, input, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));

        auto shape = engine->getTensorShape("input");
        shape.d[0] = batch_size;
        context->setInputShape("input", shape);

        context->setTensorAddress("input", d_input);
        context->setTensorAddress("output", d_output);

        context->enqueueV3(0);

        CHECK_CUDA(cudaMemcpy(output, d_output, batch_size * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void predict_batch_async(const float* input, float* output, int batch_size, cudaStream_t stream) {
        if (batch_size > max_batch) {
            if (d_input) CHECK_CUDA(cudaFree(d_input));
            if (d_output) CHECK_CUDA(cudaFree(d_output));
            CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_dim * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_output, batch_size * 3 * sizeof(float)));
            max_batch = batch_size;
        }

        CHECK_CUDA(cudaMemcpyAsync(d_input, input, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice, stream));

        auto shape = engine->getTensorShape("input");
        shape.d[0] = batch_size;
        context->setInputShape("input", shape);

        context->setTensorAddress("input", d_input);
        context->setTensorAddress("output", d_output);

        context->enqueueV3(stream);

        CHECK_CUDA(cudaMemcpyAsync(output, d_output, batch_size * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }

    void predict(const float* input, float* output) {
        predict_batch(input, output, 1);
    }

    int get_input_dim() const { return input_dim; }

private:
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    void* d_input;
    void* d_output;
    int input_dim;
    int max_batch;
};

#endif
