//
// Created by jiaopan on 7/9/20.
//

#include "onnx.h"
Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "inference" };
/**
 * init session,load model
 * @param model_path
 * @param num_threads
 * @return
 */
Ort::Session OnnxInstance::init(std::string model_path,int num_threads) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::SessionOptions session_option;
    session_option.SetIntraOpNumThreads(num_threads);
    session_option.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Ort::Session session(env, model_path.c_str(), session_option);
    this->createInputsInfo(session);
    this->createOutputsInfo(session);
    return session;
}
/**
 * get model input info
 * @param session
 */
void OnnxInstance::createInputsInfo(Ort::Session &session) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    const std::vector<Input> inputs;
    for (int i = 0; i < num_input_nodes; i++) {
        struct Input input;
        char* input_name = session.GetInputName(i, allocator);
        input.name = input_name;
        this->input_names.emplace_back(input_name);
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input.dims = tensor_info.GetShape();
        this->inputs.emplace_back(input);
    }
}
/**
 * get model output info
 * @param session
 */
void OnnxInstance::createOutputsInfo(Ort::Session &session) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_output_nodes = session.GetOutputCount();
    for (int i = 0; i < num_output_nodes; i++) {
        char* output_name = session.GetOutputName(i, allocator);
        this->output_names.emplace_back(output_name);
    }
}
/**
 * inference
 * @param session
 * @param images
 * @return
 */
std::vector<Ort::Value> OnnxInstance::inference(Ort::Session &session,std::vector<std::vector<float>> images) {
    if(images.size() != this->inputs.size())
        throw "images size != inputs size";
    for (int i = 0; i < images.size(); ++i) {
        this->inputs[i].values = images[i];
    }
    std::vector<Ort::Value> ort_inputs;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    for (int j = 0; j < this->inputs.size(); ++j) {
        ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(memory_info,
                this->inputs[j].values.data(),this->inputs[j].values.size(),
                this->inputs[j].dims.data(),this->inputs[j].dims.size()));

    }
    std::vector<Ort::Value> ort_outputs = session.Run(nullptr,
            this->input_names.data(), ort_inputs.data(),
            ort_inputs.size(), output_names.data(), output_names.size());
    return ort_outputs;
}


