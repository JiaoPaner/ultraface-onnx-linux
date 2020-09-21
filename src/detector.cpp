//
// Created by jiaopan on 7/10/20.
//


#include <opencv2/opencv.hpp>
#include "cJSON.h"
#include "utils.h"
#include "onnx.h"
#include "detector.h"

int Detector::init(const char* model_path,int num_threads){
    try {
        this->unload();
        this->session = this->onnx.init(model_path, num_threads);
        return 0;
    }
    catch (const std::exception &e) {
        std::cout << "init error:" << e.what() << std::endl;
        return 1;
    }
}

char * Detector::detect(cv::Mat image, float min_score){
    std::vector<float> input_image;
    const std::map<std::string, int> params = detectorConfig::map.at("retina");
    utils::createInputImage(input_image, image, params.at("width"), params.at("height"), params.at("channels"), false);
    std::vector<std::vector<float>> images;
    images.emplace_back(input_image);
    std::vector<Ort::Value> output_tensor = this->onnx.inference(this->session, images);
    return Detector::analysis(output_tensor, image.cols, image.rows, params, min_score);
}

char *Detector::analysis(std::vector<Ort::Value> &output_tensor, int width, int height,
                         const std::map<std::string, int> &output_name_index, float min_score) {
    for (auto value : output_tensor[0].GetTensorTypeAndShapeInfo().GetShape()) {
        std::cout << value << ",";
    }
}


int Detector::unload(){
    try {
        this->session = Ort::Session(nullptr);
        this->onnx.input_names.clear();
        this->onnx.output_names.clear();
        this->onnx.inputs.clear();
        return 0;
    }
    catch (const std::exception &e) {
        std::cout << "unload error:" << e.what() << std::endl;
        return 1;
    }
}

cv::Mat Detector::generate_anchors(int base_size, const cv::Mat &ratios, const cv::Mat &scales) {
    return cv::Mat();
}



