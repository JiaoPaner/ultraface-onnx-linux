//
// Created by jiaopan on 7/10/20.
//


#include <opencv2/opencv.hpp>
#include "cJSON.h"
#include "onnx.h"
#include "detector.h"
#include "utils.h"

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
    const std::map<std::string, int> params = detectorConfig::map.at("ultraface");
    utils::createInputImage(input_image, image, params.at("width"), params.at("height"), params.at("channels"), true);
    std::vector<std::vector<float>> images;
    images.emplace_back(input_image);
    std::vector<Ort::Value> output_tensor = this->onnx.inference(this->session, images);
    return Detector::analysis(output_tensor, image.cols, image.rows, params, min_score);
}

char *Detector::analysis(std::vector<Ort::Value> &output_tensor, int width, int height,const std::map<std::string, int> &params, float min_score) {

    const float center_variance = 0.1,size_variance = 0.2;
    float* scores = output_tensor[params.at("scores")].GetTensorMutableData<float>();
    float* bboxs = output_tensor[params.at("bboxs")].GetTensorMutableData<float>();
    std::vector<std::vector<float>> anchors;
    utils::generateAnchors(anchors,params.at("width"), params.at("height"));

    std::vector<FaceInfo> bbox,faces;
    for (int i = 0; i < anchors.size(); i++) {
        if (scores[2 * i + 1] > min_score) {
            FaceInfo rects = {0};
            float x_center = bboxs[i * 4] * center_variance * anchors[i][2] + anchors[i][0];
            float y_center = bboxs[i * 4 + 1] * center_variance * anchors[i][3] + anchors[i][1];
            float w = exp(bboxs[i * 4 + 2] * size_variance) * anchors[i][2];
            float h = exp(bboxs[i * 4 + 3] * size_variance) * anchors[i][3];

            rects.x1 = clip(x_center - w / 2.0, 1) * width;
            rects.y1 = clip(y_center - h / 2.0, 1) * height;
            rects.x2 = clip(x_center + w / 2.0, 1) * width;
            rects.y2 = clip(y_center + h / 2.0, 1) * height;
            rects.score = clip(scores[2 * i + 1], 1);
            bbox.push_back(rects);
        }
    }
    utils::nms(bbox,faces);

    cJSON  *result = cJSON_CreateObject(), *items = cJSON_CreateArray();
    for (int i = 0; i < faces.size(); ++i) {
        cJSON  *item = cJSON_CreateObject();
        cJSON_AddNumberToObject(item,"score",faces[i].score);
        cJSON  *location = cJSON_CreateObject();
        cJSON_AddNumberToObject(location,"x",faces[i].x1);
        cJSON_AddNumberToObject(location,"y",faces[i].y1);
        cJSON_AddNumberToObject(location,"width",faces[i].x2 - faces[i].x1);
        cJSON_AddNumberToObject(location,"height",faces[i].y2 - faces[i].y1);
        cJSON_AddItemToObject(item,"location",location);
        cJSON_AddItemToArray(items,item);
    }
    cJSON_AddNumberToObject(result, "code", 0);
    cJSON_AddStringToObject(result, "msg", "success");
    cJSON_AddItemToObject(result, "data", items);
    char *resultJson = cJSON_PrintUnformatted(result);
    return resultJson;
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


