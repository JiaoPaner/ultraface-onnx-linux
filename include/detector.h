//
// Created by jiaopan on 7/10/20.
//

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include "constants.h"


class Detector {
    public:
        int init(const char* model_path,int num_threads);
        char* detect(cv::Mat image, float min_score = 0.9f);
        char* analysis(std::vector<Ort::Value> &output_tensor,int width, int height, const std::map<std::string,int> &params,float min_score=0.9f);
        int unload();

        ~Detector(){};
    private:
        OnnxInstance onnx;
        Ort::Session session = Ort::Session(nullptr);
};

