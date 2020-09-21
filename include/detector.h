//
// Created by jiaopan on 7/10/20.
//

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include "constants.h"


class Detector {
    public:
        int init(const char* model_path,int num_threads);
        char* detect(cv::Mat image, float min_score);
        char* analysis(std::vector<Ort::Value> &output_tensor,int width, int height, const std::map<std::string,int> &output_name_index,float min_score=0.3f);
        int unload();

        cv::Mat generate_anchors(int base_size, const cv::Mat &ratios, const cv::Mat &scales);

        ~Detector(){};
    private:
        OnnxInstance onnx;
        Ort::Session session = Ort::Session(nullptr);
};

