//
// Created by jiaopan on 7/9/20.
//

#include <opencv2/opencv.hpp>
#include "constants.h"
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))
#define num_feature_map 4
class utils {
    public:
        static void createInputImage(std::vector<float> &input,cv::Mat image,const int width=224,int height=224, int channels=3,bool normalization=true);
        static void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output,float iou_threshold=0.3f);
        static void generateAnchors(std::vector<std::vector<float>> &anchors,int width, int height);
        static std::string base64Decode(const char* Data, int DataByte);
        static cv::Mat base64ToMat(std::string &base64_data);
};
