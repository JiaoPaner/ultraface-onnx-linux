//
// Created by jiaopan on 7/9/20.
//

#ifndef OBJECT_DETECT_CONSTANTS_H
#define OBJECT_DETECT_CONSTANTS_H

#include <vector>
#include <map>
#include <stdint.h>
#include <opencv2/opencv.hpp>
struct Input {
    const char* name = nullptr;
    std::vector<int64_t> dims;
    std::vector<float> values;
};

struct FaceObject{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

namespace detectorConfig{
    const std::map<std::string,int> retina = {
            {"width",640}, {"height",640}, {"channels",3},
            {"bboxs",1},{"landmark",2},{"scores",0},
            {"min_score",.3f}
    };
    const std::map<std::string,std::map<std::string,int>> map = {
            {"retina",retina}
    };
}

#endif //OBJECT_DETECT_CONSTANTS_H
