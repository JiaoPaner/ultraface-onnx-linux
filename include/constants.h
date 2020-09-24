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
typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} FaceInfo;
namespace detectorConfig{
    const std::map<std::string,int> ultraface = {
            {"width",320}, {"height",240}, {"channels",3},
            {"bboxs",1},{"scores",0},
            {"min_score",.9f}
    };
    const std::map<std::string,std::map<std::string,int>> map = {
            {"ultraface",ultraface}
    };
}

#endif //OBJECT_DETECT_CONSTANTS_H
