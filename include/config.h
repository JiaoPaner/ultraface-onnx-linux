#ifndef RETINAFACE_H
#define RETINAFACE_H

#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

struct anchor_win{
    float x_ctr;
    float y_ctr;
    float w;
    float h;
};

struct anchor_box{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct FacePts{
    float x[5];
    float y[5];
};

struct FaceDetectInfo{
    float score;
    anchor_box rect;
    FacePts pts;
};

struct anchor_cfg{
    public:
        int STRIDE;
        std::vector<int> SCALES;
        int BASE_SIZE;
        std::vector<float> RATIOS;
        int ALLOWED_BORDER;

        anchor_cfg(){
            STRIDE = 0;
            SCALES.clear();
            BASE_SIZE = 0;
            RATIOS.clear();
            ALLOWED_BORDER = 0;
        }
};
#endif // RETINAFACE_H
