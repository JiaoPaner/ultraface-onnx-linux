//
// Created by jiaopan on 7/15/20.
//
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "onnx.h"
#include "cJSON.h"
#include "constants.h"
#include "detector.h"


extern "C" {

    /**
     * common api
     */
    int init(const char* model_path,int num_threads = 1);
    int unload();

    /**
     * detection api
     */
    char* detectByBase64(const char* base64_data, float min_score = 0.9);
    char* detectByFile(const char* file, float min_score = 0.9);


    /**
     * usages
     */
    void getUsages();
}


