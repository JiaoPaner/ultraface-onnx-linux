//
// Created by jiaopan on 7/15/20.
//

#include "api.h"
static Detector detector;
#include <chrono>
/**
 * init
 * @param model_path
 * @param num_threads
 * @return
 */
int init(const char* model_path,int num_threads) {
    return detector.init(model_path, num_threads);
}
int unload() {
    return detector.unload();
}


/**
 * detect
 * @return
 */
char* detectByBase64(const char* base64_data, float min_score) {
    try {
        std::string data(base64_data);
        cv::Mat image = utils::base64ToMat(data);
        return detector.detect(image, min_score);
    }
    catch (const char* msg) {
        cJSON* result = cJSON_CreateObject(), * data = cJSON_CreateArray();;
        cJSON_AddNumberToObject(result, "code", 1);
        cJSON_AddStringToObject(result, "msg", msg);
        cJSON_AddItemToObject(result, "data", data);
        return cJSON_PrintUnformatted(result);
    }
}

char* detectByFile(const char* file, float min_score) {
    try {
        cv::Mat image = cv::imread(file);
        return detector.detect(image, min_score);
    }
    catch (const char* msg) {
        cJSON* result = cJSON_CreateObject(), * data = cJSON_CreateArray();;
        cJSON_AddNumberToObject(result, "code", 1);
        cJSON_AddStringToObject(result, "msg", msg);
        cJSON_AddItemToObject(result, "data", data);
        return cJSON_PrintUnformatted(result);
    }
}


/**
 * test
 * @return
 */
void getUsages(){
    std::cout << "use api." << std::endl;
    std::cout << "result.code = 0 ---> success" << std::endl;
    std::cout << "result.code = 1 ---> error" << std::endl;
}

int main(){

    //test detectByFile()
    const char* model_path = "/home/jiaopan/projects/c++/ultraface-onnx-linux/model/version-RFB-320_without_postprocessing.onnx";
    int status = init(model_path,1);
    std::cout << "status:" << status << std::endl;

    auto start = std::chrono::steady_clock::now();
    char* result = detectByFile("/home/jiaopan/Downloads/test1.jpeg",0.9);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "elapsed:" << elapsed.count() <<"\nresult:" << result << std::endl;
    //std::cout << "unload:" << unload("detector") << std::endl;

    /**/
    cv::Mat image = cv::imread("/home/jiaopan/Downloads/test1.jpeg");
    cJSON *root;
    root = cJSON_Parse(result);
    cJSON *code = cJSON_GetObjectItem(root, "code");

    if (code->valueint == 0) {
        cJSON *data = cJSON_GetObjectItem(root, "data");
        int size = cJSON_GetArraySize(data);
        cv::Rect rect;
        for (int i = 0; i < size; ++i) {
            cJSON *item = cJSON_GetArrayItem(data, i);
            cJSON *location = cJSON_GetObjectItem(item, "location");
            rect = cv::Rect(cJSON_GetObjectItem(location, "x")->valuedouble, cJSON_GetObjectItem(location, "y")->valuedouble,
                            cJSON_GetObjectItem(location, "width")->valuedouble, cJSON_GetObjectItem(location, "height")->valuedouble);
            rectangle(image, rect, cv::Scalar(255,127,0), 2, 1, 0);
        }
    }
    imwrite("output.jpg", image);

    // unload("detector");//unload loaded model
    //std::cin.get();
    return 0;
}