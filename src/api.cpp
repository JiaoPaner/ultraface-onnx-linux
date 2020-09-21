//
// Created by jiaopan on 7/15/20.
//

#include "api.h"
static Detector detector;

/**
 * init
 * @param model_path
 * @param num_threads
 * @return
 */
int init(const char* model_path,int num_threads) {
    return detector.init(model_path, num_threads);
}
int unload(const char* type) {
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
    std::cout << "=====================python=====================" << std::endl;
    std::cout
    <<"import ctypes\n"
    <<"so = ctypes.cdll.LoadLibrary(\"/home/jiaopan/projects/c++/cv_algorithm/cmake-build-debug/libcv_algorithm.so\");\n"
    <<"so.testApi();\n"
    <<"model_path = bytes(\"/home/jiaopan/projects/c++/onnx_test/squeezenet1.0-8.onnx\",\"utf-8\")\n"
    <<"code = so.init(model_path,bytes(\"squeezenet\",\"utf-8\"));\n"
    <<"print(code)\n"
    <<"file = bytes(\"/home/jiaopan/Downloads/OIP.PsqDpksIRqsptoeBxA4ZSgHaFj.jpeg\",\"utf-8\")\n"
    <<"result = so.classifyByFile(file)\n"
    <<"result = ctypes.string_at(result, -1).decode(\"utf-8\")\n"
    <<"print(result)\n"
    <<R"({"code":0,"msg":"success","data":[{"label":"Labrador retriever","score":0.638810}]})"
    << std::endl;
    std::cout << "================================================" << std::endl;
}

int main(){

    //test detectByFile()
    const char* model_path = "/home/jiaopan/projects/c++/retinaface-onnx-linux/model/retinaface.onnx";
    int status = init(model_path,1);
    std::cout << "status" << status << std::endl;
    char* result = detectByFile("/home/jiaopan/Downloads/zidane.jpg",0.5);
    std::cout << "result:" << result << std::endl;
    //std::cout << "unload:" << unload("detector") << std::endl;

    /*
    cv::Mat image = cv::imread("/home/jiaopan/Downloads/bus.jpg");
    cv::Scalar colors[20] = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),cv::Scalar(0,0,255),
                              cv::Scalar(255,255,0),cv::Scalar(255,0,255), cv::Scalar(0,255,255),
                              cv::Scalar(255,255,255), cv::Scalar(127,0,0),cv::Scalar(0,127,0),
                              cv::Scalar(0,0,127),cv::Scalar(127,127,0), cv::Scalar(127,0,127),
                              cv::Scalar(0,127,127), cv::Scalar(127,127,127),cv::Scalar(127,255,0),
                              cv::Scalar(127,0,255),cv::Scalar(127,255,255), cv::Scalar(0,127,255),
                              cv::Scalar(255,127,0), cv::Scalar(0,255,127) };  //ÑÕÉ«

    cJSON *root;
    root = cJSON_Parse(result);
    cJSON *code = cJSON_GetObjectItem(root, "code");

    if (code->valueint == 0) {
        cJSON *data = cJSON_GetObjectItem(root, "data");
        int size = cJSON_GetArraySize(data);
        cv::Rect rect;
        for (int i = 0; i < size; ++i) {
            cJSON *item = cJSON_GetArrayItem(data, i);
            cJSON *label = cJSON_GetObjectItem(item, "label");
            std::cout << label->valuestring << std::endl;
            cJSON *location = cJSON_GetObjectItem(item, "location");
            rect = cv::Rect(cJSON_GetObjectItem(location, "x")->valuedouble, cJSON_GetObjectItem(location, "y")->valuedouble,
                            cJSON_GetObjectItem(location, "width")->valuedouble, cJSON_GetObjectItem(location, "height")->valuedouble);
            rectangle(image, rect, cv::Scalar(colors[i % 4]), 3, 1, 0);
            putText(image, label->valuestring, cv::Point(rect.x + 5, rect.y + 13), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(colors[i % 4]), 1, 8);

        }
    }
    imwrite("output.jpg", image);
*/
    // unload("detector");//unload loaded model
    std::cin.get();
    return 0;
}