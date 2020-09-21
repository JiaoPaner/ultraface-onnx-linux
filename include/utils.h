//
// Created by jiaopan on 7/9/20.
//

#include <opencv2/opencv.hpp>


class utils {
    public:
        static void createInputImage(std::vector<float> &input,cv::Mat image,const int width=224,int height=224, int channels=3,bool normalization=true);
        static void createYolov5InputImage(std::vector<float> &input,cv::Mat image,const int width=640,int height=640, int channels=3);
        static void createReIdInputImage(std::vector<float>& input, cv::Mat image, int channels, int height,int width, bool normalization, bool flip);
        static void nms(const std::vector<cv::Rect>& srcRects,std::vector<cv::Rect>& resRects,std::vector<int>& resIndexs,float thresh=0.5f);
        static std::string base64Decode(const char* Data, int DataByte);
        static cv::Mat base64ToMat(std::string &base64_data);
        static std::string arrayToString(float* array, int length, std::string seq=",");
        static float* stringToArray(std::string str);
};
