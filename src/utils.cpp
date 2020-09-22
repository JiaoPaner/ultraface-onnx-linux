//
// Created by jiaopan on 7/9/20.
//

#include "utils.h"
#include <iterator>
 void utils::createInputImage(std::vector<float> &input,cv::Mat image,const int width,int height, int channels,bool normalization){
     cv::Mat dst(width, height, CV_8UC3);
     cv::resize(image, dst, cv::Size(width, height));
     std::vector<float> input_image(width * height * channels,0.f);
     //float* input_data = input_image.data();
     //std::fill(input.begin(),input.end(),0.f);
     if(normalization){
         for (int channel = 0; channel < channels; channel++) {
             for (int i = 0; i < height; i++) {
                 for (int j = 0; j < width; j++) {
                     if(channel == 0) input_image[channel*height*width + i * width + j] = (dst.ptr<uchar>(i)[j * 3 + channel]) / 255.0;
                     if(channel == 1) input_image[channel*height*width + i * width + j] = (dst.ptr<uchar>(i)[j * 3 + channel]) / 255.0;
                     if(channel == 2) input_image[channel*height*width + i * width + j] = (dst.ptr<uchar>(i)[j * 3 + channel]) / 255.0;
                 }
             }
         }
     }
     else{
         for (int channel = 0; channel < channels; channel++) {
             for (int i = 0; i < height; i++) {
                 for (int j = 0; j < width; j++) {
                     if(channel == 0) input_image[channel*height*width + i * width + j] = (dst.ptr<uchar>(i)[j * 3 + channel]);
                     if(channel == 1) input_image[channel*height*width + i * width + j] = (dst.ptr<uchar>(i)[j * 3 + channel]) ;
                     if(channel == 2) input_image[channel*height*width + i * width + j] = (dst.ptr<uchar>(i)[j * 3 + channel]);
                 }
             }
         }
     }

     input = input_image;
}
void utils::nms(const std::vector<cv::Rect> &srcRects, std::vector<cv::Rect> &resRects, std::vector<int> &resIndexs,float thresh) {
    resRects.clear();
    const size_t size = srcRects.size();
    if (!size) return;
    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i){
        idxs.insert(std::pair<int, size_t>(srcRects[i].br().y, i));
    }
    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0){
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& last = srcRects[lastElem->second];
        resIndexs.push_back(lastElem->second);
        resRects.push_back(last);
        idxs.erase(lastElem);
        for (auto pos = std::begin(idxs); pos != std::end(idxs); ){
            // grab the current rectangle
            const cv::Rect& current = srcRects[pos->second];
            float intArea = (last & current).area();
            float unionArea = last.area() + current.area() - intArea;
            float overlap = intArea / unionArea;
            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh) pos = idxs.erase(pos);
            else ++pos;
        }
    }
}

cv::Mat utils::base64ToMat(std::string &base64_data) {
    cv::Mat img;
    std::string s_mat;
    s_mat = base64Decode(base64_data.data(), base64_data.size());
    std::vector<char> base64_img(s_mat.begin(), s_mat.end());
    img = cv::imdecode(base64_img,1);//CV_LOAD_IMAGE_COLOR
    return img;
}

std::string utils::base64Decode(const char *Data, int DataByte) {
    const char DecodeTable[] ={
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            62, // '+'
            0, 0, 0,
            63, // '/'
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // '0'-'9'
            0, 0, 0, 0, 0, 0, 0,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // 'A'-'Z'
            0, 0, 0, 0, 0, 0,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // 'a'-'z'
    };
    std::string strDecode;
    int nValue;
    int i = 0;
    while (i < DataByte){
        if (*Data != '\r' && *Data != '\n'){
            nValue = DecodeTable[*Data++] << 18;
            nValue += DecodeTable[*Data++] << 12;strDecode += (nValue & 0x00FF0000) >> 16;
            if (*Data != '='){
                nValue += DecodeTable[*Data++] << 6;strDecode += (nValue & 0x0000FF00) >> 8;
                if (*Data != '='){
                    nValue += DecodeTable[*Data++];
                    strDecode += nValue & 0x000000FF;
                }
            }
            i += 4;
        }
        else{
            Data++;
            i++;
        }
    }
    return strDecode;
}