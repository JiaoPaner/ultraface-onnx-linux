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
void utils::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output,float iou_threshold) {
    std::sort(input.begin(), input.end(), [](const FaceInfo &base, const FaceInfo &target) { return base.score > target.score; });
    int box_num = input.size();
    std::vector<int> merged(box_num, 0);
    for (int i = 0; i < box_num; i++) {
        if (merged[i]) continue;
        std::vector<FaceInfo> buf;
        buf.push_back(input[i]);
        merged[i] = 1;
        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;
        float area0 = h0 * w0;
        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        output.push_back(buf[0]);
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

void utils::generateAnchors(std::vector<std::vector<float>> &anchors, int width, int height) {
    const std::vector<float> strides = { 8.0, 16.0, 32.0, 64.0 };
    const std::vector<std::vector<float>> min_boxes = {{10.0f,  16.0f,  24.0f},{32.0f,  48.0f},{64.0f,  96.0f},{128.0f, 192.0f, 256.0f} };
    std::vector<std::vector<float>> feature_map_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> width_height = {width,height};
    for (auto size : width_height) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        feature_map_size.push_back(fm_item);
        shrinkage_size.push_back(strides);
    }

    for (int index = 0; index < num_feature_map; index++) {
        float scale_w = width / shrinkage_size[0][index];
        float scale_h = height / shrinkage_size[1][index];
        for (int j = 0; j < feature_map_size[1][index]; j++) {
            for (int i = 0; i < feature_map_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index]) {
                    float w = k / width;
                    float h = k / height;
                    anchors.push_back({ clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1) });
                }
            }
        }
    }
}
