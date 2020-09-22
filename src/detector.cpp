//
// Created by jiaopan on 7/10/20.
//


#include <opencv2/opencv.hpp>
#include "cJSON.h"
#include "onnx.h"
#include "detector.h"
#include "config.h"
#include "utils.h"

anchor_win _whctrs(anchor_box anchor){
    //Return width, height, x center, and y center for an anchor (window).
    anchor_win win;
    win.w = anchor.x2 - anchor.x1 + 1;
    win.h = anchor.y2 - anchor.y1 + 1;
    win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
    win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

    return win;
}

anchor_box _mkanchors(anchor_win win){
    //Given a vector of widths (ws) and heights (hs) around a center
    //(x_ctr, y_ctr), output a set of anchors (windows).
    anchor_box anchor;
    anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
    anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
    anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
    anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

    return anchor;
}

std::vector<anchor_box> _ratio_enum(anchor_box anchor, std::vector<float> ratios){
    //Enumerate a set of anchors for each aspect ratio wrt an anchor.
    std::vector<anchor_box> anchors;
    for(size_t i = 0; i < ratios.size(); i++) {
        anchor_win win = _whctrs(anchor);
        float size = win.w * win.h;
        float scale = size / ratios[i];

        win.w = std::round(sqrt(scale));
        win.h = std::round(win.w * ratios[i]);

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

std::vector<anchor_box> _scale_enum(anchor_box anchor, std::vector<int> scales){
    //Enumerate a set of anchors for each scale wrt an anchor.
    std::vector<anchor_box> anchors;
    for(size_t i = 0; i < scales.size(); i++) {
        anchor_win win = _whctrs(anchor);

        win.w = win.w * scales[i];
        win.h = win.h * scales[i];

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

std::vector<anchor_box> generate_anchors(int base_size = 16, std::vector<float> ratios = {0.5, 1, 2},std::vector<int> scales = {8, 64}, int stride = 16, bool dense_anchor = false){
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    anchor_box base_anchor;
    base_anchor.x1 = 0;
    base_anchor.y1 = 0;
    base_anchor.x2 = base_size - 1;
    base_anchor.y2 = base_size - 1;

    std::vector<anchor_box> ratio_anchors;
    ratio_anchors = _ratio_enum(base_anchor, ratios);

    std::vector<anchor_box> anchors;
    for(size_t i = 0; i < ratio_anchors.size(); i++) {
        std::vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }

    if(dense_anchor) {
        assert(stride % 2 == 0);
        std::vector<anchor_box> anchors2 = anchors;
        for(size_t i = 0; i < anchors2.size(); i++) {
            anchors2[i].x1 += stride / 2;
            anchors2[i].y1 += stride / 2;
            anchors2[i].x2 += stride / 2;
            anchors2[i].y2 += stride / 2;
        }
        anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
    }

    return anchors;
}

std::vector<std::vector<anchor_box>> generate_anchors_fpn(bool dense_anchor = false, std::vector<anchor_cfg> cfg = {}){
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    std::vector<std::vector<anchor_box>> anchors;
    for(size_t i = 0; i < cfg.size(); i++) {
        //stride从小到大[32 16 8]
        anchor_cfg tmp = cfg[i];
        int bs = tmp.BASE_SIZE;
        std::vector<float> ratios = tmp.RATIOS;
        std::vector<int> scales = tmp.SCALES;
        int stride = tmp.STRIDE;

        std::vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
        anchors.push_back(r);
    }

    return anchors;
}

std::vector<anchor_box> anchors_plane(int height, int width, int stride, std::vector<anchor_box> base_anchors){
    /*
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: a base set of anchors
    */

    std::vector<anchor_box> all_anchors;
    for(size_t k = 0; k < base_anchors.size(); k++) {
        for(int ih = 0; ih < height; ih++) {
            int sh = ih * stride;
            for(int iw = 0; iw < width; iw++) {
                int sw = iw * stride;

                anchor_box tmp;
                tmp.x1 = base_anchors[k].x1 + sw;
                tmp.y1 = base_anchors[k].y1 + sh;
                tmp.x2 = base_anchors[k].x2 + sw;
                tmp.y2 = base_anchors[k].y2 + sh;
                all_anchors.push_back(tmp);
            }
        }
    }

    return all_anchors;
}

void clip_boxes(std::vector<anchor_box> &boxes, int width, int height){
    //Clip boxes to image boundaries.
    for(size_t i = 0; i < boxes.size(); i++) {
        if(boxes[i].x1 < 0) {
            boxes[i].x1 = 0;
        }
        if(boxes[i].y1 < 0) {
            boxes[i].y1 = 0;
        }
        if(boxes[i].x2 > width - 1) {
            boxes[i].x2 = width - 1;
        }
        if(boxes[i].y2 > height - 1) {
            boxes[i].y2 = height -1;
        }
//        boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
//        boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
//        boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
//        boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);
    }
}

void clip_boxes(anchor_box &box, int width, int height){
    //Clip boxes to image boundaries.
    if(box.x1 < 0) {
        box.x1 = 0;
    }
    if(box.y1 < 0) {
        box.y1 = 0;
    }
    if(box.x2 > width - 1) {
        box.x2 = width - 1;
    }
    if(box.y2 > height - 1) {
        box.y2 = height -1;
    }
//    boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
//    boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
//    boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
//    boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);

}
//=========================================================================================================================
std::vector<anchor_box> bbox_pred(std::vector<anchor_box> anchors, std::vector<cv::Vec4f> regress){
    //"""
    //  Transform the set of class-agnostic boxes into class-specific boxes
    //  by applying the predicted offsets (box_deltas)
    //  :param boxes: !important [N 4]
    //  :param box_deltas: [N, 4 * num_classes]
    //  :return: [N 4 * num_classes]
    //  """

    std::vector<anchor_box> rects(anchors.size());
    for(size_t i = 0; i < anchors.size(); i++) {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        float pred_ctr_x = regress[i][0] * width + ctr_x;
        float pred_ctr_y = regress[i][1] * height + ctr_y;
        float pred_w = exp(regress[i][2]) * width;
        float pred_h = exp(regress[i][3]) * height;

        rects[i].x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
        rects[i].y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
        rects[i].x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
        rects[i].y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
    }

    return rects;
}

anchor_box bbox_pred(anchor_box anchor, cv::Vec4f regress){
    anchor_box rect;

    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = exp(regress[2]) * width;
    float pred_h = exp(regress[3]) * height;

    rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    return rect;
}

std::vector<FacePts> landmark_pred(std::vector<anchor_box> anchors, std::vector<FacePts> facePts){
    std::vector<FacePts> pts(anchors.size());
    for(size_t i = 0; i < anchors.size(); i++) {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        for(size_t j = 0; j < 5; j ++) {
            pts[i].x[j] = facePts[i].x[j] * width + ctr_x;
            pts[i].y[j] = facePts[i].y[j] * height + ctr_y;
        }
    }

    return pts;
}

FacePts landmark_pred(anchor_box anchor, FacePts facePt){
    FacePts pt;
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    for(size_t j = 0; j < 5; j ++) {
        pt.x[j] = facePt.x[j] * width + ctr_x;
        pt.y[j] = facePt.y[j] * height + ctr_y;
    }

    return pt;
}

bool CompareBBox(const FaceDetectInfo & a, const FaceDetectInfo & b){
    return a.score > b.score;
}

std::vector<FaceDetectInfo> nms(std::vector<FaceDetectInfo>& bboxes, float threshold){
    std::vector<FaceDetectInfo> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(bboxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        //如果全部执行完则返回
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        bboxes_nms.push_back(bboxes[select_idx]);
        mask_merged[select_idx] = 1;

        anchor_box select_bbox = bboxes[select_idx].rect;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            anchor_box& bbox_i = bboxes[i].rect;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;   //<- float 型不加1
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;


            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
                mask_merged[i] = 1;
            }
        }
    }

    return bboxes_nms;
}
//=========================================================================================================================

int Detector::init(const char* model_path,int num_threads){
    try {
        this->unload();
        this->session = this->onnx.init(model_path, num_threads);
        return 0;
    }
    catch (const std::exception &e) {
        std::cout << "init error:" << e.what() << std::endl;
        return 1;
    }
}

char * Detector::detect(cv::Mat image, float min_score){
    std::vector<float> input_image;
    const std::map<std::string, int> params = detectorConfig::map.at("retina");
    utils::createInputImage(input_image, image, params.at("width"), params.at("height"), params.at("channels"), false);
    std::vector<std::vector<float>> images;
    images.emplace_back(input_image);
    std::vector<Ort::Value> output_tensor = this->onnx.inference(this->session, images);
    return Detector::analysis(output_tensor, image.cols, image.rows, params, min_score);
}

char *Detector::analysis(std::vector<Ort::Value> &output_tensor, int width, int height,const std::map<std::string, int> &output_name_index, float min_score) {

    float *cpuBuffers;
    float pixel_means[3] = {0.0, 0.0, 0.0},pixel_stds[3] = {1.0, 1.0, 1.0},pixel_scale = 1.0;
    int ctx_id;
    float decay4,nms_threshold = 0.4;
    bool vote,nocrop;

    int ws = (width + 31) / 32 * 32;
    int hs = (height + 31) / 32 * 32;

    std::vector<float> _ratio = {1.0};
    std::vector<anchor_cfg> cfg;

    std::vector<int> _feat_stride_fpn = { 32, 16, 8 };
    std::map<std::string, std::vector<anchor_box>> _anchors_fpn ,_anchors;
    std::map<std::string, int> _num_anchors;

    anchor_cfg tmp;
    tmp.SCALES = {32, 16};
    tmp.BASE_SIZE = 16;
    tmp.RATIOS = _ratio;
    tmp.ALLOWED_BORDER = 9999;
    tmp.STRIDE = 32;
    cfg.push_back(tmp);

    tmp.SCALES = {8, 4};
    tmp.BASE_SIZE = 16;
    tmp.RATIOS = _ratio;
    tmp.ALLOWED_BORDER = 9999;
    tmp.STRIDE = 16;
    cfg.push_back(tmp);

    tmp.SCALES = {2, 1};
    tmp.BASE_SIZE = 16;
    tmp.RATIOS = _ratio;
    tmp.ALLOWED_BORDER = 9999;
    tmp.STRIDE = 8;
    cfg.push_back(tmp);

    bool dense_anchor = false;
    std::vector<std::vector<anchor_box>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
    std::cout << "anchors_fpn:" << anchors_fpn.size() << std::endl;
    for(size_t i = 0; i < anchors_fpn.size(); i++) {
        std::string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        _anchors_fpn[key] = anchors_fpn[i];
        _num_anchors[key] = anchors_fpn[i].size();
    }

    std::vector<FaceDetectInfo> faceInfo;
    for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
        std::string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        int stride = _feat_stride_fpn[i];
        int index = i * 3;

        size_t scores_size = output_tensor[index].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<int64_t > shape = output_tensor[index].GetTensorTypeAndShapeInfo().GetShape();
        float* scores_blob = output_tensor[index].GetTensorMutableData<float>();
        std::vector<float> scores(scores_blob, scores_blob + scores_size);

        size_t bboxs_size = output_tensor[++index].GetTensorTypeAndShapeInfo().GetElementCount();
        float* bboxs_blob = output_tensor[index].GetTensorMutableData<float>();
        std::vector<float> bboxs(bboxs_blob, bboxs_blob + bboxs_size);

        size_t landmarks_size = output_tensor[++index].GetTensorTypeAndShapeInfo().GetElementCount();
        float* landmarks_blob = output_tensor[index].GetTensorMutableData<float>();
        std::vector<float> landmarks(landmarks_blob, landmarks_blob + landmarks_size);

        int width_score = shape[2];
        int height_score = shape[3];
        size_t count = scores_size;//width_score * height_score;
        size_t num_anchor = _num_anchors[key];
        std::cout << "num_anchor:" << num_anchor << std::endl;
        std::vector<anchor_box> anchors = anchors_plane(height_score, width_score, stride, _anchors_fpn[key]);

        for(size_t num = 0; num < num_anchor; num++) {
            for(size_t j = 0; j < count; j++) {
                //置信度小于阈值跳过
                float conf = scores[j + count * num];
                std::cout << "conf:" << conf << std::endl;
                if(conf <= 0.9) {
                    continue;
                }

                cv::Vec4f regress;
                float dx = bboxs[j + count * (0 + num * 4)];
                float dy = bboxs[j + count * (1 + num * 4)];
                float dw = bboxs[j + count * (2 + num * 4)];
                float dh = bboxs[j + count * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);

                //回归人脸框
                anchor_box rect = bbox_pred(anchors[j + count * num], regress);
                //越界处理
                clip_boxes(rect, ws, hs);

                FacePts pts;
                for(size_t k = 0; k < 5; k++) {
                    pts.x[k] = landmarks[j + count * (num * 10 + k * 2)];
                    pts.y[k] = landmarks[j + count * (num * 10 + k * 2 + 1)];
                }
                //回归人脸关键点
                FacePts landmarks = landmark_pred(anchors[j + count * num], pts);

                FaceDetectInfo tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                faceInfo.push_back(tmp);
            }
        }
    }
    std::cout << "faceInfo.size():" <<faceInfo.size() << std::endl;
    faceInfo = nms(faceInfo, nms_threshold);
    std::cout << "faceInfo.size():" <<faceInfo.size() << std::endl;
    cv::Mat image = cv::imread("/home/jiaopan/Downloads/zidane.jpg");
    for(size_t i = 0; i < faceInfo.size(); i++) {
        cv::Rect rect = cv::Rect(cv::Point2f(faceInfo[i].rect.x1, faceInfo[i].rect.y1), cv::Point2f(faceInfo[i].rect.x2, faceInfo[i].rect.y2));
        cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);

        for(size_t j = 0; j < 5; j++) {
            cv::Point2f pt = cv::Point2f(faceInfo[i].pts.x[j], faceInfo[i].pts.y[j]);
            cv::circle(image, pt, 1, cv::Scalar(0, 255, 0), 2);
        }
    }

    //cv::waitKey(0);
    cv::imwrite("out.jpg",image);
    return nullptr;
}


int Detector::unload(){
    try {
        this->session = Ort::Session(nullptr);
        this->onnx.input_names.clear();
        this->onnx.output_names.clear();
        this->onnx.inputs.clear();
        return 0;
    }
    catch (const std::exception &e) {
        std::cout << "unload error:" << e.what() << std::endl;
        return 1;
    }
}


