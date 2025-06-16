#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Modified to accept cv::Mat instead of path
std::vector<float> preprocess_image(const cv::Mat& img_input, const cv::Size& input_size, cv::Mat& out_img) {
    cv::Mat img;
    cv::cvtColor(img_input, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, input_size);
    out_img = img.clone();

    img.convertTo(img, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    std::vector<float> input_tensor;
    for (int c = 0; c < 3; ++c) {
        input_tensor.insert(input_tensor.end(),
                            (float*)channels[c].datastart,
                            (float*)channels[c].dataend);
    }

    return input_tensor;
}

#include <opencv2/opencv.hpp>

// Function to compute IOU between two boxes
float iou(const cv::Rect2f& box1, const cv::Rect2f& box2) {
    float inter_area = (box1 & box2).area();
    float union_area = box1.area() + box2.area() - inter_area;
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

// COCO keypoint connection pairs
const std::vector<std::pair<int, int>> COCO_PAIRS = {
    {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {11, 13}, {13, 15}, {12, 14}, {14, 16},
    {5, 6}, {11, 12}, {5, 11}, {6, 12},
    {0, 1}, {1, 3}, {0, 2}, {2, 4},
    {0, 5}, {0, 6}
};

// Helper to convert xywh to xyxy
cv::Rect2f xywh_to_xyxy(const cv::Rect2f& box) {
    float x = box.x;
    float y = box.y;
    float w = box.width;
    float h = box.height;
    return cv::Rect2f(x - w/2, y - h/2, w, h);
}

// Draw pose keypoints and lines
void draw_pose(cv::Mat& img,
               const std::vector<std::vector<float>>& detections,
               const std::vector<std::vector<float>>& keypoints_output,
               const std::vector<cv::Rect2f>& model_boxes,
               float w0, float h0) {
    
    for (const auto& det : detections) {
        cv::Rect2f det_box(det[0], det[1], det[2] - det[0], det[3] - det[1]);
        
        float best_iou = 0.0f;
        int best_idx = -1;
        for (int i = 0; i < model_boxes.size(); ++i) {
            float iou_score = iou(det_box, xywh_to_xyxy(model_boxes[i]));
            if (iou_score > best_iou && iou_score > 0.5f) {
                best_iou = iou_score;
                best_idx = i;
            }
        }

        if (best_idx == -1) continue;

        const auto& flat_kpts = keypoints_output[best_idx];
        if (flat_kpts.size() != 51) continue;

        std::vector<cv::Point2f> keypoints(17);
        std::vector<float> confs(17);

        for (int k = 0; k < 17; ++k) {
            float x = flat_kpts[k * 3] * w0 / 640.0f;
            float y = flat_kpts[k * 3 + 1] * h0 / 640.0f;
            float conf = flat_kpts[k * 3 + 2];
            keypoints[k] = cv::Point2f(x, y);
            confs[k] = conf;
        }

        for (int i = 0; i < 17; ++i) {
            if (confs[i] > 0.3f) {
                cv::circle(img, keypoints[i], 3, cv::Scalar(0, 0, 255), -1);
            }
        }

        for (const auto& [start_idx, end_idx] : COCO_PAIRS) {
            if (confs[start_idx] > 0.3f && confs[end_idx] > 0.3f) {
                cv::line(img, keypoints[start_idx], keypoints[end_idx], cv::Scalar(255, 0, 0), 2);
            }
        }
    }
}


int main() {
    // Load ONNX model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pose");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    Ort::Session session(env, "subgraph3.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    const char* output_name = output_name_ptr.get();
    std::vector<const char*> input_names = {input_name};
    std::vector<const char*> output_names = {output_name};

    std::array<int64_t, 4> input_shape = {1, 3, 640, 640};
    cv::Size input_size(640, 640);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Open input video
    cv::VideoCapture cap("input_video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input_video.mp4\n";
        return -1;
    }

    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("output_video.mp4", cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(w, h));

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat image_for_draw;
        std::vector<float> input_tensor_values = preprocess_image(frame, input_size, image_for_draw);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_names.data(), &input_tensor, 1,
                                          output_names.data(), 1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        const int num_channels = 56;
        const int num_preds = 8400;

        std::vector<std::vector<float>> output(num_channels, std::vector<float>(num_preds));
        for (int c = 0; c < num_channels; ++c)
            for (int i = 0; i < num_preds; ++i)
                output[c][i] = output_data[c * num_preds + i];

        std::vector<std::vector<float>> boxes(4, std::vector<float>(num_preds));
        for (int i = 0; i < 4; ++i) boxes[i] = output[i];

        std::vector<float> objectness = output[4];
        std::vector<std::vector<float>> class_scores(51, std::vector<float>(num_preds));
        for (int i = 0; i < 51; ++i) class_scores[i] = output[5 + i];

        auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        for (float& val : objectness) val = sigmoid(val);
        for (auto& vec : class_scores)
            for (float& val : vec) val = sigmoid(val);

        std::vector<std::vector<float>> class_scores_T(num_preds, std::vector<float>(51));
        for (int i = 0; i < num_preds; ++i)
            for (int j = 0; j < 51; ++j)
                class_scores_T[i][j] = class_scores[j][i];

        std::vector<float> scores(num_preds);
        std::vector<int> class_ids(num_preds);
        for (int i = 0; i < num_preds; ++i) {
            float max_score = 0;
            int cls_id = 0;
            for (int j = 0; j < 51; ++j) {
                float conf = objectness[i] * class_scores_T[i][j];
                if (conf > max_score) {
                    max_score = conf;
                    cls_id = j;
                }
            }
            scores[i] = max_score;
            class_ids[i] = cls_id;
        }

        float conf_threshold = 0.6f;
        std::vector<std::vector<float>> final_detections;
        for (int i = 0; i < num_preds; ++i) {
            if (scores[i] < conf_threshold) continue;
            float cx = boxes[0][i], cy = boxes[1][i], w = boxes[2][i], h = boxes[3][i];
            float x1 = cx - w / 2, y1 = cy - h / 2, x2 = cx + w / 2, y2 = cy + h / 2;

            float x_scale = static_cast<float>(frame.cols) / 640.0f;
            float y_scale = static_cast<float>(frame.rows) / 640.0f;
            x1 *= x_scale; y1 *= y_scale;
            x2 *= x_scale; y2 *= y_scale;

            final_detections.push_back({x1, y1, x2, y2, scores[i], static_cast<float>(class_ids[i])});
        }

        

        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_indices;
        for (const auto& det : final_detections) {
            nms_boxes.emplace_back(cv::Rect(cv::Point(det[0], det[1]), cv::Point(det[2], det[3])));
            nms_scores.push_back(det[4]);
        }

        cv::dnn::NMSBoxes(nms_boxes, nms_scores, 0.4, 0.5, nms_indices);

        for (int idx : nms_indices) {
            const auto& det = final_detections[idx];
            int x1 = static_cast<int>(det[0]);
            int y1 = static_cast<int>(det[1]);
            int x2 = static_cast<int>(det[2]);
            int y2 = static_cast<int>(det[3]);
            int cls_id = static_cast<int>(det[5]);
            float score = det[4];

            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), {0, 255, 0}, 2);

            std::ostringstream oss;


            float person_score = class_scores[idx][0];  // [class][prediction index]
            //std::ostringstream oss;
            oss << "Cls: " << cls_id << " " << person_score;
            //oss << "Cls: " << cls_id << " " << score;
            cv::putText(frame, oss.str(), {x1, y1 - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
        }


        



        std::vector<cv::Rect2f> model_boxes;
        for (int i = 0; i < num_preds; ++i) {
            float cx = boxes[0][i] * (float)frame.cols / 640.0f;
            float cy = boxes[1][i] * (float)frame.rows / 640.0f;
            float w = boxes[2][i] * (float)frame.cols / 640.0f;
            float h = boxes[3][i] * (float)frame.rows / 640.0f;
            model_boxes.emplace_back(cv::Rect2f(cx, cy, w, h));
        }

        // Prepare keypoints_output
        std::vector<std::vector<float>> keypoints_output;
        for (int i = 0; i < num_preds; ++i) {
            std::vector<float> keypoints;
            for (int c = 5; c < 56; ++c) {
                keypoints.push_back(output[c][i]);
            }
            keypoints_output.push_back(keypoints);
        }

        // Prepare detection boxes for pose drawing (only from NMS)
        std::vector<std::vector<float>> nms_detections;
        for (int idx : nms_indices) {
            nms_detections.push_back(final_detections[idx]);
        }

        draw_pose(frame, nms_detections, keypoints_output, model_boxes, static_cast<float>(frame.cols), static_cast<float>(frame.rows));


        writer.write(frame);
        cv::imshow("Output", frame);
        if (cv::waitKey(1) == 27) break;  // Press ESC to exit
    }

    cap.release();
    writer.release();
    std::cout << "Saved output to output_video.mp4\n";

    return 0;
}
