#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct Detection {
    cv::Rect bbox;
    float score;
    int class_id;
};

std::vector<float> preprocess_image(const std::string& img_path, const cv::Size& input_size, cv::Mat& out_img) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + img_path);
    }

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
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

// Non-Maximum Suppression (NMS)
std::vector<Detection> apply_nms(std::vector<Detection>& detections, float iou_threshold = 0.45) {
    std::vector<Detection> filtered_detections;
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    for (size_t i = 0; i < detections.size(); ++i) {
        bool keep = true;
        for (const auto& fd : filtered_detections) {
            float intersection_area = (detections[i].bbox & fd.bbox).area();
            float union_area = detections[i].bbox.area() + fd.bbox.area() - intersection_area;
            float iou = intersection_area / union_area;
            if (iou > iou_threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            filtered_detections.push_back(detections[i]);
        }
    }

    return filtered_detections;
}

// Scale bounding boxes back to original image size
void scale_boxes(std::vector<Detection>& detections, const cv::Size& original_size, const cv::Size& model_input_size) {
    float x_scale = static_cast<float>(original_size.width) / model_input_size.width;
    float y_scale = static_cast<float>(original_size.height) / model_input_size.height;

    for (auto& det : detections) {
        det.bbox.x = static_cast<int>(det.bbox.x * x_scale);
        det.bbox.y = static_cast<int>(det.bbox.y * y_scale);
        det.bbox.width = static_cast<int>(det.bbox.width * x_scale);
        det.bbox.height = static_cast<int>(det.bbox.height * y_scale);
    }
}

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pose");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    const char* model_path = "subgraph3.onnx";
    Ort::Session session(env, model_path, session_options);

    // Model input info
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    // Get output name and hold it
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_ptr.get();

    // Keep both alive
    std::vector<Ort::AllocatedStringPtr> name_holders;
    name_holders.push_back(std::move(input_name_ptr));
    name_holders.push_back(std::move(output_name_ptr));

    // Prepare input and output name vectors
    std::vector<const char*> input_names = {input_name};
    std::vector<const char*> output_names = {output_name};

    // Preprocess image
    cv::Size input_size(640, 640);  // Model input size
    cv::Mat image_for_draw;
    std::string image_path = "bus.jpg";
    cv::Mat original_image = cv::imread(image_path);
    std::vector<float> input_tensor_values = preprocess_image(image_path, input_size, image_for_draw);

    // Prepare input tensor
    std::array<int64_t, 4> input_shape = {1, 3, input_size.height, input_size.width};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(), &input_tensor, 1,
        output_names.data(), 1);

    // Process output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    // Assuming output shape is [1, num_detections, 6] where each detection is [x1, y1, x2, y2, score, class_id]
    size_t num_detections = output_shape[1];
    constexpr size_t num_attributes = 6; // x1, y1, x2, y2, score, class_id

    float score_threshold = 0.5; // Adjust based on your needs
    std::vector<Detection> detections;

    for (size_t i = 0; i < num_detections; ++i) {
        float* detection = output_data + i * num_attributes;
        float score = detection[4];
        
        if (score > score_threshold) {
            Detection det;
            det.bbox.x = static_cast<int>(detection[0]);
            det.bbox.y = static_cast<int>(detection[1]);
            det.bbox.width = static_cast<int>(detection[2] - detection[0]);
            det.bbox.height = static_cast<int>(detection[3] - detection[1]);
            det.score = score;
            det.class_id = static_cast<int>(detection[5]);
            detections.push_back(det);
        }
    }

    // Apply NMS
    std::vector<Detection> filtered_detections = apply_nms(detections);

    // Scale boxes back to original image size
    scale_boxes(filtered_detections, original_image.size(), input_size);

    // Draw detections on original image
    for (const auto& det : filtered_detections) {
        cv::rectangle(original_image, det.bbox, cv::Scalar(0, 255, 0), 2);
        std::string label = "Class " + std::to_string(det.class_id) + ": " + std::to_string(det.score);
        cv::putText(original_image, label, cv::Point(det.bbox.x, det.bbox.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    // Save or display the result
    cv::imwrite("output.jpg", original_image);
    cv::imshow("Detections", original_image);
    cv::waitKey(0);

    return 0;
}