// #include <iostream>
// #include <onnxruntime_cxx_api.h>

// int main() {
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
//     session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

//     // Load model from local path
//     const char* model_path = "subgraph3.onnx";
//     Ort::Session session(env, model_path, session_options);

//     std::cout << "ONNX model loaded successfully!" << std::endl;
//     return 0;
// }
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

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

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pose");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    const char* model_path = "subgraph3.onnx";
    Ort::Session session(env, model_path, session_options);

    // Model input info
    // Ort::AllocatorWithDefaultOptions allocator;
    // auto input_name = session.GetInputName(0, allocator);
    Ort::AllocatorWithDefaultOptions allocator;
    // Get input name and hold it
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
    cv::Size input_size(640, 640);  // change if your model requires different size
    cv::Mat image_for_draw;
    std::vector<float> input_tensor_values = preprocess_image("bus.jpg", input_size, image_for_draw);

    // Prepare input tensor shape: {1, 3, 640, 640}
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

    // Output shape and type info
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Inference successful! Output shape: [";
    for (size_t i = 0; i < output_shape.size(); ++i) {
        std::cout << output_shape[i] << (i + 1 < output_shape.size() ? ", " : "]\n");
    }
    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    // Output is (1, 56, 8400) → flatten to 56 x 8400
    const int num_channels = 56;
    const int num_preds = 8400;

    std::vector<std::vector<float>> output(num_channels, std::vector<float>(num_preds));
    for (int c = 0; c < num_channels; ++c) {
        for (int i = 0; i < num_preds; ++i) {
            output[c][i] = output_data[c * num_preds + i];
        }
    }

    // Split channels
    std::vector<std::vector<float>> boxes(4, std::vector<float>(num_preds));
    for (int i = 0; i < 4; ++i) boxes[i] = output[i];

    std::vector<float> objectness = output[4];
    std::vector<std::vector<float>> class_scores(51, std::vector<float>(num_preds));
    for (int i = 0; i < 51; ++i) class_scores[i] = output[5 + i];

    // Apply sigmoid
    auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    for (float& val : objectness) val = sigmoid(val);
    for (auto& vec : class_scores)
        for (float& val : vec) val = sigmoid(val);

    // Transpose class_scores to (8400 x 51)
    std::vector<std::vector<float>> class_scores_T(num_preds, std::vector<float>(51));
    for (int i = 0; i < num_preds; ++i)
        for (int j = 0; j < 51; ++j)
            class_scores_T[i][j] = class_scores[j][i];
    

    // Compute confidence = objectness * class_scores
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

    // Apply confidence threshold
    float conf_threshold = 0.7f;
    std::vector<std::vector<float>> final_detections;
    for (int i = 0; i < num_preds; ++i) {
        if (scores[i] < conf_threshold) continue;

        float cx = boxes[0][i];
        float cy = boxes[1][i];
        float w = boxes[2][i];
        float h = boxes[3][i];

        float x1 = cx - w / 2;
        float y1 = cy - h / 2;
        float x2 = cx + w / 2;
        float y2 = cy + h / 2;

        // Scale to original image size
        float x_scale = static_cast<float>(image_for_draw.cols) / 640.0f;
        float y_scale = static_cast<float>(image_for_draw.rows) / 640.0f;

        x1 *= x_scale;
        y1 *= y_scale;
        x2 *= x_scale;
        y2 *= y_scale;

        final_detections.push_back({x1, y1, x2, y2, scores[i], static_cast<float>(class_ids[i])});
    }

    // Run OpenCV NMS
    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;
    std::vector<int> nms_indices;

    for (const auto& det : final_detections) {
        nms_boxes.emplace_back(cv::Rect(cv::Point(det[0], det[1]), cv::Point(det[2], det[3])));
        nms_scores.push_back(det[4]);
    }

    cv::dnn::NMSBoxes(nms_boxes, nms_scores, 0.4, 0.5, nms_indices);

    // Draw results

    for (int idx : nms_indices) {
    const auto& det = final_detections[idx];
    int x1 = static_cast<int>(det[0]);
    int y1 = static_cast<int>(det[1]);
    int x2 = static_cast<int>(det[2]);
    int y2 = static_cast<int>(det[3]);
    float score = det[4];
    int cls_id = static_cast<int>(det[5]);

    // Draw bounding box
    cv::rectangle(image_for_draw, cv::Point(x1, y1), cv::Point(x2, y2), {0, 255, 0}, 2);

    // Print class 0 ("person") score for this prediction
    float person_score = class_scores[idx][0];  // [class][prediction index]
    std::ostringstream oss;
    oss << "Cls: " << cls_id << " " << person_score;
    std::string label = oss.str();

    cv::putText(image_for_draw, label, {x1, y1 - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);
}

    cv::cvtColor(image_for_draw, image_for_draw, cv::COLOR_RGB2BGR);
    cv::imwrite("output_subgraph3.jpg", image_for_draw);
    std::cout << "Saved detection image to output_subgraph3.jpg\n";

    // // Show image
    // cv::imshow("Detections", image_for_draw);
    // cv::waitKey(0);

    return 0;
}
