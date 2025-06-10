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
    // Assumption: Output shape [1, N, 6] = [x1, y1, x2, y2, score, class]
    float* output = output_tensors[0].GetTensorMutableData<float>();
    int num_detections = output_shape[1];
    int box_size = output_shape[2];

    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int i = 0; i < num_detections; ++i) {
        float* det = output + i * box_size;
        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float score = det[4];
        int class_id = static_cast<int>(det[5]);

        if (score < conf_threshold)
            continue;

        cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
        boxes.push_back(box);
        scores.push_back(score);
        class_ids.push_back(class_id);
    }

    // Apply NMS
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, nms_indices);

    // Scale boxes back to original image size
    float x_scale = static_cast<float>(image_for_draw.cols) / input_size.width;
    float y_scale = static_cast<float>(image_for_draw.rows) / input_size.height;

    for (int idx : nms_indices) {
        cv::Rect box = boxes[idx];
        box.x = static_cast<int>(box.x * x_scale);
        box.y = static_cast<int>(box.y * y_scale);
        box.width = static_cast<int>(box.width * x_scale);
        box.height = static_cast<int>(box.height * y_scale);

        cv::rectangle(image_for_draw, box, cv::Scalar(0, 255, 0), 2);
        std::string label = "Class " + std::to_string(class_ids[idx]) + ": " + std::to_string(scores[idx]);
        cv::putText(image_for_draw, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    // Show image
    cv::imshow("Detections", image_for_draw);
    cv::waitKey(0);

    return 0;
}
