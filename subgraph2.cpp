// #include <iostream>
// #include <vector>
// #include <opencv2/opencv.hpp>
// #include <onnxruntime_cxx_api.h>

// std::vector<float> preprocess_image(const std::string& img_path, const cv::Size& input_size, cv::Mat& out_img) {
//     cv::Mat img = cv::imread(img_path);
//     if (img.empty()) {
//         throw std::runtime_error("Failed to load image: " + img_path);
//     }

//     cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
//     cv::resize(img, img, input_size);
//     out_img = img.clone();

//     img.convertTo(img, CV_32F, 1.0 / 255.0);

//     std::vector<cv::Mat> channels(3);
//     cv::split(img, channels);

//     std::vector<float> input_tensor;
//     for (int c = 0; c < 3; ++c) {
//         input_tensor.insert(input_tensor.end(),
//                             (float*)channels[c].datastart,
//                             (float*)channels[c].dataend);
//     }

//     return input_tensor;
// }

// int main() {
//     // Initialize ONNX Runtime
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pose");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
//     session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

//     const char* model_path = "subgraph3.onnx";
//     Ort::Session session(env, model_path, session_options);

//     // Model input info
//     // Ort::AllocatorWithDefaultOptions allocator;
//     // auto input_name = session.GetInputName(0, allocator);
//     Ort::AllocatorWithDefaultOptions allocator;
//     // Get input name and hold it
//     Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     const char* input_name = input_name_ptr.get();
//     auto input_dims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

//     // Get output name and hold it
//     Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     const char* output_name = output_name_ptr.get();

//     // Keep both alive
//     std::vector<Ort::AllocatedStringPtr> name_holders;
//     name_holders.push_back(std::move(input_name_ptr));
//     name_holders.push_back(std::move(output_name_ptr));

//     // Prepare input and output name vectors
//     std::vector<const char*> input_names = {input_name};
//     std::vector<const char*> output_names = {output_name};



//     // Preprocess image
//     cv::Size input_size(640, 640);  // change if your model requires different size
//     cv::Mat image_for_draw;
//     std::vector<float> input_tensor_values = preprocess_image("bus.jpg", input_size, image_for_draw);

//     // Prepare input tensor shape: {1, 3, 640, 640}
//     std::array<int64_t, 4> input_shape = {1, 3, input_size.height, input_size.width};

//     Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
//         OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_shape.data(), input_shape.size());

//     // Run inference
    



//     auto output_tensors = session.Run(
//         Ort::RunOptions{nullptr},
//         input_names.data(), &input_tensor, 1,
//         output_names.data(), 1);

//     // Output shape and type info
//     auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
//     std::cout << "Inference successful! Output shape: [";
//     for (size_t i = 0; i < output_shape.size(); ++i) {
//         std::cout << output_shape[i] << (i + 1 < output_shape.size() ? ", " : "]\n");
//     }
//     float* output_data = output_tensors[0].GetTensorMutableData<float>();

//     // Output is (1, 56, 8400) → flatten to 56 x 8400
//     const int num_channels = 56;
//     const int num_preds = 8400;

//     std::vector<std::vector<float>> output(num_channels, std::vector<float>(num_preds));
//     for (int c = 0; c < num_channels; ++c) {
//         for (int i = 0; i < num_preds; ++i) {
//             output[c][i] = output_data[c * num_preds + i];
//         }
//     }

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



void print_shape(const std::vector<int64_t>& shape) {
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

std::vector<std::vector<float>> get_constant_pose() {
    std::vector<int> strides = {80, 40, 20};
    std::vector<int> temp = {80, 40, 20};

    std::vector<float> temp1, temp2;

    for (size_t i = 0; i < strides.size(); ++i) {
        for (int j = 0; j < strides[i]; ++j) {
            for (int k = 0; k < temp[i]; ++k) {
                temp1.push_back(static_cast<float>(k));
                temp2.push_back(static_cast<float>(j));
            }
        }
    }

    return {temp1, temp2};  // Shape: (2, N)
}

std::vector<std::vector<float>> get_constant(const std::array<int64_t, 4>& model_input_shape) {
    std::vector<int> strides = {8, 16, 32};
    std::vector<float> temp3, temp4;

    for (int s : strides) {
        float temp1 = static_cast<float>(model_input_shape[2]) / s;
        float temp2 = static_cast<float>(model_input_shape[3]) / s;

        for (float j = 0.5f; j < temp1; j += 1.0f) {
            for (float k = 0.5f; k < temp2; k += 1.0f) {
                temp3.push_back(k);
                temp4.push_back(j);
            }
        }
    }

    return {temp3, temp4};  // Equivalent to shape (2, N)
}

std::vector<float> get_mul_constant(const std::array<int64_t, 4>& model_input_shape) {
    std::vector<int> strides = {8, 16, 32};
    std::vector<float> constants;

    for (int s : strides) {
        int temp1 = static_cast<int>(model_input_shape[2]) / s;
        int temp2 = static_cast<int>(model_input_shape[3]) / s;
        int count = temp1 * temp2;

        for (int i = 0; i < count; ++i) {
            constants.push_back(static_cast<float>(s));
        }
    }

    return constants;  // Shape: (1, N)
}

cv::Mat get_constant_pose_mat() {
    std::vector<std::vector<float>> constants = get_constant_pose();  // shape: 2 x N
    int rows = 2;
    int cols = static_cast<int>(constants[0].size());

    cv::Mat mat(rows, cols, CV_32F);
    for (int i = 0; i < cols; ++i) {
        mat.at<float>(0, i) = constants[0][i];
        mat.at<float>(1, i) = constants[1][i];
    }

    // reshape to (1, 2, N)
    return mat.reshape(1, {1, 2, cols});
}
cv::Mat get_constant_xy_mat(const std::array<int64_t, 4>& input_shape) {
    std::vector<std::vector<float>> constants = get_constant(input_shape);  // shape: 2 x N
    int rows = 2;
    int cols = static_cast<int>(constants[0].size());

    cv::Mat mat(rows, cols, CV_32F);
    for (int i = 0; i < cols; ++i) {
        mat.at<float>(0, i) = constants[0][i];
        mat.at<float>(1, i) = constants[1][i];
    }

    // reshape to (1, 2, N)
    return mat.reshape(1, {1, 2, cols});
}

// cv::Mat get_constant_mul_mat(const std::array<int64_t, 4>& input_shape) {
//     std::vector<float> constants = get_mul_constant(input_shape);  // shape: 1 x N
//     int cols = static_cast<int>(constants.size());

//     cv::Mat mat(1, cols, CV_32F);
//     memcpy(mat.data, constants.data(), cols * sizeof(float));

//     // reshape to (1, 1, N)
//     return mat.reshape(1, {1, 1, cols});
// }

cv::Mat get_constant_mul_mat(const std::array<int64_t, 4>& input_shape) {
    std::vector<int> strides = {8, 16, 32};
    std::vector<float> constants;

    for (int s : strides) {
        int height = static_cast<int>(input_shape[2]) / s;
        int width  = static_cast<int>(input_shape[3]) / s;
        int count  = height * width;

        for (int i = 0; i < count; ++i) {
            constants.push_back(static_cast<float>(s));
        }
    }

    // Create a 1xN cv::Mat
    cv::Mat mat(constants);
    mat = mat.reshape(1, 1);  // 1 row, N cols => shape: 1 x 8400

    return mat;  // shape: 1 x 8400
}


cv::Mat get_constant_mul_mat_expanded(const std::array<int64_t, 4>& input_shape,
                                      const std::vector<int>& target_shape) {
    std::vector<int> strides = {8, 16, 32};
    std::vector<float> constants;

    // Generate base vector (length = 8400)
    for (int s : strides) {
        int height = static_cast<int>(input_shape[2]) / s;
        int width  = static_cast<int>(input_shape[3]) / s;
        int count  = height * width;

        for (int i = 0; i < count; ++i) {
            constants.push_back(static_cast<float>(s));
        }
    }

    int L = static_cast<int>(constants.size());  // e.g., 8400

    // Compute how many times we need to repeat the [1, L] row to fill target shape
    int expected_total = 1;
    for (int dim : target_shape) expected_total *= dim;

    // The last dim of target_shape must match the base length L
    CV_Assert(target_shape.back() == L);

    int repeat_count = expected_total / L;

    // Create base row: shape (1 x L)
    cv::Mat base = cv::Mat(constants).reshape(1, 1);  // shape: 1 x L

    // Repeat the base row
    std::vector<cv::Mat> repeated(repeat_count, base.clone());

    // Concatenate vertically
    cv::Mat expanded;
    cv::vconcat(repeated, expanded);  // shape: repeat_count x L

    // Reshape to final desired shape
    expanded = expanded.reshape(1, target_shape);  // use reshape with std::vector<int>

    return expanded;
}


cv::Mat get_constant_pose_expanded() {
    std::vector<int> strides = {80, 40, 20};
    std::vector<int> temp    = {80, 40, 20};

    std::vector<float> temp1;  // x-axis
    std::vector<float> temp2;  // y-axis

    for (int i = 0; i < strides.size(); ++i) {
        for (int j = 0; j < strides[i]; ++j) {
            for (int k = 0; k < temp[i]; ++k) {
                temp1.push_back(static_cast<float>(k));
                temp2.push_back(static_cast<float>(j));
            }
        }
    }

    // Create cv::Mat(2, 8400)
    cv::Mat pose_xy(2, static_cast<int>(temp1.size()), CV_32F);
    std::memcpy(pose_xy.ptr(0), temp1.data(), temp1.size() * sizeof(float));
    std::memcpy(pose_xy.ptr(1), temp2.data(), temp2.size() * sizeof(float));

    // Now pose_xy has shape (2, 8400), we want (1, 17, 2, 8400)
    const int C1 = 17;
    const int C2 = 2;
    const int L = static_cast<int>(temp1.size());  // 8400

    // Prepare vector of shape (17 × 2 × 8400)
    std::vector<cv::Mat> pose_planes;
    for (int i = 0; i < C1; ++i) {
        for (int j = 0; j < C2; ++j) {
            // Get row j (either x or y)
            cv::Mat row = pose_xy.row(j).clone();
            pose_planes.push_back(row);
        }
    }

    // Stack into shape (34 × 8400)
    cv::Mat stacked_pose;
    cv::vconcat(pose_planes, stacked_pose);  // shape: 34 x 8400

    // Reshape to (1, 17, 2, 8400)
    stacked_pose = stacked_pose.reshape(1, {1, C1, C2, L});

    return stacked_pose;
}

cv::Mat concat_along_axis3(const cv::Mat& mat1, const cv::Mat& mat2, const cv::Mat& mat3, int axis) {
    CV_Assert(mat1.dims == mat2.dims && mat2.dims == mat3.dims);
    CV_Assert(axis >= 0 && axis < mat1.dims);

    int dims = mat1.dims;

    // Ensure all dimensions match except the concatenation axis
    for (int i = 0; i < dims; ++i) {
        if (i != axis) {
            CV_Assert(mat1.size[i] == mat2.size[i]);
            CV_Assert(mat2.size[i] == mat3.size[i]);
        }
    }

    // Calculate new shape
    std::vector<int> new_sizes(dims);
    for (int i = 0; i < dims; ++i) {
        new_sizes[i] = (i == axis) ? mat1.size[i] + mat2.size[i] + mat3.size[i] : mat1.size[i];
    }

    // Allocate output
    cv::Mat output(dims, new_sizes.data(), mat1.type());

    // Helper lambda to build cv::Range for slicing
    auto build_ranges = [&](int start, int end) {
        std::vector<cv::Range> ranges(dims, cv::Range::all());
        ranges[axis] = cv::Range(start, end);
        return ranges;
    };

    // Copy mat1
    for (int i = 0; i < mat1.size[axis]; ++i) {
        auto range = build_ranges(i, i + 1);
        mat1(range).copyTo(output(range));
    }

    // Copy mat2
    int offset2 = mat1.size[axis];
    for (int i = 0; i < mat2.size[axis]; ++i) {
        auto src_range = build_ranges(i, i + 1);
        auto dst_range = build_ranges(i + offset2, i + 1 + offset2);
        mat2(src_range).copyTo(output(dst_range));
    }

    // Copy mat3
    int offset3 = mat1.size[axis] + mat2.size[axis];
    for (int i = 0; i < mat3.size[axis]; ++i) {
        auto src_range = build_ranges(i, i + 1);
        auto dst_range = build_ranges(i + offset3, i + 1 + offset3);
        mat3(src_range).copyTo(output(dst_range));
    }

    return output;
}



cv::Mat concat_along_axis(const cv::Mat& mat1, const cv::Mat& mat2, int axis) {
    CV_Assert(mat1.dims == mat2.dims);
    CV_Assert(axis >= 0 && axis < mat1.dims);

    int dims = mat1.dims;

    // Ensure compatibility across all dimensions except the concatenation axis
    for (int i = 0; i < dims; ++i) {
        if (i != axis) {
            CV_Assert(mat1.size[i] == mat2.size[i]);
        }
    }

    // Prepare new shape
    std::vector<int> new_sizes(dims);
    for (int i = 0; i < dims; ++i) {
        new_sizes[i] = (i == axis) ? mat1.size[i] + mat2.size[i] : mat1.size[i];
    }

    cv::Mat output(dims, new_sizes.data(), mat1.type());

    // Helper function to build slice ranges
    auto build_ranges = [&](int fixed_dim_start, int fixed_dim_end) {
        std::vector<cv::Range> ranges(dims, cv::Range::all());
        ranges[axis] = cv::Range(fixed_dim_start, fixed_dim_end);
        return ranges;
    };

    // Copy mat1
    for (int i = 0; i < mat1.size[axis]; ++i) {
        auto range = build_ranges(i, i + 1);
        mat1(range).copyTo(output(range));
    }

    // Copy mat2
    for (int i = 0; i < mat2.size[axis]; ++i) {
        auto src_range = build_ranges(i, i + 1);
        auto dst_range = build_ranges(i + mat1.size[axis], i + mat1.size[axis] + 1);
        mat2(src_range).copyTo(output(dst_range));
    }

    return output;
}

cv::Mat reshape_1_51_8400(const cv::Mat& input) {
    // Input shape must be (1, 17, 3, 8400)
    CV_Assert(input.dims == 4);
    CV_Assert(input.size[0] == 1);
    CV_Assert(input.size[1] == 17);
    CV_Assert(input.size[2] == 3);
    CV_Assert(input.size[3] == 8400);

    // Total elements must match: 1*17*3*8400 == 1*51*8400
    int total_elements = input.total();
    CV_Assert(total_elements == 1 * 51 * 8400);

    // Reshape to 2D: 51 x 8400 (OpenCV reshape works to 2D first)
    cv::Mat reshaped = input.reshape(1, {51, 8400});  // 2D: 51 x 8400

    // Now reshape back to 3D: 1 x 51 x 8400 (simulate with dims)
    int sizes[] = {1, 51, 8400};
    return cv::Mat(3, sizes, input.type(), reshaped.data).clone();  // Safe copy
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pose");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    const char* model_path = "subgraph1.onnx";

    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input name
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();

    std::vector<const char*> input_names = {input_name};

    // Get output names
    size_t num_outputs = session.GetOutputCount();
    std::vector<const char*> output_names;
    std::vector<Ort::AllocatedStringPtr> output_name_holders;
    for (size_t i = 0; i < num_outputs; ++i) {
        Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(i, allocator);
        output_names.push_back(out_name.get());
        output_name_holders.push_back(std::move(out_name));  // keep alive
    }

    // Preprocess input
    cv::Size input_size(640, 640);
    cv::Mat image_for_draw;
    std::vector<float> input_tensor_values = preprocess_image("bus.jpg", input_size, image_for_draw);
    std::array<int64_t, 4> input_shape = {1, 3, input_size.height, input_size.width};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names.data(), &input_tensor, 1,
                                      output_names.data(), output_names.size());

    // Print shapes of intermediate outputs
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        std::vector<int64_t> shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Intermediate Output " << i << " shape: ";
        print_shape(shape);
    }

    // Extract raw pointers from ONNX outputs
    float* concat_output_0 = output_tensors[0].GetTensorMutableData<float>();
    float* conv_output_0   = output_tensors[1].GetTensorMutableData<float>();
    float* sigmoid_out_0   = output_tensors[2].GetTensorMutableData<float>();

    // Define dimensions for OpenCV tensor views
    int dims_concat[4] = {1, 17, 3, 8400};   // 1x17x3x8400
    int dims_conv[3]   = {1, 4, 8400};       // 1x4x8400
    int dims_sigmoid[3] = {1, 1, 8400}; 

    // Wrap float* into cv::Mat with no data copy
    cv::Mat x1(4, dims_concat, CV_32F, concat_output_0);
    cv::Mat x2(3, dims_conv,   CV_32F, conv_output_0);
    cv::Mat x_sigmoid(3, dims_sigmoid, CV_32F, sigmoid_out_0);

    // Debug print to confirm dimensions
    std::cout << "x1 shape: " << x1.size[0] << " x " << x1.size[1] << " x " << x1.size[2] << " x " << x1.size[3] << std::endl;
    std::cout << "x2 shape: " << x2.size[0] << " x " << x2.size[1] << " x " << x2.size[2] << std::endl;
    std::cout << "sigmoid shape: " << x_sigmoid.size[0] << " x " << x_sigmoid.size[1] << " x " << x_sigmoid.size[2] << std::endl;


    // Slice
    // slice1_concat_l: x1[:, :, 0:2, :]
    cv::Range ranges1[] = { cv::Range::all(), cv::Range::all(), cv::Range(0, 2), cv::Range::all() };
    cv::Mat slice1_concat_l = x1(ranges1);

    // slice2_concat_r: x1[:, :, 2:3, :]
    cv::Range ranges2[] = { cv::Range::all(), cv::Range::all(), cv::Range(2, 3), cv::Range::all() };
    cv::Mat slice2_concat_r = x1(ranges2);

    // slice3_conv_l: x2[:, 0:2, :]
    cv::Range ranges3[] = { cv::Range::all(), cv::Range(0, 2), cv::Range::all() };
    cv::Mat slice3_conv_l = x2(ranges3);

    // slice4_conv_r: x2[:, 2:4, :]
    cv::Range ranges4[] = { cv::Range::all(), cv::Range(2, 4), cv::Range::all() };
    cv::Mat slice4_conv_r = x2(ranges4);

    // Constants
    std::vector<std::vector<float>> const_vec = get_constant(input_shape);
    int rows = const_vec.size();
    int cols = const_vec[0].size();
    std::vector<float> const_flat;
    for (const auto& row : const_vec)
        const_flat.insert(const_flat.end(), row.begin(), row.end());

    cv::Mat const_mat(rows, cols, CV_32F, const_flat.data());

    // cv::Mat const_pose = get_constant_pose();                // 1x2x8400
    // cv::Mat const_xy = get_constant({640, 640});             // 1x2x8400
    // cv::Mat const_mul = get_mul_constant({640, 640});        // 1x1x8400

    cv::Mat const_pose = get_constant_pose_expanded();               // shape: 1 x 2 x 8400
    cv::Mat const_xy   = get_constant_xy_mat(input_shape);      // shape: 1 x 2 x 8400
    cv::Mat const_mul  = get_constant_mul_mat_expanded(input_shape,{1, 17, 2, 8400});     // shape: 1 x 1 x 8400


    // Arithmetic
    cv::Mat x3_concat_l = slice1_concat_l * 2.0f;

    // Add constant_pose (shape: 1 x 2 x 8400)
    std::cout << "x2 shape: " << x3_concat_l.size << " x " << const_pose.size <<  std::endl;

    // Expand const_pose to 1 x 17 x 2 x 8400
    // Create expanded matrix: 1 x 17 x 2 x 8400
    int dims[] = {1, 17, 2, 8400};
    cv::Mat expanded_pose(4, dims, CV_32F);

    // Pointer to const_pose data
    float* src_data = reinterpret_cast<float*>(const_pose.data);

    for (int i = 0; i < 17; ++i) {
        // Get pointer to the destination slice
        float* dst_data = expanded_pose.ptr<float>(0, i);

        // Copy 2 x 8400 = 16800 floats
        std::memcpy(dst_data, src_data, sizeof(float) * 2 * 8400);
    }

    // cv::Mat x4_concat_l = x3_concat_l + expanded_pose;

    cv::Mat x4_concat_l;
    cv::add(x3_concat_l, const_pose, x4_concat_l); // safe element-wise addition


    cv::Mat x5_concat_l;
    cv::multiply(x4_concat_l, const_mul, x5_concat_l); 

    cv::Mat x6_concat_r=slice2_concat_r;
    //cv::Mat x5_concat_l = x4_concat_l *  const_mul;

    //x4_concat_l shape (1, 17, 2, 8400) (1, 8400)



    //cv::add(x3_concat_l, const_pose, x4_concat_l);
    
    // cv::Mat x5_concat_l; cv::multiply(x4_concat_l, const_mul, x5_concat_l);

    cv::Mat x7_concat_r;
    cv::exp(-slice2_concat_r, x7_concat_r);
    x7_concat_r = 1.0 / (1.0 + x7_concat_r);

    // cv::Mat a = cv::Mat::ones(3, std::vector<int>{1, 2, 3}.data(), CV_32F);
    // cv::Mat b = cv::Mat::ones(3, std::vector<int>{1, 2, 3}.data(), CV_32F);
    cv::Mat x_concat1 = concat_along_axis(x5_concat_l, x7_concat_r, 2);  // concatenate on axis 0


    //cv::Mat x_concat1 = concat_along_axis(x5_concat_l, x7_concat_r, 2); // axis=2


    // Reshape to 1x51x8400
    //x_concat1 = x_concat1.reshape(1, {1, 51, 8400});
    cv::Mat x_concat1_reshaped = reshape_1_51_8400(x_concat1);


    // // More arithmetic
    cv::Mat add1, sub1;
    cv::add(slice4_conv_r, const_xy, add1);
    cv::subtract(const_xy, slice3_conv_l, sub1);

    cv::Mat x8_conv_sub; cv::subtract(add1, sub1, x8_conv_sub);
    cv::Mat x9_conv_add; cv::add(sub1, add1, x9_conv_add);
    cv::Mat x10_conv_r = x9_conv_add * 0.5f;

    std::cout << "mat1.dims: " << x10_conv_r.size << std::endl;
    std::cout << "mat2.dims: " << x8_conv_sub.size << std::endl;



    cv::Mat x_concat2_conv = concat_along_axis(x10_conv_r, x8_conv_sub, 1);
    // cv::Mat x11_conv;
    // cv::multiply(x_concat2_conv, const_mul, x11_conv); 

    cv::Mat const_mul2  = get_constant_mul_mat_expanded(input_shape,{1, 4, 8400});  


    cv::Mat x11_conv; cv::multiply(x_concat2_conv, const_mul2, x11_conv);
    std::cout << "x11_conv.dims: " << x11_conv.size << std::endl;
    std::cout << "x_sigmoid.dims: " << x_sigmoid.size << std::endl;
    std::cout << "x_concat1_reshaped.dims: " << x_concat1_reshaped.size << std::endl;


    cv::Mat final_concat = concat_along_axis3(x11_conv, x_sigmoid, x_concat1_reshaped,1);

    // // Final concat: [x11_conv, sigmoid_output, x_concat1]
    // std::vector<cv::Mat> final_concat_parts = {x11_conv, sigmoid_output, x_concat1};
    // cv::Mat output;
    // cv::vconcat(final_concat_parts, output);  // Final shape: 1×56×8400

    std::cout << "out.dims: " << final_concat.size << std::endl;

    }


