#include "schlam/feature_detection/xfeat/XFeatModel.h"

#include <iostream>
#include <utility>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char** argv) {
    if (argc > 3) {
        std::cerr << "Usage: xfeat_model [model.onnx [grayscale-image]]\n";
        return 2;
    }
    try {
        const auto path = argc >= 2 ? argv[1] : SCHLAM_DEFAULT_XFEAT_MODEL_PATH;
        XFeatModel model(path);
        const auto image = argc == 3 ? cv::imread(argv[2], cv::IMREAD_GRAYSCALE) :
            cv::Mat(96, 128, CV_8UC1, cv::Scalar(127));
        const auto output = model.run(image);
        std::cout << "Loaded XFeat backbone: " << path << '\n';
        for (const auto& [name, tensor] : {
                 std::pair("features", output.features),
                 std::pair("keypoint_logits", output.keypointLogits),
                 std::pair("reliability", output.reliability)}) {
            std::cout << name << ": [" << tensor.size[0] << ", " << tensor.size[1]
                      << ", " << tensor.size[2] << ", " << tensor.size[3] << "]\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "XFeat model error: " << error.what() << '\n';
        return 1;
    }
}
