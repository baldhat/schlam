//
// Created by jonathan_michel on 10/24/26.
//

#ifndef SCHLAM_FILE_HELPER_H
#define SCHLAM_FILE_HELPER_H
#include <cstddef>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>

namespace helpers {
    template <int R, int C>
    std::vector<Eigen::Matrix<double, R, C> > readLinesAsMatrices(std::ifstream &file, bool readKeyFirst = true) {
        std::vector<Eigen::Matrix<double, R, C>> matrices;

        std::string line;
        std::size_t index{0};
        while (std::getline(file, line)) {
            std::istringstream stream(line);
            std::string key;
            if (readKeyFirst)
                stream >> key;

            Eigen::Matrix<double, R, C> mat;
            for (int r = 0; r < R; ++r) {
                for (int c = 0; c < C; ++c) {
                    if (!(stream >> mat(r, c))) {
                        throw std::runtime_error(
                            "Invalid matrix data for " + key
                        );
                    }
                }
            }
            matrices.push_back(mat);
            index++;
        }
        return matrices;
    }

    std::chrono::system_clock::time_point stringNanoToTimePoint(const std::string &nano_str);
    std::chrono::system_clock::time_point longNanoToTimePoint(const std::uint64_t &aNanos);
    std::chrono::system_clock::time_point stringSecondsToTimePoint(const std::string &second_str);
}

#endif //SCHLAM_FILE_HELPER_H
