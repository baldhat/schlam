//
// Created by baldhat on 2/7/26.
//

#ifndef SCHLAM_KEYPOINT_H
#define SCHLAM_KEYPOINT_H

#include <cstdint>
#include <memory>

#include "IFeatureDescriptor.h"


class KeyPoint {
public:
    KeyPoint(const std::uint32_t aImgX, const std::uint32_t aImgY, const std::uint8_t aLevel = 0, double aScore = 0,
             double aAngle = 0, std::shared_ptr<IFeatureDescriptor> apDescriptor = nullptr);

    void scaleBy(const double aXFactor, const double aYFactor);

    std::uint32_t getImgX() const;
    std::uint32_t getImgY() const;
    std::uint8_t getLevel() const;

    void setAngle(const double aAngle);
    void setScore(const double aScore);

private:
    std::uint32_t mImgX{0};
    std::uint32_t mImgY{0};
    double mScore{0};
    std::uint8_t mLevel{0};
    double mAngle{0};
    std::shared_ptr<IFeatureDescriptor> mpDescriptor;

};


#endif //SCHLAM_KEYPOINT_H
