//
// Created by baldhat on 2/7/26.
//

#ifndef SCHLAM_KEYPOINT_H
#define SCHLAM_KEYPOINT_H

#include <cstdint>
#include <array>


class KeyPoint {
public:
    KeyPoint(const std::uint32_t aImgX, const std::uint32_t aImgY, const std::uint8_t aLevel = 0, double aScore = 0,
             double aAngle = 0, const std::array<uint8_t, 32>& aDescriptor = std::array<uint8_t, 32>());

    void scaleBy(const double aXFactor, const double aYFactor);

    std::uint32_t getImgX() const;
    std::uint32_t getImgY() const;
    std::uint8_t getLevel() const;
    double getScore() const;
    std::array<uint8_t, 32> getDescriptor() const;

    void setAngle(const double aAngle);
    double getAngle() const;
    void setScore(const double aScore);
    void setLevel(const double aLevel);
    void setDescriptor(const  std::array<uint8_t, 32>& aDescriptor);

private:
    std::uint32_t mImgX{0};
    std::uint32_t mImgY{0};
    double mScore{0};
    std::uint8_t mLevel{0};
    double mAngle{0};
    std::array<uint8_t, 32> mDescriptor;
};


#endif //SCHLAM_KEYPOINT_H
