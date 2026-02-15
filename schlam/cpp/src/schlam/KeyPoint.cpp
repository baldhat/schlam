//
// Created by baldhat on 2/7/26.
//

#include "KeyPoint.h"


KeyPoint::KeyPoint(const std::uint32_t aImgX, const std::uint32_t aImgY, const std::uint8_t aLevel, double aScore,
                   double aAngle,const std::array<uint8_t, 32>& apDescriptor)
    : mImgX(aImgX)
      , mImgY(aImgY)
      , mScore(aScore)
      , mLevel(aLevel), mAngle(aAngle), mDescriptor(apDescriptor) {

}

void KeyPoint::scaleBy(const double aXFactor, const double aYFactor) {
    mImgX *= aXFactor;
    mImgY *= aYFactor;
}

std::uint32_t KeyPoint::getImgX() const {
    return mImgX;
}
std::uint32_t KeyPoint::getImgY() const {
    return mImgY;
}
std::uint8_t KeyPoint::getLevel() const {
    return mLevel;
}
double KeyPoint::getAngle() const {
    return mAngle;
}
double KeyPoint::getScore() const {
    return mScore;
}
std::array<uint8_t, 32> KeyPoint::getDescriptor() const {
    return mDescriptor;
}

void KeyPoint::setScore(const double aScore) {
    mScore = aScore;
}
void KeyPoint::setLevel(const double aLevel) {
    mLevel = aLevel;
}
void KeyPoint::setDescriptor(const std::array<uint8_t, 32>& aDescriptor) {
    mDescriptor = aDescriptor;
}
void KeyPoint::setAngle(const double aAngle) {
    mAngle = aAngle;
}


