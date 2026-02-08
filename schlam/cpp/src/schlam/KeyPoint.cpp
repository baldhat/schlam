//
// Created by baldhat on 2/7/26.
//

#include "KeyPoint.h"


KeyPoint::KeyPoint(const std::uint32_t aImgX, const std::uint32_t aImgY, const std::uint8_t aLevel, double aScore,
                   double aAngle, std::shared_ptr<IFeatureDescriptor> apDescriptor)
    : mImgX(aImgX)
      , mImgY(aImgY)
      , mScore(aScore)
      , mLevel(aLevel), mAngle(aAngle), mpDescriptor(apDescriptor) {

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
void KeyPoint::setAngle(const double aAngle) {
    mAngle = aAngle;
}
void KeyPoint::setScore(const double aScore) {
    mScore = aScore;
}

double KeyPoint::getScore() const {
    return mScore;
}
