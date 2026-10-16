//
// Created by baldhat on 2/7/26.
//

#ifndef SCHLAM_KEYPOINT_H
#define SCHLAM_KEYPOINT_H

#include <cstdint>
#include <cstddef>
#include <array>
#include <optional>

#include "MapTypes.h"

using Descriptor = std::array<uint8_t, 32>;

class KeyPoint {
public:
    KeyPoint(const double aImgX, const double aImgY, const std::uint8_t aLevel = 0, double aScore = 0,
             double aAngle = 0, const Descriptor& aDescriptor = Descriptor());

    void scaleBy(const double aXFactor, const double aYFactor);

    double getImgX() const;
    double getImgY() const;
    std::uint8_t getLevel() const;
    double getScore() const;
    Descriptor getDescriptor() const;
    std::optional<std::size_t> getIndex() const;

    void setAngle(const double aAngle);
    double getAngle() const;
    void setScore(const double aScore);
    void setLevel(const double aLevel);
    void setDescriptor(const Descriptor& aDescriptor);

    bool getMatched() const;
    void setObservation(MapPointId pointId);
    void clearObservation();
    void setIndex(std::size_t index);
    void setKeyFrameId(KeyFrameId frameId);
    std::optional<KeyFrameId> getKeyFrameId() const;

private:
    double mImgX{0};
    double mImgY{0};
    double mScore{0};
    std::uint8_t mLevel{0};
    double mAngle{0};
    Descriptor mDescriptor;
    std::optional<MapPointId> mMapPointId;
    std::optional<std::size_t> mIndex;
    std::optional<KeyFrameId> mKeyFrameId;
};


#endif //SCHLAM_KEYPOINT_H
