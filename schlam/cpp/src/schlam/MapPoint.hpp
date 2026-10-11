//
// Created by baldhat on 10/11/26.
//

#ifndef SCHLAM_MAPPOINT_H
#define SCHLAM_MAPPOINT_H

#include <array>
#include <memory>

#include <eigen3/Eigen/Core>


class Frame;

struct Observation {
    std::shared_ptr<Frame> frame;

};

class MapPoint {
public:
    MapPoint(Eigen::Vector3d position, std::array<std::uint8_t, 32> descriptor);

    Eigen::Vector3d mPosition;
    std::array<std::uint8_t, 32> mDescriptor;
    //std::vector<Observation> mObservations;

};


#endif //SCHLAM_MAPPOINT_H
