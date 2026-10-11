//
// Created by baldhat on 10/11/26.
//

#include "MapPoint.hpp"

MapPoint::MapPoint(Eigen::Vector3d position, std::array<std::uint8_t, 32> descriptor)
 : mPosition(position), mDescriptor(descriptor){

}
