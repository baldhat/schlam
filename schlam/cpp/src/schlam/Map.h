//
// Created by baldhat on 10/11/26.
//

#ifndef SCHLAM_MAP_H
#define SCHLAM_MAP_H
#include <string>
#include <vector>

#include "MapPoint.hpp"


class Map {
public:
    std::vector<MapPoint> mPoints;
    std::string mFrame;
};


#endif //SCHLAM_MAP_H
