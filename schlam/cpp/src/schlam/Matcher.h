//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_MATCHER_H
#define SCHLAM_MATCHER_H

#include "KeyPoint.h"
#include <memory>
#include <vector>

std::vector<std::array<std::uint32_t, 2> > match(const std::vector<KeyPoint> &aKeypoints1,
                                                 const std::vector<KeyPoint> &aKeypoints2,
                                                 const double aMinDistance);


#endif //SCHLAM_MATCHER_H
