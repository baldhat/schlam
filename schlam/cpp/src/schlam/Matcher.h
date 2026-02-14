//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_MATCHER_H
#define SCHLAM_MATCHER_H

#include "KeyPoint.h"
#include <vector>

std::vector<std::array<std::uint32_t, 2> > match(const std::vector<KeyPoint> &aKeypoints1,
                                                 const std::vector<KeyPoint> &aKeypoints2,
                                                 const double aMinDistance);

std::array<std::vector<KeyPoint>, 2> getMatched(const std::vector<KeyPoint> &aKeypoints1,
                                                 const std::vector<KeyPoint> &aKeypoints2,
                                                 std::vector<std::array<std::uint32_t, 2>>& aMatches);

#endif //SCHLAM_MATCHER_H
