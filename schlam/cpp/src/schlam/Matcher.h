//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_MATCHER_H
#define SCHLAM_MATCHER_H

#include "KeyPoint.h"
#include "QuadTreeNode.h"

#include <vector>


std::array<std::vector<KeyPoint *>, 2> matchGlobally(QuadTreeNode *aTree1,
                                                     QuadTreeNode *aTree2,
                                                     const double aMaxDistance);

std::array<std::vector<KeyPoint *>, 2> matchWindow(const QuadTreeNode *aTree1,
                                                   const QuadTreeNode *aTree2,
                                                   const double aMaxDistance, const double aWindowSize);

std::array<std::vector<KeyPoint *>, 2> getMatched(const std::vector<KeyPoint *> &aKeypoints1,
                                                  const std::vector<KeyPoint *> &aKeypoints2,
                                                  std::vector<std::array<std::uint32_t, 2> > &aMatches);

#endif //SCHLAM_MATCHER_H
