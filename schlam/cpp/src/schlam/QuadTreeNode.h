//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_QUADTREE_H
#define SCHLAM_QUADTREE_H

#include "KeyPoint.h"

#include <vector>


class QuadTreeNode {
    public:

    QuadTreeNode(const double aLeft, const double aRight, const double aTop, const double aBottom, const std::vector<KeyPoint> aFeatures);

    std::vector<std::shared_ptr<QuadTreeNode>> divide() const;

    bool isLeaf() const;
    bool isEmpty() const;
    KeyPoint getMaxFeature() const;

private:
    // X borders
    double mLeft{0}, mRight{0};
    // Y borders
    double mTop{0}, mBottom{0};

    // helpers
    double mHalfX{0}, mHalfY{0};

    // All features within this node and its subnodes
    const std::vector<KeyPoint> mFeatures;

    bool mIsLeaf{false}, mIsEmpty{false};


    KeyPoint findMaxFeature() const;
};


#endif //SCHLAM_QUADTREE_H