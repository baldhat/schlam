//
// Created by baldhat on 2/8/26.
//

#include "QuadTreeNode.h"

#include <cmath>

QuadTreeNode::QuadTreeNode(const double aLeft, const double aRight, const double aTop, const double aBottom,
                           const std::vector<KeyPoint> aFeatures)
    : mLeft(aLeft), mRight(aRight), mTop(aTop), mBottom(aBottom), mFeatures(aFeatures) {
    mIsLeaf = mFeatures.size() <= 1;
    mIsEmpty = mFeatures.size() == 0;

    mHalfX = std::floor((mRight - mLeft) / 2.0);
    mHalfY = std::floor((mBottom - mTop) / 2.0);
}

std::vector<std::shared_ptr<QuadTreeNode> > QuadTreeNode::divide() const {
    std::vector<KeyPoint> featuresTL;
    std::vector<KeyPoint> featuresTR;
    std::vector<KeyPoint> featuresBL;
    std::vector<KeyPoint> featuresBR;

    for (const auto &feature: mFeatures) {
        bool left = feature.getImgX() < (mLeft + mHalfX);
        bool top = feature.getImgY() < (mTop + mHalfY);
        if (left) {
            if (top) {
                featuresTL.push_back(feature);
            } else {
                featuresBL.push_back(feature);
            }
        } else {
            if (top) {
                featuresTR.push_back(feature);
            } else {
                featuresBR.push_back(feature);
            }
        }
    }

    return {
        std::make_shared<QuadTreeNode>(mLeft, mLeft + mHalfX, mTop, mTop + mHalfY, featuresTL),
        std::make_shared<QuadTreeNode>(mLeft + mHalfX, mRight, mTop, mTop + mHalfY, featuresTR),
        std::make_shared<QuadTreeNode>(mLeft, mLeft + mHalfX, mTop + mHalfY, mBottom, featuresBL),
        std::make_shared<QuadTreeNode>(mLeft + mHalfX, mRight, mTop + mHalfY, mBottom, featuresBR)
    };
}

bool QuadTreeNode::isLeaf() const {
    return mIsLeaf;
}

bool QuadTreeNode::isEmpty() const {
    return mIsEmpty;
}

KeyPoint QuadTreeNode::findMaxFeature() const {
    KeyPoint max = mFeatures[0];
    for (auto& feature : mFeatures) {
        if (feature.getScore() > max.getScore()) {
            max = feature;
        }
    }
    return max;
}


KeyPoint QuadTreeNode::getMaxFeature() const {
    if (mIsLeaf) {
        return mFeatures[0];
    } else {
        return findMaxFeature();
    }
}
