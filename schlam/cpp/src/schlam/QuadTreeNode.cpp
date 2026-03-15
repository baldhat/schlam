//
// Created by baldhat on 2/8/26.
//

#include "QuadTreeNode.h"

#include <cmath>
#include <iostream>

QuadTreeNode::QuadTreeNode(const double aLeft, const double aRight,
                           const double aTop, const double aBottom,
                           const std::vector<KeyPoint> &aFeatures)
    : mLeft(aLeft), mRight(aRight), mTop(aTop), mBottom(aBottom),
      mFeatures(aFeatures),
      mHalfX(std::floor((mRight - mLeft) / 2.0)),
      mHalfY(std::floor((mBottom - mTop) / 2.0)),
      mMaxScore(aFeatures[0].getScore()) {
}

QuadTreeNode::QuadTreeNode(const double aLeft, const double aRight,
                           const double aTop, const double aBottom,
                           const std::vector<KeyPoint> &aFeatures,
                           QuadTreeNode *aParent)
    : mLeft(aLeft), mRight(aRight), mTop(aTop), mBottom(aBottom),
      mFeatures(aFeatures),
      mHalfX(std::floor((mRight - mLeft) / 2.0)),
      mHalfY(std::floor((mBottom - mTop) / 2.0)),
      mMaxScore(aFeatures[0].getScore()),
      mParent(aParent) {
}


void QuadTreeNode::divide() {
    std::vector<KeyPoint> featuresTL;
    std::vector<KeyPoint> featuresTR;
    std::vector<KeyPoint> featuresBL;
    std::vector<KeyPoint> featuresBR;

    for (const auto &feature: mFeatures) {
        const bool left = feature.getImgX() < (mLeft + mHalfX);
        const bool top = feature.getImgY() < (mTop + mHalfY);
        if (left) {
            if (top) {
                featuresTL.emplace_back(feature);
            } else {
                featuresBL.emplace_back(feature);
            }
        } else {
            if (top) {
                featuresTR.emplace_back(feature);
            } else {
                featuresBR.emplace_back(feature);
            }
        }
    }

    if (featuresTL.size() > 0) {
        mTL = std::make_unique<QuadTreeNode>(mLeft, mLeft + mHalfX, mTop,
                                             mTop + mHalfY, featuresTL, this);
    }
    if (featuresTR.size() > 0) {
        mTR = std::make_unique<QuadTreeNode>(mLeft + mHalfX, mRight, mTop,
                                             mTop + mHalfY, featuresTR, this);
    }
    if (featuresBL.size() > 0) {
        mBL = std::make_unique<QuadTreeNode>(mLeft, mLeft + mHalfX, mTop + mHalfY,
                                             mBottom, featuresBL, this);
    }
    if (featuresBR.size() > 0) {
        mBR = std::make_unique<QuadTreeNode>(mLeft + mHalfX, mRight, mTop + mHalfY,
                                             mBottom, featuresBR, this);
    }

    mFeatures.clear();
}

std::vector<KeyPoint*> QuadTreeNode::getFeatures() const {
    if (mFeatures.size() >= 1) {
        return {const_cast<KeyPoint*>(&mFeatures[0])};
    } else {
        std::vector<KeyPoint*> features;
        for (const auto &child: getChildren()) {
            auto childFeatures = child->getFeatures();
            features.insert(features.end(), childFeatures.begin(), childFeatures.end());
        }
        if (features.size() == 0) {
            return {};
        }
        return features;
    }
}

bool QuadTreeNode::overlaps(double xMin, double xMax, double yMin, double yMax) const {
    // horizontal
    if (!((mLeft > xMin && mLeft < xMax) || (mRight > xMin && mRight < xMax))) {
        return false;
    }
    // vertical
    if (!((mTop > yMin && mTop < yMax) || (mBottom > yMin && mBottom < yMax))) {
        return false;
    }
    return true;
}

std::vector<KeyPoint*> QuadTreeNode::getFeatures(double xMin, double xMax, double yMin, double yMax) const {
    if (mFeatures.size() == 1) {
        auto &feature = mFeatures[0];
        if (feature.getImgX() >= xMin && feature.getImgX() <= xMax && feature.getImgY() >= yMin && feature.getImgY() <=
            yMax) {
            return {const_cast<KeyPoint*>(&feature)};
        } else {
            return {};
        }
    } else {
        std::vector<KeyPoint*> features;
        for (const auto &child: getChildren()) {
            if (child->overlaps(xMin, xMax, yMin, yMax)) {
                auto childFeatures = child->getFeatures(xMin, xMax, yMin, yMax);
                features.insert(features.end(), childFeatures.begin(), childFeatures.end());
            }
        }
        return features;
    }
}

void QuadTreeNode::removeChild(const QuadTreeNode *aChild) {
    if (aChild == mTL.get()) {
        mTL.reset();
    } else if (aChild == mTR.get()) {
        mTR.reset();
    } else if (aChild == mBL.get()) {
        mBL.reset();
    } else if (aChild == mBR.get()) {
        mBR.reset();
    } else {
        std::cout << "WARN: Didn't find child to be deleted!" << std::endl;
    }
}

bool QuadTreeNode::isLeaf() const { return mFeatures.size() == 1; }

std::vector<QuadTreeNode *> QuadTreeNode::getChildren() const {
    std::vector<QuadTreeNode *> children;
    if (mTL) {
        children.push_back(mTL.get());
    }
    if (mTR) {
        children.push_back(mTR.get());
    }
    if (mBL) {
        children.push_back(mBL.get());
    }
    if (mBR) {
        children.push_back(mBR.get());
    }
    return children;
}

void QuadTreeNode::retainBestFeaturesPerLeaf() {
    if (getChildren().size() == 0) {
        mFeatures = {mFeatures[0]};
    } else {
        for (auto const &child: getChildren()) {
            child->retainBestFeaturesPerLeaf();
        }
    }
}

double QuadTreeNode::getMaxScore() const {
    return mMaxScore;
}


QuadTreeNode *QuadTreeNode::getParent() const {
    return mParent;
}
