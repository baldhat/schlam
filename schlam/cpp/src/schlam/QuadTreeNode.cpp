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
      mHalfX(std::floor((mRight - mLeft) / 2.0)),
      mHalfY(std::floor((mBottom - mTop) / 2.0)),
      mMaxScore(aFeatures.empty() ? -std::numeric_limits<double>::max() : aFeatures[0].getScore()) {
    mFeatures.reserve(aFeatures.size());
    for (const auto& feature : aFeatures) {
        mFeatures.push_back(std::make_shared<KeyPoint>(feature));
    }
}

QuadTreeNode::QuadTreeNode(const double aLeft, const double aRight,
                           const double aTop, const double aBottom,
                           const std::vector<std::shared_ptr<KeyPoint>>& aFeatures,
                           QuadTreeNode *aParent)
    : mLeft(aLeft), mRight(aRight), mTop(aTop), mBottom(aBottom),
      mFeatures(aFeatures),
      mHalfX(std::floor((mRight - mLeft) / 2.0)),
      mHalfY(std::floor((mBottom - mTop) / 2.0)),
      mMaxScore(aFeatures.empty() ? -std::numeric_limits<double>::max() : aFeatures[0]->getScore()),
      mParent(aParent) {
}


void QuadTreeNode::divide() {
    if (!isLeaf() || mFeatures.size() < 2 ||
        (mRight - mLeft <= 1 && mBottom - mTop <= 1)) {
        return;
    }
    std::vector<std::shared_ptr<KeyPoint>> featuresTL;
    std::vector<std::shared_ptr<KeyPoint>> featuresTR;
    std::vector<std::shared_ptr<KeyPoint>> featuresBL;
    std::vector<std::shared_ptr<KeyPoint>> featuresBR;

    for (const auto &feature: mFeatures) {
        const bool left = feature->getImgX() < (mLeft + mHalfX);
        const bool top = feature->getImgY() < (mTop + mHalfY);
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

std::vector<std::shared_ptr<KeyPoint>> QuadTreeNode::getFeatures() const {
    if (isLeaf()) {
        return mFeatures;
    } else {
        std::vector<std::shared_ptr<KeyPoint>> features;
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
    return mLeft <= xMax && mRight >= xMin && mTop <= yMax && mBottom >= yMin;
}

std::vector<std::shared_ptr<KeyPoint>> QuadTreeNode::getFeatures(double xMin, double xMax, double yMin, double yMax) const {
    if (isLeaf()) {
        std::vector<std::shared_ptr<KeyPoint>> features;
        for (const auto &feature : mFeatures) {
            if (feature->getImgX() >= xMin && feature->getImgX() <= xMax &&
                feature->getImgY() >= yMin && feature->getImgY() <= yMax) {
                features.push_back(feature);
            }
        }
        return features;
    } else {
        std::vector<std::shared_ptr<KeyPoint>> features;
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

bool QuadTreeNode::isLeaf() const { return !mTL && !mTR && !mBL && !mBR; }

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
    if (isLeaf()) {
        if (!mFeatures.empty()) {
            mFeatures.erase(mFeatures.begin() + 1, mFeatures.end());
        }
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
