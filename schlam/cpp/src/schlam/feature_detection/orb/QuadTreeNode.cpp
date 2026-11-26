//
// Created by baldhat on 2/8/26.
//

#include "QuadTreeNode.h"
#include "../../KeyPoint.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <tracy/Tracy.hpp>

QuadTreeNode::QuadTreeNode(const double aLeft, const double aRight,
                           const double aTop, const double aBottom,
                           const std::vector<KeyPoint> &aFeatures)
    : mBounds{aLeft, aRight, aTop, aBottom},
      mMaxScore(aFeatures.empty() ? -std::numeric_limits<double>::max() : aFeatures[0].getScore()) {
    mFeatures.reserve(aFeatures.size());
    for (const auto& feature : aFeatures) {
        mFeatures.push_back(std::make_shared<KeyPoint>(feature));
    }
}

QuadTreeNode::QuadTreeNode(const double aLeft, const double aRight,
                           const double aTop, const double aBottom,
                           std::vector<std::shared_ptr<KeyPoint>> aFeatures,
                           QuadTreeNode *aParent)
    : mBounds{aLeft, aRight, aTop, aBottom},
      mMaxScore(aFeatures.empty() ? -std::numeric_limits<double>::max() : aFeatures[0]->getScore()),
      mParent(aParent),
      mFeatures(std::move(aFeatures)) {
}

QuadTreeNode::Split::Split(Bounds bounds)
    : xMid(bounds.left + std::floor((bounds.right - bounds.left) / 2.0)),
      yMid(bounds.top + std::floor((bounds.bottom - bounds.top) / 2.0)),
      quadrants{{
          {bounds.left, xMid, bounds.top, yMid},
          {xMid, bounds.right, bounds.top, yMid},
          {bounds.left, xMid, yMid, bounds.bottom},
          {xMid, bounds.right, yMid, bounds.bottom},
      }} {
}

bool QuadTreeNode::canDivide(Bounds bounds) {
    return bounds.right - bounds.left > 1 || bounds.bottom - bounds.top > 1;
}

void QuadTreeNode::divide() {
    ZoneScoped;
    if (!isLeaf() || mFeatures.size() < 2 || !canDivide(mBounds)) {
        return;
    }
    const Split split(mBounds);
    std::array<std::vector<std::shared_ptr<KeyPoint>>, 4> buckets;
    for (const auto& feature : mFeatures) {
        buckets[split.quadrant(feature->getImgX(), feature->getImgY())].push_back(feature);
    }

    std::array<std::unique_ptr<QuadTreeNode>, 4> children;
    for (std::size_t quadrant = 0; quadrant < buckets.size(); ++quadrant) {
        if (buckets[quadrant].empty()) continue;
        const auto& bounds = split.quadrants[quadrant];
        children[quadrant] = std::make_unique<QuadTreeNode>(bounds.left, bounds.right,
            bounds.top, bounds.bottom, std::move(buckets[quadrant]), this);
    }

    mChildren = std::move(children);
    std::vector<std::shared_ptr<KeyPoint>>{}.swap(mFeatures);
}

std::vector<std::shared_ptr<KeyPoint>> QuadTreeNode::getFeatures() const {
    const auto* root = this;
    while (root->mParent) root = root->mParent;
    if (root != this) root->getFeatures();

    std::vector<std::shared_ptr<KeyPoint>> features;
    collectFeatures(features);
    if (root == this) {
        for (std::size_t index = 0; index < features.size(); ++index) {
            features[index]->setIndex(index);
        }
    }
    return features;
}

void QuadTreeNode::collectFeatures(std::vector<std::shared_ptr<KeyPoint>>& features) const {
    if (isLeaf()) {
        features.insert(features.end(), mFeatures.begin(), mFeatures.end());
    } else {
        for (const auto &child: getChildren()) {
            child->collectFeatures(features);
        }
    }
}

bool QuadTreeNode::overlaps(double xMin, double xMax, double yMin, double yMax) const {
    return mBounds.left <= xMax && mBounds.right >= xMin &&
           mBounds.top <= yMax && mBounds.bottom >= yMin;
}


void QuadTreeNode::getFeatures(double xMin, double xMax, double yMin, double yMax, std::vector<std::shared_ptr<KeyPoint>>& result) const {
    if (isLeaf()) {
        for (const auto &feature : mFeatures) {
            if (feature->getImgX() >= xMin && feature->getImgX() <= xMax &&
                feature->getImgY() >= yMin && feature->getImgY() <= yMax) {
                result.push_back(feature);
            }
        }
    } else {
        for (const auto& child : mChildren) {
            if (child && child->overlaps(xMin, xMax, yMin, yMax)) {
                child->getFeatures(xMin, xMax, yMin, yMax, result);
            }
        }
    }
}

std::vector<std::shared_ptr<KeyPoint>> QuadTreeNode::getFeatures(double xMin, double xMax, double yMin, double yMax) const {
    ZoneScoped;
    std::vector<std::shared_ptr<KeyPoint>> result;
    getFeatures(xMin, xMax, yMin, yMax, result);
    return result;
}

void QuadTreeNode::removeChild(const QuadTreeNode *aChild) {
    for (auto& child : mChildren) {
        if (child.get() == aChild) {
            child.reset();
            return;
        }
    }
    std::cout << "WARN: Didn't find child to be deleted!" << std::endl;
}

bool QuadTreeNode::isLeaf() const {
    return std::ranges::none_of(mChildren, [](const auto& child) { return child != nullptr; });
}

QuadTreeNode::Children QuadTreeNode::getChildren() const {
    Children children;
    for (const auto& child : mChildren) {
        if (child) children.mNodes[children.mCount++] = child.get();
    }
    return children;
}

void QuadTreeNode::retainBestFeaturesPerLeaf() {
    ZoneScoped;
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
