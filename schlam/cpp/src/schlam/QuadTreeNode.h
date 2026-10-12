//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_QUADTREE_H
#define SCHLAM_QUADTREE_H

#include "KeyPoint.h"

#include <vector>
#include <memory>
#include <limits>


class QuadTreeNode {
public:
    QuadTreeNode(const double aLeft, const double aRight, const double aTop, const double aBottom,
                 const std::vector<KeyPoint> &aFeatures);

    QuadTreeNode(const double aLeft, const double aRight,
                 const double aTop, const double aBottom,
                 const std::vector<std::shared_ptr<KeyPoint>>& aFeatures,
                 QuadTreeNode *aParent);

    void divide();

    bool isLeaf() const;

    void retainBestFeaturesPerLeaf();

    // Full retrieval assigns indices in the root's feature array; window queries preserve them.
    std::vector<std::shared_ptr<KeyPoint>> getFeatures() const;
    std::vector<std::shared_ptr<KeyPoint>> getFeatures(double xMin, double xMax, double yMin, double yMax) const;

    void removeChild(const QuadTreeNode* aChild);

    std::vector<QuadTreeNode *> getChildren() const;

    double getMaxScore() const;

    QuadTreeNode *getParent() const;

private:
    // X borders
    double mLeft{0}, mRight{0};
    // Y borders
    double mTop{0}, mBottom{0};

    // helpers
    double mHalfX{0}, mHalfY{0};

    double mMaxScore = -std::numeric_limits<double>::max();

    QuadTreeNode *mParent{nullptr};

    std::unique_ptr<QuadTreeNode> mTL;
    std::unique_ptr<QuadTreeNode> mTR;
    std::unique_ptr<QuadTreeNode> mBL;
    std::unique_ptr<QuadTreeNode> mBR;

    std::vector<std::shared_ptr<KeyPoint>> mFeatures;

    bool overlaps(double xMin, double xMax, double yMin, double yMax) const;
    void collectFeatures(std::vector<std::shared_ptr<KeyPoint>>& features) const;
};


#endif //SCHLAM_QUADTREE_H
