//
// Created by baldhat on 2/8/26.
//

#ifndef SCHLAM_QUADTREE_H
#define SCHLAM_QUADTREE_H

class KeyPoint;

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>


class QuadTreeNode {
public:
    struct Bounds {
        double left, right, top, bottom;
    };

    class Children {
    public:
        auto begin() const { return mNodes.begin(); }
        auto end() const { return mNodes.begin() + mCount; }

    private:
        friend class QuadTreeNode;
        std::array<QuadTreeNode*, 4> mNodes{};
        std::size_t mCount = 0;
    };

    QuadTreeNode(const double aLeft, const double aRight, const double aTop, const double aBottom,
                 const std::vector<KeyPoint> &aFeatures);

    QuadTreeNode(const double aLeft, const double aRight,
                 const double aTop, const double aBottom,
                 std::vector<std::shared_ptr<KeyPoint>> aFeatures,
                 QuadTreeNode *aParent);

    void divide();

    bool isLeaf() const;

    void retainBestFeaturesPerLeaf();

    // Full retrieval assigns indices in the root's feature array; window queries preserve them.
    std::vector<std::shared_ptr<KeyPoint>> getFeatures() const;
    std::vector<std::shared_ptr<KeyPoint>> getFeatures(double xMin, double xMax, double yMin, double yMax) const;
    void getFeatures(double xMin, double xMax, double yMin, double yMax, std::vector<std::shared_ptr<KeyPoint>>& result) const;

    void removeChild(const QuadTreeNode* aChild);

    Children getChildren() const;

    Bounds getBounds() const { return mBounds; }

    double getMaxScore() const;

    QuadTreeNode *getParent() const;

private:
    friend class ORBFeatureDetector;

    struct Split {
        explicit Split(Bounds bounds);

        std::size_t quadrant(double x, double y) const {
            return (y < yMid ? 0 : 2) + (x < xMid ? 0 : 1);
        }

        double xMid, yMid;
        std::array<Bounds, 4> quadrants;
    };

    static bool canDivide(Bounds bounds);

    Bounds mBounds;
    double mMaxScore = -std::numeric_limits<double>::max();

    QuadTreeNode *mParent{nullptr};

    std::array<std::unique_ptr<QuadTreeNode>, 4> mChildren;
    std::vector<std::shared_ptr<KeyPoint>> mFeatures;

    bool overlaps(double xMin, double xMax, double yMin, double yMax) const;
    void collectFeatures(std::vector<std::shared_ptr<KeyPoint>>& features) const;
};


#endif //SCHLAM_QUADTREE_H
