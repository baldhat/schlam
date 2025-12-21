
#include "coordinate_frame.hpp"
#include "rigid_transform_3d.hpp"

namespace tft {
    CoordinateFrame::CoordinateFrame(const std::string& aName)
        : mName(aName) {

    }

    void CoordinateFrame::addTransform(const std::shared_ptr<RigidTransform3D> aTransform) {
        mTransforms.push_back(aTransform);
    }

    std::vector<std::shared_ptr<RigidTransform3D>> CoordinateFrame::getTransforms() {
        return mTransforms;
    }

    std::string CoordinateFrame::getName() {
        return mName;
    }
}