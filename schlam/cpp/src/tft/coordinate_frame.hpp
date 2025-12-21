#pragma once

#include "rigid_transform_3d.hpp"

#include <string>
#include <vector>
#include <memory>

namespace tft {

    class CoordinateFrame {
    public:
        CoordinateFrame(const std::string& aName);
        ~CoordinateFrame() = default;

        void addTransform(const std::shared_ptr<RigidTransform3D> aTransform);
        std::vector<std::shared_ptr<RigidTransform3D>> getTransforms();

        std::string getName();

    private:
        const std::string mName;
        std::vector<std::shared_ptr<RigidTransform3D>> mTransforms;
    };

}