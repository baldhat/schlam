
#include "transformer.hpp"
#include <stdexcept>
#include <thread>

namespace tft {
    Transformer::Transformer() {

    }

    void Transformer::registerTransform(const RigidTransform3D& transform) {
        mTransforms.push_back(transform);
    }

    Transformable3D Transformer::transform(const Transformable3D& Transformable3D, const std::string& target) {
        throw std::runtime_error("not implemented");
    }

    RigidTransform3D Transformer::findTransform(const std::string& source, const std::string& target) {
        for (auto& transform : mTransforms) {
            if (transform.source == source && transform.target == target) {
                return transform;
            } else if (transform.target == source && transform.source == target) {
                return transform.inverse();
            }
        }
        throw std::runtime_error("No transform from " + source + " to " + target);
    }
}