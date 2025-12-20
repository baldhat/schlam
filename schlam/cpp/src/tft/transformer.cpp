
#include "transformer.hpp"
#include <stdexcept>

namespace tft {
    Transformer::Transformer() {

    }

    void Transformer::registerTransform(const RigidTransform3D& transform) {
        transforms.push_back(transform);
    }

    Transformable3D Transformer::transform(const Transformable3D& Transformable3D, const std::string& target) {
        throw std::runtime_error("not implemented");
    }

    RigidTransform3D Transformer::findTransform(const std::string& source, const std::string& target) {
        throw std::runtime_error("not implemented");
    }
}