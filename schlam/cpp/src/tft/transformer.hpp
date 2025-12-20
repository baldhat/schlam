#pragma once

#include "rigid_transform_3d.hpp"

#include <string>
#include <vector>
#include <map>

namespace tft {

class Transformer {
public:
  Transformer();
  ~Transformer() = default;

  // TODO: Add some buffering to avoid calculating long chains

  void registerTransform(const RigidTransform3D &transform);
  Transformable3D transform(const Transformable3D &Transformable3D,
                            const std::string &target);

private:
  std::vector<RigidTransform3D> transforms;

  RigidTransform3D findTransform(const std::string &source, const std::string &target);
};
} // namespace tft