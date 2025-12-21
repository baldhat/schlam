#pragma once

#include "rigid_transform_3d.hpp"
#include "coordinate_frame.hpp"

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace tft {

class Transformer {
public:
  Transformer();
  ~Transformer() = default;

  // TODO: Add some buffering to avoid calculating long chains

  void registerTransform(const std::shared_ptr<RigidTransform3D> transform);
  
  Transformable3D transform(const Transformable3D &Transformable3D,
                            const std::string &target);

  std::shared_ptr<RigidTransform3D> findTransform(const std::string &source, const std::string &target);

private:
  std::vector<std::shared_ptr<CoordinateFrame>> mTrees;

  void createNewTree(const std::shared_ptr<RigidTransform3D> transform);

};
} // namespace tft