#pragma once

#include "rigid_transform_3d.hpp"

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace tft {

struct Edge {
    std::string target;
    std::shared_ptr<RigidTransform3D> transform;
};

class Transformer {
public:
  Transformer();
  ~Transformer() = default;

  void registerTransform(const std::shared_ptr<RigidTransform3D> transform);
  
  Transformable3D transform(const Transformable3D &Transformable3D,
                            const std::string &target);

  std::shared_ptr<RigidTransform3D> findTransform(const std::string &source, const std::string &target);

  std::vector<std::shared_ptr<RigidTransform3D>> getRootedTransforms() const;

private:
  std::map<std::string, std::vector<Edge>> mEdges;

  // Caches the transform world->frame
  std::map<std::string, std::shared_ptr<RigidTransform3D>> mRootedFrames;

  std::string mRootName{"world"};

  void createNewTree(const std::shared_ptr<RigidTransform3D> transform);

};
} // namespace tft