
#include "transformer.hpp"
#include "rigid_transform_3d.hpp"

#include <iostream>
#include <memory>
#include <queue>
#include <set>

namespace tft {
Transformer::Transformer() {
  mRootedFrames.emplace(std::make_pair(mRootName, RigidTransform3D::identity()));
}

void Transformer::registerTransform(
    const std::shared_ptr<RigidTransform3D> aTransform) {

  // forward edge
  Edge edge = {aTransform->mTarget, aTransform};
  if (mEdges.count(aTransform->mSource) == 0) {
    mEdges.emplace(
        std::pair<std::string, std::vector<Edge>>(aTransform->mSource, {edge}));
  } else {

    auto outgoingEdges = mEdges[aTransform->mSource];
    if (std::find_if(outgoingEdges.begin(), outgoingEdges.end(), [&](const Edge& edge_){return edge_.target == aTransform->mTarget;}) == outgoingEdges.end()) {
      mEdges[aTransform->mSource].push_back(edge);
    } else {
      std::cout << "Not adding edge, because it already exists: " << std::endl;
    }
  }

  // backward edge
  Edge inverse = {aTransform->mSource, aTransform->inverse()};
  if (mEdges.count(aTransform->mTarget) == 0) {
    mEdges.emplace(std::pair<std::string, std::vector<Edge>>(aTransform->mTarget,
                                                             {inverse}));
  } else {
    auto outgoingEdges = mEdges[aTransform->mTarget];
    if (std::find_if(outgoingEdges.begin(), outgoingEdges.end(), [&](const Edge& edge_){return edge_.target == aTransform->mSource;}) == outgoingEdges.end()) {
      mEdges[aTransform->mTarget].push_back(inverse);
    } else {
      std::cout << "Not adding edge, because it already exists: " << std::endl;
    }
  }
}

Transformable3D Transformer::transform(const Transformable3D &transformable,
                                       const std::string &target) {
  auto transform = findTransform(transformable.getCF(), target);
  if (transform) {
    return transform->apply(transformable);
  } else {
    std::cout << "WARN: No transform found from " << transformable.getCF()
              << "->" << target << std::endl;
  }
  return transformable;
}

std::shared_ptr<RigidTransform3D>
Transformer::findTransform(const std::string &source,
                           const std::string &target) {
  // Return identity if source and target are equals
  if (source == target) {
    return RigidTransform3D::identity();
  }

  if (mEdges.count(source) == 0 || mEdges.count(target) == 0) {
    return nullptr;
  }

  // Check if the frames are rooted and the transform can just be looked up
  if (mRootedFrames.count(source) > 0 && mRootedFrames.count(target) > 0) {
    return mRootedFrames[target] * mRootedFrames[source]->inverse();
  }

  // Do an actual BFS search in the tree
  std::queue<std::tuple<std::string, std::shared_ptr<RigidTransform3D>>> queue;
  std::set<std::string> visited;
  queue.push({source, RigidTransform3D::identity(source, source)});

  while (!queue.empty()) {
    auto [currentNode, currentTransform] = queue.front();
    queue.pop();
    visited.emplace(currentNode);

    if (currentNode == target) {
      // If we found the transform to the world frame, store it
      if (target == mRootName && mRootedFrames.count(source) == 0) {
        std::cout << "Add rooted frame: " << source << std::endl;
        mRootedFrames.emplace(std::make_pair(source, currentTransform->inverse()));
      }
      return currentTransform;
    }

    for (auto &edge : mEdges[currentNode]) {
      if (visited.find(edge.target) == visited.end()) {
        queue.push({edge.target, edge.transform * currentTransform});
      }
    }
  }
  return nullptr;
}

std::vector<std::shared_ptr<RigidTransform3D>> Transformer::getRootedTransforms() const {
  std::vector<std::shared_ptr<RigidTransform3D>> transforms;
  for (const auto& [name, transform] : mRootedFrames) {
    transforms.push_back(transform);
  }
  return transforms;
}
} // namespace tft