
#include "transformer.hpp"
#include "coordinate_frame.hpp"
#include "rigid_transform_3d.hpp"
#include <stdexcept>
#include <thread>
#include <queue>
#include <set>

namespace tft {
    Transformer::Transformer() {

    }

    void Transformer::registerTransform(const std::shared_ptr<RigidTransform3D> aTransform) {
        if (mTrees.empty()) {
            createNewTree(aTransform);
        } else {
            for (auto& root : mTrees) {
                std::queue<std::shared_ptr<CoordinateFrame>> searchList;
                std::set<std::string> searchedList;
                
                searchList.push(root);
                
                while (!searchList.empty()) {
                    auto currentCF = searchList.front();

                    if (currentCF->getName() == aTransform->target) {
                        currentCF->addTransform(aTransform);
                        return;
                    } else if (currentCF->getName() == aTransform->source) {
                        currentCF->addTransform(aTransform->inverse());
                        return;
                    }

                    for (auto& transform : currentCF->getTransforms()) {
                        if (searchedList.count(transform->target) == 0) {
                            searchList.push(transform)
                        }
                    }

                    searchList.pop();
                    searchedList.emplace(currentCF->getName());
                }
            }
        }
        throw std::runtime_error("Not implemented");
    }

    Transformable3D Transformer::transform(const Transformable3D& Transformable3D, const std::string& target) {
        throw std::runtime_error("not implemented");
    }

    std::shared_ptr<RigidTransform3D> Transformer::findTransform(const std::string& source, const std::string& target) {
        // Return identity if source and target are equals
        if (source == target) {
            return std::make_shared<RigidTransform3D>(source, target, Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0));
        }

        // for (auto& transform : mTransforms) {
        //     if (transform->source == source && transform->target == target) {
        //         return transform;
        //     } else if (transform->target == source && transform->source == target) {
        //         return transform->inverse();
        //     }
        // }
        throw std::runtime_error("No transform from " + source + " to " + target);
    }

    void Transformer::createNewTree(const std::shared_ptr<RigidTransform3D> transform) {
        auto cfSource = std::make_shared<CoordinateFrame>(transform->source);
        auto cfTarget = std::make_shared<CoordinateFrame>(transform->target);
        cfSource->addTransform(transform);
        cfTarget->addTransform(transform);
        mTrees.push_back(cfSource);
    }
}