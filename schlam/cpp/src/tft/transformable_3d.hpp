#pragma once

#include <eigen3/Eigen/Core>

#include <string>

namespace tft {



class Transformable3D {
public:
  Transformable3D(const Eigen::Vector3d &vector,
                  const std::string &coordinateFrame);
  ~Transformable3D() = default;

  std::string getCF() const;
  Eigen::Vector3d vector;

  double x() const;
  double y() const;
  double z() const;

protected:
  std::string coordinateFrame;
};

std::ostream& operator<<(std::ostream& os, const Transformable3D& obj);

} // namespace tft