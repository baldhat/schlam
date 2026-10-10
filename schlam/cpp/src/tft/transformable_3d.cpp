
#include "transformable_3d.hpp"

#include <stdexcept>

namespace tft {

Transformable3D::Transformable3D(const Eigen::Vector3d &aVector,
                                 const std::string &aCoordinateFrame)
    : vector(aVector), coordinateFrame(aCoordinateFrame) {}

double Transformable3D::x() const { return vector.x(); }

double Transformable3D::y() const { return vector.y(); }

double Transformable3D::z() const { return vector.z(); }

std::string Transformable3D::getCF() const { return coordinateFrame; }

std::ostream& operator<<(std::ostream& os, const Transformable3D& obj) {
    os << "[" << obj.x() << ", " << obj.y() << ", " << obj.z() << ") [" << obj.getCF() << "]"; 
    return os;
}

} // namespace tft