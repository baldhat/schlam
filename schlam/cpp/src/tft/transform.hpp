#pragma once

#include "transformable_3d.hpp"

#include <string>

namespace tft {
    /**
    * @brief A Transform object describes how to express data described in the target coordinate frame
    *        in the source coordinate frame and reversed.
    */
    class Transform {
    public:
        Transform(const std::string& source, const std::string& target);
        ~Transform() = default;

        virtual Transformable3D apply(const Transformable3D& source) = 0;
        virtual Transformable3D apply(const Transformable3D&& source) = 0;
        virtual Transformable3D applyInverse(const Transformable3D& source) = 0;
        virtual Transformable3D applyInverse(const Transformable3D&& source) = 0;

        const std::string source;
        const std::string target;

    private:
        
    };

}