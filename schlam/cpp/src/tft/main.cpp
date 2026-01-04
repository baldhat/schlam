#include "../plotting/plotter.hpp"

#include "rigid_transform_3d.hpp"
#include "transformable_3d.hpp"
#include "transformer.hpp"

#include <eigen3/Eigen/Core>

#include <eigen3/Eigen/src/Core/Matrix.h>
#include <thread>
#include <vector>

int main()
{
    auto pTransformer = std::make_shared<tft::Transformer>();
    Plotter plotter(pTransformer);
    
   
    auto render_loop = std::thread(std::bind(&Plotter::run, &plotter));

    Eigen::Matrix3d rotZ180{{-1, 0, 0}, {0, -1, 0}, {0, 0, 1}}; 
    Eigen::Matrix3d rotZ90{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};

    std::vector<std::shared_ptr<tft::RigidTransform3D>> transforms;
    transforms.push_back(std::make_shared<tft::RigidTransform3D>("frame1", "world", rotZ180, Eigen::Vector3d(-1, -1, -1)));
    transforms.push_back(std::make_shared<tft::RigidTransform3D>("frame2", "world", rotZ90, Eigen::Vector3d(3, 0, 0)));
    transforms.push_back(std::make_shared<tft::RigidTransform3D>("frame3", "world", rotZ90, Eigen::Vector3d(1, 1, 1)));
    transforms.push_back(std::make_shared<tft::RigidTransform3D>("frame4", "frame3", rotZ90, Eigen::Vector3d(1, 1, 1)));
    transforms.push_back(std::make_shared<tft::RigidTransform3D>("frame5", "frame4", rotZ90, Eigen::Vector3d(1, 1, 1)));


    for (auto transform : transforms) {
        pTransformer->registerTransform(transform);
        
        plotter.addTransform(transform);

        auto x = tft::Transformable3D(Eigen::Vector3d(1, 0, 0), transform->mSource);
        auto y = tft::Transformable3D(Eigen::Vector3d(0, 1, 0), transform->mSource);
        auto z = tft::Transformable3D(Eigen::Vector3d(0, 0, 1), transform->mSource);

        std::cout << "x: " << x << std::endl;
        std::cout << "y: " << y << std::endl;
        std::cout << "z: " << z << std::endl;

        auto x1 = transform->apply(x);
        auto y1 = transform->apply(y);
        auto z1 = transform->apply(z);

        std::cout << "x1: " << x1 << std::endl;
        std::cout << "y1: " << y1 << std::endl;
        std::cout << "z1: " << z1 << std::endl;

        auto x1b = transform->applyInverse(x1);
        auto y1b = transform->applyInverse(y1);
        auto z1b = transform->applyInverse(z1);

        std::cout << "x1b: " << x1b << std::endl;
        std::cout << "y1b: " << y1b << std::endl;
        std::cout << "z1b: " << z1b << std::endl;

        auto inverse = transform->inverse();
        auto x1r = inverse->apply(x1);
        auto y1r = inverse->apply(y1);
        auto z1r = inverse->apply(z1);

        std::cout << "x1r: " << x1b << std::endl;
        std::cout << "y1r: " << y1b << std::endl;
        std::cout << "z1r: " << z1b << std::endl;
        std::cout << std::endl;
    }

    render_loop.join();
    

    return 0;
}