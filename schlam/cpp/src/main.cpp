#include "plotting/plotter.hpp"
#include "tft/rigid_transform_3d.hpp"
#include "tft/transformable_3d.hpp"

#include <eigen3/Eigen/Core>

#include <chrono>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <thread>
#include <vector>

int main()
{
    //Plotter plotter;
    
   
    // auto render_loop = std::thread(std::bind(&Plotter::run, &plotter));

    // std::this_thread::sleep_for(std::chrono::seconds(5));

    
    // std::vector<Eigen::Vector3f> points2;
    // for (float x = -5; x <= 5; x += 1.0f) {
    //     for (float z = -5; z <= 5; z += 1.0f) {
    //         points2.emplace_back(x, 0.5f * (x*x + z*z)/25.0f, z); // some height variation
    //     }
    // }
    // plotter.updatePointCloud(points2);

    // render_loop.join();

    auto x = tft::Transformable3D(Eigen::Vector3f(1, 0, 0), "world");
    auto y = tft::Transformable3D(Eigen::Vector3f(0, 1, 0), "world");
    auto z = tft::Transformable3D(Eigen::Vector3f(0, 0, 1), "world");
    
    Eigen::Matrix3f rotZ180;
    rotZ180 << -1, 0, 0, 0, -1, 0, 0, 0, 1; 
    Eigen::Matrix3f rotZ90;
    rotZ90 << 0, -1, 0, 1, 0, 0, 0, 0, 1;

    std::vector<tft::RigidTransform3D> transforms;
    transforms.push_back(tft::RigidTransform3D("world", "frame1", rotZ180, Eigen::Vector3f(0, 0, 0)));
    transforms.push_back(tft::RigidTransform3D("world", "frame2", rotZ90, Eigen::Vector3f(0, 0, 0)));
    transforms.push_back(tft::RigidTransform3D("world", "frame3", rotZ90, Eigen::Vector3f(1, 1, 1)));


    for (auto& transform : transforms) {
        std::cout << "x: " << x << std::endl;
        std::cout << "y: " << y << std::endl;
        std::cout << "z: " << z << std::endl;

        auto x1 = transform.apply(x);
        auto y1 = transform.apply(y);
        auto z1 = transform.apply(z);

        std::cout << "x1: " << x1 << std::endl;
        std::cout << "y1: " << y1 << std::endl;
        std::cout << "z1: " << z1 << std::endl;

        auto x1b = transform.applyInverse(x1);
        auto y1b = transform.applyInverse(y1);
        auto z1b = transform.applyInverse(z1);

        std::cout << "x1b: " << x1b << std::endl;
        std::cout << "y1b: " << y1b << std::endl;
        std::cout << "z1b: " << z1b << std::endl;
        std::cout << std::endl;
    }
    

    return 0;
}
