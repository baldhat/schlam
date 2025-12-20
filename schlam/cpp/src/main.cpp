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

    auto pTransformer = std::make_shared<tft::Transformer>();
    Plotter plotter(pTransformer);
    
   
    auto render_loop = std::thread(std::bind(&Plotter::run, &plotter));

    std::this_thread::sleep_for(std::chrono::seconds(5));

    
    std::vector<Eigen::Vector3d> points2;
    for (float x = -5; x <= 5; x += 1.0f) {
        for (float z = -5; z <= 5; z += 1.0f) {
            points2.emplace_back(x, 0.5f * (x*x + z*z)/25.0f, z); // some height variation
        }
    }
    plotter.updatePointCloud(points2);

    render_loop.join();
    

    return 0;
}
