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


    
    

    render_loop.join();
    return 0;
}
