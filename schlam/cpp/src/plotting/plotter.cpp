
#include "plotter.hpp"

#include <GL/gl.h>
#include <pangolin/handler/handler.h>

// -----------------------------
// Helper drawing functions
// -----------------------------
void Plotter::DrawGrid(int size, float step)
{
    glColor3f(0.5f, 0.5f, 0.5f);
    glBegin(GL_LINES);
    for (int i = -size; i <= size; ++i) {
        glVertex3f(i*step, 0.f, -size*step);
        glVertex3f(i*step, 0.f,  size*step);

        glVertex3f(-size*step, 0.f, i*step);
        glVertex3f( size*step, 0.f, i*step);
    }
    glEnd();
}

// -----------------------------
// Plotter methods
// -----------------------------
Plotter::Plotter() {
    setup();
}

void Plotter::setup() {
    // create a window and bind its context to the main thread
    pangolin::CreateWindowAndBind("3D Visualizer", 1280, 720);

    // enable depth
    glEnable(GL_DEPTH_TEST);

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
}


void Plotter::updatePointCloud(const std::vector<Eigen::Vector3f>& points)
{
    mCloud = points;
    std::cout << "Updated point cloud" << std::endl;
}

void Plotter::run()
{
    pangolin::BindToContext("3D Visualizer");
    glEnable(GL_DEPTH_TEST);

    // 1. Camera setup
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 420, 420, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(20, -20, 20, 0, 0, 0, pangolin::AxisNegY)
    );

    // 2. 3D viewport
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, 1024.0f/768.0f)
        .SetHandler(&handler);

    // 3. Main loop
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // Draw grid
        glLineWidth(1.0f);
        DrawGrid(10, 1.0f);

        // Draw points
        if (!mCloud.empty()) {
            glPointSize(5.0f);
            glColor3f(1.f, 0.f, 0.f);
            glBegin(GL_POINTS);
            for (const auto& p : mCloud) {
                glVertex3f(p.x(), p.y(), p.z());
            }
            glEnd();
        }

        // Draw axis
        glPushMatrix();
        glTranslatef(0.0f, 0.0f, 0.0f);
        pangolin::glDrawAxis(2.0);
        glPopMatrix();

        pangolin::FinishFrame();
    }
}
