
#include "plotter.hpp"

#include <Eigen/src/Geometry/Quaternion.h>
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
        glVertex3f(i*step, -size*step, 0.f);
        glVertex3f(i*step, size*step, 0.f  );

        glVertex3f(-size*step, i*step, 0.f);
        glVertex3f( size*step, i*step, 0.f);
    }
    glEnd();
}

// -----------------------------
// Plotter methods
// -----------------------------
Plotter::Plotter(std::shared_ptr<tft::Transformer> apTransformer) 
        : mpTransformer(apTransformer) {
    setup();
}

void Plotter::setup() {
    // create a window and bind its context to the main thread
    pangolin::CreateWindowAndBind("3D Visualizer", 1920, 1080, pangolin::Params({{"samples", "4"}}));

    // enable depth
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    // Enable Blending for smooth edges
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Enable Line Smoothing
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    // Enable Point Smoothing
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
}


void Plotter::updatePointCloud(const std::vector<Eigen::Vector3d>& points)
{
    mCloud = points;
    std::cout << "Updated point cloud" << std::endl;
}

void Plotter::addTransform(const tft::RigidTransform3D& transform)
{
    mTransforms.push_back(transform);
}

pangolin::OpenGlMatrix Plotter::GetPangolinModelMatrix(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    pangolin::OpenGlMatrix m;
    m.SetIdentity();

    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            m.m[j*4 + i] = R(i, j); // Note: column-major index j*4 + i
        }
    }

    m.m[12] = t.x();
    m.m[13] = t.y();
    m.m[14] = t.z();

    return m;
}


void Plotter::plotTransform(const tft::RigidTransform3D& transform) {
    // Invert the transform, because gl apparently has the inverse definition
    auto transformInWorld = mpTransformer->findTransform(transform.target, "world").inverse();

    glPushMatrix();

    pangolin::OpenGlMatrix Twc = GetPangolinModelMatrix(transformInWorld.rotation, transformInWorld.translation);
    glMultMatrixd(Twc.m);
    pangolin::glDrawAxis(1.0); 

    glPopMatrix();
}

void Plotter::run()
{
    pangolin::BindToContext("3D Visualizer");
    glEnable(GL_DEPTH_TEST);

    // 1. Camera setup
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1920, 1080, 840, 840, 1920/2, 1080/2, 0.1, 1000),
        pangolin::ModelViewLookAt(20, -20, 20, 0, 0, 0, pangolin::AxisZ)
    );

    // 2. 3D viewport
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, 1920.0f/1080.0f)
        .SetHandler(&handler);

    // 3. Main loop
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // Draw grid
        glLineWidth(1.0f);
        DrawGrid(10, 1.0f);

        // // Draw points
        // if (!mCloud.empty()) {
        //     glPointSize(5.0f);
        //     glColor3f(1.f, 0.f, 0.f);
        //     glBegin(GL_POINTS);
        //     for (const auto& p : mCloud) {
        //         glVertex3f(p.x(), p.y(), p.z());
        //     }
        //     glEnd();
        // }

        // Draw all transforms
        // Always draw world transform
        glPushMatrix();
        glLineWidth(3.0f);
        glTranslatef(0.0f, 0.0f, 0.0f);
        pangolin::glDrawAxis(1.0);
        glPopMatrix();
        for (auto& transform : mTransforms) {
            plotTransform(transform);
        }

        pangolin::FinishFrame();
    }
}
