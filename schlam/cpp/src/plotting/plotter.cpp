
#include "plotter.hpp"

#include <Eigen/src/Geometry/Quaternion.h>
#include <GL/gl.h>

#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler.h>
#include <pangolin/display/default_font.h>

// -----------------------------
// Helper drawing functions
// -----------------------------
void Plotter::DrawGrid(int size, float step)
{
    glColor3f(0.f, 0.f, 0.f);
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
    pangolin::Params p;
    p.Set("window_title", "3D Visualizer");
    p.Set("w", "1920");
    p.Set("h", "1080");

    pangolin::CreateWindowAndBind("3D Visualizer", 1920, 1080, p);

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
}


void Plotter::updatePointCloud(const std::vector<Eigen::Vector3d>& points)
{
    mCloud = points;
    std::cout << "Updated point cloud" << std::endl;
}

void Plotter::addTransform(const std::shared_ptr<tft::RigidTransform3D> transform)
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


void Plotter::plotTransform(const std::shared_ptr<tft::RigidTransform3D> transform, 
                            const double radius, 
                            const double length,
                            const bool showFrameName) {
    // Invert the transform, because gl apparently has the inverse definition
    auto transformInWorld = mpTransformer->findTransform(transform->source, "world");

    glPushMatrix();

    pangolin::OpenGlMatrix Twc = GetPangolinModelMatrix(transformInWorld->rotation, transformInWorld->translation);
    glMultMatrixd(Twc.m);
    drawAxes(radius, length);
    glColor3f(1.0, 1.0, 1.0); // Set text color
    
    if (showFrameName) {
        pangolin::default_font().Text(transform->source).Draw(0, 0, 0);
    }
    glPopMatrix();
}

void Plotter::drawAxes(const double radius, const double length) {
    glPushMatrix();

    // X axis (red)
    glColor3f(1.f, 0.f, 0.f);
    glPushMatrix();
    glRotatef(90.f, 0.f, 1.f, 0.f);  // Z → X
    drawCylinder(radius, length);
    glPopMatrix();

    // Y axis (green)
    glColor3f(0.f, 1.f, 0.f);
    glPushMatrix();
    glRotatef(-90.f, 1.f, 0.f, 0.f); // Z → Y
    drawCylinder(radius, length);
    glPopMatrix();

    // Z axis (blue)
    glColor3f(0.f, 0.f, 1.f);
    drawCylinder(radius, length);

    glPopMatrix();
}

void Plotter::drawCylinder(float radius, float length, int slices)
{
    const float TWO_PI = 2.0f * M_PI;

    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i <= slices; ++i)
    {
        float theta = TWO_PI * i / slices;
        float x = radius * cos(theta);
        float y = radius * sin(theta);

        // Normal
        glNormal3f(cos(theta), sin(theta), 0.0f);

        // Bottom
        glVertex3f(x, y, 0.0f);
        // Top
        glVertex3f(x, y, length);
    }
    glEnd();
}

void Plotter::run()
{
    pangolin::BindToContext("3D Visualizer");
    // enable depth
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE); 
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    GLint samples = 0;
    glGetIntegerv(GL_SAMPLES, &samples);
    std::cout << "MSAA samples: " << samples << std::endl;

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

    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);

    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(200));
    pangolin::Var<bool> menu_showFrames("menu.Show Frames",true,true);
    pangolin::Var<bool> menu_showFrameNames("menu.Show Frame Names",true,true);

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
        if (menu_showFrames) {
            glPushMatrix();
            glLineWidth(3.0f);
            glTranslatef(0.0f, 0.0f, 0.0f);
            drawAxes(0.05, 1.0);
            glPopMatrix();
            for (auto& transform : mpTransformer->getRoot()) {
                plotTransform(transform, 0.05, 1.0, menu_showFrameNames);
            }
        }

        pangolin::FinishFrame();
    }
}

