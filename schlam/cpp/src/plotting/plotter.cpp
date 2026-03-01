#include "plotter.hpp"

#include <Eigen/src/Geometry/Quaternion.h>
#include <GL/gl.h>

#include <pangolin/display/default_font.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler.h>

#include "schlam/KeyPoint.h"

// -----------------------------
// Helper drawing functions
// -----------------------------
void Plotter::DrawGrid(int size, float step) {
    glColor3f(0.f, 0.f, 0.f);
    glBegin(GL_LINES);
    for (int i = -size; i <= size; ++i) {
        glVertex3f(i * step, -size * step, 0.f);
        glVertex3f(i * step, size * step, 0.f);

        glVertex3f(-size * step, i * step, 0.f);
        glVertex3f(size * step, i * step, 0.f);
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

bool Plotter::shouldPause() {
    std::lock_guard guard(mPauseMutex);
    return mPause;
}

void Plotter::setup() {
    pangolin::CreateWindowAndBind("3D Visualizer", 1920, 1080);

    glEnable(GL_MULTISAMPLE);

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
}

void Plotter::updatePointCloud(const std::vector<Eigen::Vector3f> &aPoints, const std::string aCF) {
    std::vector<Eigen::Vector3f> points;
    auto transform = mpTransformer->findTransform(aCF, "world");
    for (const auto& pt : aPoints) {
        points.push_back(transform->mRotation * pt + transform->mTranslation);
    }
    mCloud = points;
}

void Plotter::addTransform(
    const std::shared_ptr<tft::RigidTransform3D> transform) {
    mTransforms.push_back(transform);
}

void Plotter::updateFrustum(const std::shared_ptr<ImageData> aImageData) {
    std::lock_guard guard(mFrustumMutex);
    mFrustum = aImageData;
    mFrustumPose = mpTransformer->findTransform(mFrustum->mCoordinateFrame, "world");
    m3DImageChanged = true;
}

pangolin::OpenGlMatrix
Plotter::GetPangolinModelMatrix(const Eigen::Matrix3f &R,
                                const Eigen::Vector3f &t) const {
    pangolin::OpenGlMatrix m;
    m.SetIdentity();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m.m[j * 4 + i] = R(i, j); // Note: column-major index j*4 + i
        }
    }

    m.m[12] = t.x();
    m.m[13] = t.y();
    m.m[14] = t.z();

    return m;
}

void Plotter::plotTransform(
    const std::shared_ptr<tft::RigidTransform3D> transform, const double radius,
    const double length, const bool showFrameName) {
    // Invert the transform, because gl apparently has the inverse definition
    auto transformInWorld =
            mpTransformer->findTransform(transform->mTarget, "world");

    glPushMatrix();

    pangolin::OpenGlMatrix Twc = GetPangolinModelMatrix(
        transformInWorld->mRotation, transformInWorld->mTranslation);
    glMultMatrixd(Twc.m);
    drawAxes(radius, length);
    glColor3f(1.0, 1.0, 1.0); // Set text color

    if (showFrameName) {
        pangolin::default_font().Text(transform->mTarget).Draw(0, 0, 0);
    }
    glPopMatrix();
}

void Plotter::drawAxes(const double radius, const double length) {
    glPushMatrix();

    // X axis (red)
    glColor3f(1.f, 0.f, 0.f);
    glPushMatrix();
    glRotatef(90.f, 0.f, 1.f, 0.f); // Z → X
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

void Plotter::drawCylinder(float radius, float length, int slices) {
    const float TWO_PI = 2.0f * M_PI;

    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i <= slices; ++i) {
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

void Plotter::plotFrustum(std::shared_ptr<ImageData> aImageData, std::shared_ptr<tft::RigidTransform3D> aTransform, double alpha) {
    std::lock_guard guard(mFrustumMutex);
    glPushMatrix();

    if (m3DImageChanged) {
        m3DImageTexture = std::make_unique<pangolin::GlTexture>(
            aImageData->mImage.cols,
            aImageData->mImage.rows,
            GL_LUMINANCE8, true, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE
        );

        m3DImageTexture->Upload(aImageData->mImage.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
        m3DImageChanged = false;
    }

    pangolin::OpenGlMatrix Twc = GetPangolinModelMatrix(
        aTransform->mRotation, aTransform->mTranslation);
    glMultMatrixd(Twc.m);

    auto kInv = aImageData->mIntrinsics.inverse().eval();
    float scale = 0.1f; // The depth at which to draw
    pangolin::glDrawFrustum(kInv, aImageData->mImage.cols, aImageData->mImage.rows, scale);

    glDepthMask(GL_FALSE);
    glEnable(GL_TEXTURE_2D);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    m3DImageTexture->Bind();

    glColor4f(1.0f, 1.0f, 1.0f, alpha); // Reset color so texture isn't tinted
    glBegin(GL_QUADS);

    // Define the 4 corners in Camera Space using K-inverse
    glTexCoord2f(0, 1);
    Eigen::Vector3f bl = kInv * Eigen::Vector3f(0, aImageData->mImage.rows, 1) * scale;
    glVertex3d(bl[0], bl[1], bl[2]);
    glTexCoord2f(1, 1);
    Eigen::Vector3f br = kInv * Eigen::Vector3f(aImageData->mImage.cols, aImageData->mImage.rows, 1) * scale;
    glVertex3d(br[0], br[1], br[2]);
    glTexCoord2f(1, 0);
    Eigen::Vector3f tr = kInv * Eigen::Vector3f(aImageData->mImage.cols, 0, 1) * scale;
    glVertex3d(tr[0], tr[1], tr[2]);
    glTexCoord2f(0, 0);
    Eigen::Vector3f tl = kInv * Eigen::Vector3f(0, 0, 1) * scale;
    glVertex3d(tl[0], tl[1], tl[2]);

    glEnd();

    m3DImageTexture->Unbind();
    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
    glDepthMask(GL_TRUE);
}

void Plotter::plotFeatures(const cv::Mat &aImage, const std::vector<KeyPoint> &aFeatures) {
    if (aImage.empty()) {
        std::cout << "Invalid image!" << std::endl;
        return;
    }
    cv::Mat colorResult;
    if (aImage.channels() == 1) {
        cv::cvtColor(aImage, colorResult, cv::COLOR_GRAY2BGR);
    } else {
        colorResult = aImage.clone();
    }

    for (const auto &p: aFeatures) {
        int targetX = p.getImgX(); // * aFactor;
        int targetY = p.getImgY(); //* aFactor;

        if (targetX >= 0 && targetX < colorResult.cols &&
            targetY >= 0 && targetY < colorResult.rows) {
            cv::circle(colorResult, cv::Point(targetX, targetY), 0, cv::Scalar(0, 0, 255), -1);
        }
    }

    setFeatureTexture(colorResult);
}

void Plotter::plotMatches(const cv::Mat &aImage1, const cv::Mat &aImage2, const std::vector<KeyPoint> &aFeatures1,
                  const std::vector<KeyPoint> &aFeatures2, const std::vector<std::array<std::uint32_t, 2>> aMatches) {
    if (aImage1.empty() || aImage2.empty()) {
        std::cout << "Invalid image!" << std::endl;
        return;
    }

    cv::Mat combined;
    cv::vconcat(aImage1, aImage2, combined);

    cv::Mat colorResult;
    if (combined.channels() == 1) {
        cv::cvtColor(combined, colorResult, cv::COLOR_GRAY2BGR);
    } else {
        colorResult = combined.clone();
    }

    int offset = aImage1.rows;

    for (const auto& [idx1, idx2] : aMatches) {
        if (idx1 < aFeatures1.size() && idx2 < aFeatures2.size()) {
            cv::Point2f pt1{static_cast<float>(aFeatures1[idx1].getImgX()), static_cast<float>(aFeatures1[idx1].getImgY())};
            cv::Point2f pt2{static_cast<float>(aFeatures2[idx2].getImgX()), static_cast<float>(aFeatures2[idx2].getImgY() + offset)};
            cv::line(colorResult, pt1, pt2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
    }

    for (const auto& kp : aFeatures1) {
        cv::Point2f pt{static_cast<float>(kp.getImgX()), static_cast<float>(kp.getImgY())};
        cv::circle(colorResult, pt, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }

    for (const auto& kp : aFeatures2) {
        cv::Point2f pt{static_cast<float>(kp.getImgX()), static_cast<float>(kp.getImgY() + offset)};
        cv::circle(colorResult, pt, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }

    setMatcherTexture(colorResult);
}

void Plotter::setFeatureTexture(const cv::Mat &aImage) {
    mFeatureImage = aImage;
    mFeatureImageChanged = true;
}

void Plotter::setMatcherTexture(const cv::Mat &aImage) {
    mMatcherImage = aImage;
    mMatcherImageChanged = true;
}

void Plotter::showFeatures() {
    if (mFeatureImage.empty()) return;

    if (mFeatureImageChanged) {
        mFeatureImageTexture = std::make_unique<pangolin::GlTexture>(
            mFeatureImage.cols,
            mFeatureImage.rows,
            GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE
        );

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        mFeatureImageTexture->Upload(mFeatureImage.data, GL_BGR, GL_UNSIGNED_BYTE);
        mFeatureImageChanged = false;
    }

    glColor3f(1.0, 1.0, 1.0);
    mFeatureImageTexture->RenderToViewportFlipY();
}

void Plotter::showMatches() {
    if (mMatcherImage.empty()) return;

    if (mMatcherImageChanged) {
        mMatcherImageTexture = std::make_unique<pangolin::GlTexture>(
            mMatcherImage.cols,
            mMatcherImage.rows,
            GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE
        );

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        mMatcherImageTexture->Upload(mMatcherImage.data, GL_BGR, GL_UNSIGNED_BYTE);
        mMatcherImageChanged = false;
    }

    glColor3f(1.0, 1.0, 1.0);
    mMatcherImageTexture->RenderToViewportFlipY();
}


void Plotter::run() {
    pangolin::BindToContext("3D Visualizer");
    // enable depth
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLint samples = 0;
    glGetIntegerv(GL_SAMPLES, &samples);
    std::cout << "MSAA samples: " << samples << std::endl;

    // 1. Camera setup
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1920, 1080, 840, 840, 1920 / 2, 1080 / 2, 0.1,
                                   1000),
        pangolin::ModelViewLookAt(20, -20, 20, 0, 0, 0, pangolin::AxisZ));

    // 2. 3D viewport
    pangolin::Handler3D handler(s_cam, pangolin::AxisZ);
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, 1920.0f / 1080.0f)
            .SetHandler(&handler);


    pangolin::View &featureView = pangolin::CreateDisplay()
            .SetBounds(0.6, 1.0, 0.6, 1.0, 1920.f / 1080)
            .SetLock(pangolin::LockRight, pangolin::LockTop);

    pangolin::View &matcherView = pangolin::CreateDisplay()
        .SetBounds(0.0, 0.6, 0.6, 1.0, 1920.f / (1080*2))
        .SetLock(pangolin::LockRight, pangolin::LockBottom);

    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0,
                                            pangolin::Attach::Pix(200));
    pangolin::Var<bool> menu_pause("menu.Pause", false, true);
    pangolin::Var<bool> menu_showFrames("menu.Show Frames", true, true);
    pangolin::Var<bool> menu_showFrameNames("menu.Show Frame Names", true, true);
    pangolin::Var<bool> menu_showFrustums("menu.Show Frustums", true, true);
    pangolin::Var<double> menu_3DImageAlpha("menu.3D Image Alpha", true, 0, 1);
    pangolin::Var<bool> menu_showFeatures("menu.Show Features", false, true);
    pangolin::Var<bool> menu_showMatches("menu.Show Matches", true, true);

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

        if (menu_showFrustums && mFrustum) {
            plotFrustum(mFrustum, mFrustumPose, menu_3DImageAlpha);
        }

        if (menu_pause) {
            std::lock_guard guard(mPauseMutex);
            mPause = true;
        } else {
            std::lock_guard guard(mPauseMutex);
            mPause = false;
        }

        // Draw all transforms
        if (menu_showFrames) {
            for (auto &transform: mpTransformer->getRootedTransforms()) {
                plotTransform(transform, 0.005, 0.1, menu_showFrameNames);
            }
        }

        featureView.Activate();
        if (menu_showFeatures) {
            showFeatures();
        }

        matcherView.Activate();
        if (menu_showMatches) {
            showMatches();
        }



        pangolin::FinishFrame();
    }
}
