#include "plotter.hpp"

#include <Eigen/src/Geometry/Quaternion.h>
#include <GL/gl.h>
#include <functional>

#include <pangolin/display/default_font.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler.h>

#include "schlam/KeyPoint.h"
#include "random.hpp"

// -----------------------------
// Helper drawing functions
// -----------------------------
void Plotter::DrawGrid(int size, double step) {
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
Plotter::Plotter(std::shared_ptr<tft::Transformer> apTransformer, const Eigen::Matrix3d& aCameraIntrinsics)
    : mpTransformer(apTransformer)
    , mCameraIntrinsics(aCameraIntrinsics) {
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

void Plotter::updatePointCloud(const std::vector<Eigen::Vector3d> &aPoints, const std::string aCF) {
    std::vector<Eigen::Vector3d> points;
    auto transform = mpTransformer->findTransform(aCF, "world");
    for (const auto &pt: aPoints) {
        points.push_back(transform->mRotation * pt + transform->mTranslation);
    }
    mCloud = points;
}

void Plotter::addTransform(
    const std::shared_ptr<tft::RigidTransform3D> transform) {
    mTransforms.push_back(transform);
}

void Plotter::updateFrustum(const std::shared_ptr<ImageData> aImageData,
                           const std::shared_ptr<tft::RigidTransform3D> aPredictedPose) {
    const auto cam0Pose = aImageData
        ? mpTransformer->findTransform(aImageData->mCoordinateFrame, "world") : nullptr;
    std::lock_guard guard(mFrustumMutex);
    mFrustum = aImageData;
    mFrustumPose = cam0Pose;
    mPredictedFrustumPose = aPredictedPose;
    m3DImageChanged = true;
}

void Plotter::updateTimestamp(const utils::TTimestamp& aTimestamp) {
    std::lock_guard guard(mTimestampMutex);
    if (!mSequenceStartTimestamp) mSequenceStartTimestamp = aTimestamp;
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        aTimestamp - *mSequenceStartTimestamp).count();
    mTimestampText = std::to_string(milliseconds) + " ms";
}

void Plotter::resetCameraPaths(const Eigen::Vector3d& aReferencePosition) {
    std::lock_guard guard(mCameraPathsMutex);
    mGroundTruthCameraPath = {aReferencePosition};
    mPredictedCameraPath = {aReferencePosition};
}

void Plotter::addGroundTruthCameraPosition(const Eigen::Vector3d& aPosition) {
    std::lock_guard guard(mCameraPathsMutex);
    mGroundTruthCameraPath.push_back(aPosition);
}

void Plotter::addPredictedCameraPosition(const Eigen::Vector3d& aPosition) {
    std::lock_guard guard(mCameraPathsMutex);
    mPredictedCameraPath.push_back(aPosition);
}

void Plotter::drawCameraPaths(bool aShowGroundTruth, bool aShowPredicted) {
    std::lock_guard guard(mCameraPathsMutex);
    glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_LINE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);

    const auto drawPath = [](const std::vector<Eigen::Vector3d>& path,
                             float red, float green, float blue, float width) {
        if (path.size() < 2) return;
        glColor3f(red, green, blue);
        glLineWidth(width);
        glBegin(GL_LINE_STRIP);
        for (const auto& position : path) {
            glVertex3d(position.x(), position.y(), position.z());
        }
        glEnd();
    };

    if (aShowGroundTruth) drawPath(mGroundTruthCameraPath, 0.f, 1.f, 0.f, 3.f);
    if (aShowPredicted) drawPath(mPredictedCameraPath, 0.f, 0.f, 1.f, 2.f);
    glPopAttrib();
}

pangolin::OpenGlMatrix
Plotter::GetPangolinModelMatrix(const Eigen::Matrix3d &R,
                                const Eigen::Vector3d &t) const {
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

void Plotter::plotMatches(const cv::Mat &oldFrame, const cv::Mat &newFrame,
                          const std::array<std::vector<std::shared_ptr<KeyPoint>>, 2>& aMatches) {
    if (oldFrame.empty() || newFrame.empty()) {
        std::cerr << "Invalid image!" << std::endl;
        return;
    }

    cv::Mat combined;
    cv::vconcat(oldFrame, newFrame, combined);

    cv::Mat colorResult;
    if (combined.channels() == 1) {
        cv::cvtColor(combined, colorResult, cv::COLOR_GRAY2BGR);
    } else {
        colorResult = combined.clone();
    }

    const int offset = oldFrame.rows;
    const auto &features1 = aMatches[0];
    const auto &features2 = aMatches[1];

    const int halfSize = 4;
    const cv::Scalar green{0, 255, 0};
    auto& engine = utils::randomEngine();
    std::uniform_int_distribution<int> colorChannel(64, 255);

    for (size_t i = 0; i < std::min(features1.size(), features2.size()); ++i) {
        cv::Point2d pt1{static_cast<double>(features1[i]->getImgX()), static_cast<double>(features1[i]->getImgY()+ halfSize) };
        cv::Point2d pt2{static_cast<double>(features2[i]->getImgX()), static_cast<double>(features2[i]->getImgY() + offset - halfSize)};
        const int blue = colorChannel(engine);
        const int greenValue = colorChannel(engine);
        const int red = colorChannel(engine);
        const cv::Scalar lineColor(blue, greenValue, red);
        cv::line(colorResult, pt1, pt2, lineColor, 1, cv::LINE_AA);
    }

    for (const auto &kp : features1) {
        cv::Point pt{static_cast<int>(kp->getImgX()), static_cast<int>(kp->getImgY())};
        cv::rectangle(colorResult,
                      {pt.x - halfSize, pt.y - halfSize},
                      {pt.x + halfSize, pt.y + halfSize},
                      green, 1);
    }

    for (const auto &kp : features2) {
        cv::Point pt{static_cast<int>(kp->getImgX()), static_cast<int>(kp->getImgY() + offset)};
        cv::rectangle(colorResult,
                      {pt.x - halfSize, pt.y - halfSize},
                      {pt.x + halfSize, pt.y + halfSize},
                      green, 1);
    }

    setMatcherTexture(colorResult);
}

void Plotter::drawCylinder(double radius, double length, int slices) {
    const double TWO_PI = 2.0f * M_PI;

    glBegin(GL_TRIANGLE_STRIP);
    for (int i = 0; i <= slices; ++i) {
        double theta = TWO_PI * i / slices;
        double x = radius * cos(theta);
        double y = radius * sin(theta);

        // Normal
        glNormal3f(cos(theta), sin(theta), 0.0f);

        // Bottom
        glVertex3f(x, y, 0.0f);
        // Top
        glVertex3f(x, y, length);
    }
    glEnd();
}

void Plotter::drawFrustums(bool aShowCam0, bool aShowPredicted, double aAlpha) {
    std::lock_guard guard(mFrustumMutex);
    if (!mFrustum || mFrustum->mImage.empty()) return;
    if (!(aShowCam0 && mFrustumPose) && !(aShowPredicted && mPredictedFrustumPose)) return;

    glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT | GL_TEXTURE_BIT | GL_LINE_BIT);
    glDisable(GL_TEXTURE_2D);

    if (m3DImageChanged || !m3DImageTexture) {
        const auto& image = mFrustum->mImage;
        if (!m3DImageTexture || m3DImageTexture->width != image.cols ||
            m3DImageTexture->height != image.rows) {
            m3DImageTexture = std::make_unique<pangolin::GlTexture>(
                image.cols, image.rows, GL_LUMINANCE8, true, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE);
        }
        const cv::Mat pixels = image.isContinuous() ? image : image.clone();
        glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        m3DImageTexture->Upload(pixels.data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
        glPopClientAttrib();
        m3DImageChanged = false;
    }

    if (aShowCam0 && mFrustumPose) {
        glColor3f(0.f, 1.f, 0.f);
        glLineWidth(3.f);
        plotFrustum(mFrustum, mFrustumPose, aAlpha);
    }
    if (aShowPredicted && mPredictedFrustumPose) {
        glColor3f(0.f, 0.f, 1.f);
        glLineWidth(2.f);
        plotFrustum(mFrustum, mPredictedFrustumPose, aAlpha);
    }
    glPopAttrib();
}

void Plotter::plotFrustum(std::shared_ptr<ImageData> aImageData, std::shared_ptr<tft::RigidTransform3D> aTransform,
                          double alpha) {
    glPushMatrix();

    pangolin::OpenGlMatrix Twc = GetPangolinModelMatrix(
        aTransform->mRotation, aTransform->mTranslation);
    glMultMatrixd(Twc.m);

    auto kInv = aImageData->mIntrinsics.inverse().eval();
    double scale = 0.1f; // The depth at which to draw
    pangolin::glDrawFrustum(kInv, aImageData->mImage.cols, aImageData->mImage.rows, scale);

    glDepthMask(GL_FALSE);
    glEnable(GL_TEXTURE_2D);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    m3DImageTexture->Bind();

    glColor4f(1.0f, 1.0f, 1.0f, alpha); // Reset color so texture isn't tinted
    glBegin(GL_QUADS);

    // Define the 4 corners in Camera Space using K-inverse
    glTexCoord2f(0, 1);
    Eigen::Vector3d bl = kInv * Eigen::Vector3d(0, aImageData->mImage.rows, 1) * scale;
    glVertex3d(bl[0], bl[1], bl[2]);
    glTexCoord2f(1, 1);
    Eigen::Vector3d br = kInv * Eigen::Vector3d(aImageData->mImage.cols, aImageData->mImage.rows, 1) * scale;
    glVertex3d(br[0], br[1], br[2]);
    glTexCoord2f(1, 0);
    Eigen::Vector3d tr = kInv * Eigen::Vector3d(aImageData->mImage.cols, 0, 1) * scale;
    glVertex3d(tr[0], tr[1], tr[2]);
    glTexCoord2f(0, 0);
    Eigen::Vector3d tl = kInv * Eigen::Vector3d(0, 0, 1) * scale;
    glVertex3d(tl[0], tl[1], tl[2]);

    glEnd();

    m3DImageTexture->Unbind();
    glDisable(GL_TEXTURE_2D);
    glPopMatrix();
    glDepthMask(GL_TRUE);
}

void Plotter::plotFeatures(const cv::Mat &aImage, const std::vector<std::shared_ptr<KeyPoint>>& aFeatures) {
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
        int targetX = p->getImgX(); // * aFactor;
        int targetY = p->getImgY(); //* aFactor;

        if (targetX >= 0 && targetX < colorResult.cols &&
            targetY >= 0 && targetY < colorResult.rows) {
            cv::circle(colorResult, cv::Point(targetX, targetY), 0, cv::Scalar(0, 0, 255), -1);
        }
    }

    setFeatureTexture(colorResult);
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

void Plotter::showTimestamp() {
    std::string text;
    {
        std::lock_guard guard(mTimestampMutex);
        text = mTimestampText;
    }
    if (text.empty()) return;

    glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
    glDisable(GL_DEPTH_TEST);
    glColor3f(0.f, 0.f, 0.f);
    pangolin::default_font().Text(text).DrawWindow(10, 10);
    glPopAttrib();
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
        pangolin::ProjectionMatrix(752, 480, mCameraIntrinsics(0, 0), mCameraIntrinsics(1, 1), mCameraIntrinsics(0, 2), mCameraIntrinsics(1, 2), 0.01,
                                   100),
        pangolin::ModelViewLookAt(0, 0, 3, 3, 3, 0, pangolin::AxisZ));

    // 2. 3D viewport
    pangolin::Handler3D handler(s_cam, pangolin::AxisZ);
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, 1920.0f / 1080.0f)
            .SetHandler(&handler);


    pangolin::View &featureView = pangolin::CreateDisplay()
            .SetBounds(0.6, 1.0, 0.6, 1.0, 1920.f / 1080)
            .SetLock(pangolin::LockRight, pangolin::LockTop);

    pangolin::View &matcherView = pangolin::CreateDisplay()
            .SetBounds(0.0, 0.6, 0.6, 1.0, 1920.f / (1080 * 2))
            .SetLock(pangolin::LockRight, pangolin::LockBottom);

    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0,
                                            pangolin::Attach::Pix(200));
    pangolin::Var<bool> menu_pause("menu.Pause", false, true);
    const auto viewCamera = [this, &s_cam](const std::shared_ptr<tft::RigidTransform3D>& cameraPose) {
        if (!cameraPose) return;
        const auto worldToCamera = cameraPose->inverse();
        // Camera axes are right/down/forward; OpenGL is right/up/backward.
        const Eigen::Matrix3d cameraToOpenGL = Eigen::Vector3d(1.0, -1.0, -1.0).asDiagonal();
        s_cam.SetModelViewMatrix(GetPangolinModelMatrix(
            cameraToOpenGL * worldToCamera->mRotation,
            cameraToOpenGL * worldToCamera->mTranslation));
    };
    pangolin::Var<std::function<void()>> menu_viewCam0("menu.View cam0", [this, viewCamera] {
        std::shared_ptr<tft::RigidTransform3D> cam0Pose;
        {
            std::lock_guard guard(mFrustumMutex);
            if (mFrustumPose && mFrustumPose->mSource == "cam0") {
                cam0Pose = mFrustumPose;
            }
        }
        viewCamera(cam0Pose);
    });
    pangolin::Var<std::function<void()>> menu_viewPredicted("menu.View predicted", [this, viewCamera] {
        std::shared_ptr<tft::RigidTransform3D> predictedPose;
        {
            std::lock_guard guard(mFrustumMutex);
            predictedPose = mPredictedFrustumPose;
        }
        viewCamera(predictedPose);
    });
    pangolin::Var<bool> menu_showFrames("menu.Show Frames", true, true);
    pangolin::Var<bool> menu_showFrameNames("menu.Show Frame Names", true, true);
    pangolin::Var<bool> menu_showCam0Frustum("menu.cam0 Frustum", true, true);
    pangolin::Var<bool> menu_showPredictedFrustum("menu.Predicted Frustum", true, true);
    pangolin::Var<double> menu_3DImageAlpha("menu.3D Image Alpha", true, 0, 1);
    pangolin::Var<bool> menu_showFeatures("menu.Show Features", true, true);
    pangolin::Var<bool> menu_showMatches("menu.Show Matches", true, true);
    pangolin::Var<bool> menu_showGroundTruthPath("menu.Ground Truth Path", true, true);
    pangolin::Var<bool> menu_showPredictedPath("menu.Predicted Path", true, true);

    auto& timestampView = pangolin::CreateDisplay();
    timestampView.zorder = 1;
    timestampView.SetBounds(0.0, pangolin::Attach::Pix(35), 0.0, 1.0)
        .SetDrawFunction([this](pangolin::View& view) {
            view.Activate();
            showTimestamp();
        });
    pangolin::DisplayBase().ResizeChildren();

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
            for (const auto &p: mCloud) {
                glVertex3f(p.x(), p.y(), p.z());
            }
            glEnd();
        }

        drawCameraPaths(menu_showGroundTruthPath, menu_showPredictedPath);

        drawFrustums(menu_showCam0Frustum, menu_showPredictedFrustum, menu_3DImageAlpha);

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
