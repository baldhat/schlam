#include "PangolinPlotter.hpp"

#include <Eigen/src/Geometry/Quaternion.h>
#include <GL/gl.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>

#include <pangolin/display/default_font.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/gl/glpangoglu.h>
#include <pangolin/handler/handler.h>

#include "schlam/KeyPoint.h"
#include "schlam/Map.h"

#include <stdexcept>

namespace {
constexpr int kSidebarWidth = 248;

void setColor(const PangolinPlotter::Color& color) {
    glColor3ub(color.red, color.green, color.blue);
}

cv::Scalar toBgr(const PangolinPlotter::Color& color) {
    return cv::Scalar(color.blue, color.green, color.red);
}

PangolinPlotter::Color blend(const PangolinPlotter::Color& near, const PangolinPlotter::Color& far, float amount) {
    const auto channel = [amount](unsigned char start, unsigned char end) {
        return static_cast<unsigned char>(start + (end - start) * amount);
    };
    return {channel(near.red, far.red), channel(near.green, far.green), channel(near.blue, far.blue)};
}

void drawRect(float left, float bottom, float width, float height, const PangolinPlotter::Color& color) {
    setColor(color);
    glBegin(GL_QUADS);
    glVertex2f(left, bottom);
    glVertex2f(left + width, bottom);
    glVertex2f(left + width, bottom + height);
    glVertex2f(left, bottom + height);
    glEnd();
}

void drawWindowText(const std::string& text, float x, float y, const PangolinPlotter::Color& color) {
    setColor(color);
    pangolin::default_font().Text(text).DrawWindow(x, y);
}

void drawOutlinedWorldText(const std::string& text, const Eigen::Vector3d& position,
                           const PangolinPlotter::Color& color, const PangolinPlotter::Color& outline) {
    GLdouble projection[16], modelview[16], screenX, screenY, screenZ;
    GLint viewport[4];
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    glGetIntegerv(GL_VIEWPORT, viewport);
    if (!pangolin::glProject(position.x(), position.y(), position.z(), modelview, projection,
                             viewport, &screenX, &screenY, &screenZ) || screenZ < 0 || screenZ > 1) return;

    glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT);
    glDisable(GL_DEPTH_TEST);
    const auto label = pangolin::default_font().Text(text);
    setColor(outline);
    for (const auto [dx, dy] : std::array<std::pair<int, int>, 4>{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}}) {
        label.DrawWindow(screenX + dx, screenY + dy);
    }
    setColor(color);
    label.DrawWindow(screenX, screenY);
    glPopAttrib();
}

class Sidebar : public pangolin::Handler {
public:
    enum Toggle { Pause, Frames, FrameNames, Cam0Frustum, PredictedFrustum,
                  KeyframeFrustums, Features, Matches, GroundTruthPath, PredictedPath,
                  OrbitCamera, ToggleCount };
    struct Entry { const char* label; bool enabled; };

    explicit Sidebar(const PangolinPlotter::Theme& theme) : mTheme(theme) {}

    std::array<Entry, ToggleCount> toggles{{
        {"Pause", false}, {"Show frames", true}, {"Frame names", true},
        {"cam0 frustum", true}, {"Predicted frustum", true}, {"Keyframe frustums", true},
        {"Feature image", true}, {"Match image", true},
        {"Ground truth path", true}, {"Predicted path", true}, {"Orbit camera", false}
    }};
    double imageAlpha{0.6};
    double orbitDistance{12.0};
    std::function<void()> viewCam0;
    std::function<void()> viewPredicted;

    bool enabled(Toggle toggle) const { return toggles[toggle].enabled; }

    void Render(pangolin::View& view) const {
        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT |
                     GL_SCISSOR_BIT | GL_VIEWPORT_BIT);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        view.ActivateAndScissor();
        pangolin::DisplayBase().ActivatePixelOrthographic();
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);

        const float left = static_cast<float>(view.v.l);
        const float top = static_cast<float>(view.v.t());
        const float width = static_cast<float>(view.v.w);
        drawRect(left, view.v.b, width, view.v.h, mTheme.sidebarBackground);
        drawRect(left + width - 1, view.v.b, 1, view.v.h, mTheme.sidebarBorder);

        for (int i = 0; i < 2; ++i) {
            const float y = top - kFirstButtonOffset - i * 42 + mScroll;
            drawRect(left + 14, y, width - 28, 34, mTheme.sidebarSurface);
            drawRect(left + 14, y, 3, 34, mTheme.sidebarAccent);
            drawWindowText(i == 0 ? "View cam0" : "View predicted", left + 28, y + 10,
                           mTheme.sidebarText);
        }
        for (int i = 0; i < ToggleCount; ++i) {
            const float y = top - kFirstToggleOffset - i * 36 + mScroll;
            const bool on = toggles[i].enabled;
            drawRect(left + 14, y + 6, 18, 18, on ? mTheme.sidebarAccent : mTheme.sidebarSurface);
            if (!on) {
                drawRect(left + 16, y + 8, 14, 14, mTheme.sidebarBorder);
            } else {
                drawRect(left + 20, y + 12, 6, 6, mTheme.sidebarAccentText);
            }
            drawWindowText(toggles[i].label, left + 42, y + 8, mTheme.sidebarText);
        }
        const float sliderY = top - kSliderOffset + mScroll;
        drawWindowText("IMAGE OPACITY", left + 16, sliderY + 39, mTheme.sidebarMutedText);
        drawRect(left + 16, sliderY + 10, width - 32, 8, mTheme.sidebarSurface);
        drawRect(left + 16, sliderY + 10, static_cast<float>((width - 32) * imageAlpha), 8,
                 mTheme.sidebarAccent);
        drawRect(left + 12 + static_cast<float>((width - 32) * imageAlpha), sliderY + 6,
                 8, 16, mTheme.sidebarText);
        drawWindowText(std::to_string(static_cast<int>(imageAlpha * 100)) + "%", left + 16,
                       sliderY - 15, mTheme.sidebarText);

        const float orbitSliderY = top - kOrbitSliderOffset + mScroll;
        drawWindowText("ORBIT DISTANCE", left + 16, orbitSliderY + 39, mTheme.sidebarMutedText);
        drawRect(left + 16, orbitSliderY + 10, width - 32, 8, mTheme.sidebarSurface);
        const double orbitFraction = (orbitDistance - kMinOrbitDistance) /
                                     (kMaxOrbitDistance - kMinOrbitDistance);
        drawRect(left + 16, orbitSliderY + 10, static_cast<float>((width - 32) * orbitFraction), 8,
                 mTheme.sidebarAccent);
        drawRect(left + 12 + static_cast<float>((width - 32) * orbitFraction), orbitSliderY + 6,
                 8, 16, mTheme.sidebarText);
        std::ostringstream orbitLabel;
        orbitLabel << std::fixed << std::setprecision(1) << orbitDistance;
        drawWindowText(orbitLabel.str(), left + 16, orbitSliderY - 15, mTheme.sidebarText);

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        glPopAttrib();
    }

    void Mouse(pangolin::View& view, pangolin::MouseButton button, int x, int y,
               bool pressed, int) override {
        if (pressed && (button == pangolin::MouseWheelUp || button == pangolin::MouseWheelDown)) {
            const int limit = std::max(0, kContentHeight - view.v.h);
            mScroll = std::clamp(mScroll + (button == pangolin::MouseWheelDown ? 40 : -40), 0, limit);
            return;
        }
        if (button != pangolin::MouseButtonLeft) return;
        if (!pressed) {
            mDraggingSlider = false;
            mDraggingOrbitDistance = false;
            return;
        }
        if (y > view.v.t() || y < view.v.b) return;

        const int top = view.v.t() + mScroll;
        for (int i = 0; i < 2; ++i) {
            const int bottom = top - kFirstButtonOffset - i * 42;
            if (y >= bottom && y <= bottom + 34) {
                toggles[OrbitCamera].enabled = false;
                if (i == 0 && viewCam0) viewCam0();
                if (i == 1 && viewPredicted) viewPredicted();
                return;
            }
        }
        for (int i = 0; i < ToggleCount; ++i) {
            const int bottom = top - kFirstToggleOffset - i * 36;
            if (y >= bottom && y <= bottom + 32) {
                toggles[i].enabled = !toggles[i].enabled;
                return;
            }
        }
        if (y >= top - kSliderOffset && y <= top - kSliderOffset + 26) {
            mDraggingSlider = true;
            setAlpha(view, x);
        } else if (y >= top - kOrbitSliderOffset && y <= top - kOrbitSliderOffset + 26) {
            mDraggingOrbitDistance = true;
            setOrbitDistance(view, x);
        }
    }

    void MouseMotion(pangolin::View& view, int x, int, int) override {
        if (mDraggingSlider) setAlpha(view, x);
        if (mDraggingOrbitDistance) setOrbitDistance(view, x);
    }

private:
    static constexpr int kFirstButtonOffset = 48;
    static constexpr int kFirstToggleOffset = 146;
    static constexpr int kSliderOffset = 572;
    static constexpr int kOrbitSliderOffset = 642;
    static constexpr int kContentHeight = 690;
    static constexpr double kMinOrbitDistance = 0.5;
    static constexpr double kMaxOrbitDistance = 30.0;

    void setAlpha(const pangolin::View& view, int x) {
        imageAlpha = std::clamp((x - view.v.l - 16.0) / (view.v.w - 32.0), 0.0, 1.0);
    }

    void setOrbitDistance(const pangolin::View& view, int x) {
        const double fraction = std::clamp((x - view.v.l - 16.0) / (view.v.w - 32.0), 0.0, 1.0);
        orbitDistance = kMinOrbitDistance + fraction * (kMaxOrbitDistance - kMinOrbitDistance);
    }

    const PangolinPlotter::Theme& mTheme;
    int mScroll{0};
    bool mDraggingSlider{false};
    bool mDraggingOrbitDistance{false};
};
} // namespace

// -----------------------------
// Helper drawing functions
// -----------------------------
void PangolinPlotter::DrawGrid(int size, double step) {
    setColor(mTheme.grid);
    glBegin(GL_LINES);
    for (int i = -size; i <= size; ++i) {
        if (!mZUP) {
            glVertex3f(i * step, 0.f, -size * step);
            glVertex3f(i * step, 0.f, size * step);

            glVertex3f(-size * step, 0.f, i * step);
            glVertex3f(size * step, 0.f, i * step);
        } else {
            glVertex3f(i * step, -size * step, 0.f);
            glVertex3f(i * step, size * step, 0.f);

            glVertex3f(-size * step, i * step, 0.f);
            glVertex3f(size * step, i * step, 0.f);
        }
    }
    glEnd();
}

// -----------------------------
// Plotter methods
// -----------------------------
PangolinPlotter::PangolinPlotter(std::shared_ptr<tft::Transformer> apTransformer, const Eigen::Matrix3d& aCameraIntrinsics)
    : PangolinPlotter(apTransformer, aCameraIntrinsics, Theme{}, 1920 / 1080.0, true) {}

PangolinPlotter::PangolinPlotter(std::shared_ptr<tft::Transformer> apTransformer, const Eigen::Matrix3d& aCameraIntrinsics,
                 const Theme& aTheme, const double aRatio, const bool zUP)
    : mTheme(aTheme)
    , mCameraIntrinsics(aCameraIntrinsics)
    , mpTransformer(apTransformer)
    , mImageAspectRatio(aRatio)
    , mZUP(zUP) {

    pangolin::CreateWindowAndBind("3D Visualizer", 1920, 1080);
    glEnable(GL_MULTISAMPLE);

    // unset the current context from the main thread
    pangolin::GetBoundWindow()->RemoveCurrent();
}

bool PangolinPlotter::shouldPause() {
    std::lock_guard guard(mPauseMutex);
    return mPause;
}

void PangolinPlotter::setMap(const Map* map) {
    const auto mapToWorld = map ? mpTransformer->findTransform(map->mFrame, "world") : nullptr;
    if (map && !mapToWorld) {
        throw std::invalid_argument("Cannot visualize a map without its world transform");
    }
    std::lock_guard guard(mMapMutex);
    mpMap = map;
    mMapToWorld = mapToWorld;
    ++mMapRevision;
}

void PangolinPlotter::addTransform(
    const std::shared_ptr<tft::RigidTransform3D> transform) {
    mTransforms.push_back(transform);
}

void PangolinPlotter::updateFrustum(const std::shared_ptr<ImageData> aImageData,
                           const std::shared_ptr<tft::RigidTransform3D> aPredictedPose) {
    const auto cam0Pose = aImageData
        ? mpTransformer->findTransform(aImageData->mCoordinateFrame, "world") : nullptr;
    std::lock_guard guard(mCameraVisualizationMutex);
    mFrustum = aImageData;
    mFrustumPose = cam0Pose;
    mPredictedFrustumPose = aPredictedPose;
    m3DImageChanged = true;
}

void PangolinPlotter::updateTimestamp(const utils::TTimestamp& aTimestamp) {
    std::lock_guard guard(mTimestampMutex);
    if (!mSequenceStartTimestamp) mSequenceStartTimestamp = aTimestamp;
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        aTimestamp - *mSequenceStartTimestamp).count();
    mTimestampText = std::to_string(milliseconds) + " ms";
}

void PangolinPlotter::recordFrameTime(std::chrono::steady_clock::duration elapsed) {
    const double milliseconds = std::chrono::duration<double, std::milli>(elapsed).count();
    std::lock_guard guard(mFrameTimeMutex);
    mRecentFrameTimesMs.push_back(milliseconds);
    mRecentFrameTimeSumMs += milliseconds;
    if (mRecentFrameTimesMs.size() > 20) {
        mRecentFrameTimeSumMs -= mRecentFrameTimesMs.front();
        mRecentFrameTimesMs.pop_front();
    }
}

void PangolinPlotter::resetCameraPaths(const Eigen::Vector3d& aReferencePosition) {
    std::lock_guard guard(mCameraVisualizationMutex);
    mGroundTruthCameraPath = {aReferencePosition};
    mPredictedCameraPath = {aReferencePosition};
}

void PangolinPlotter::addGroundTruthCameraPosition(const Eigen::Vector3d& aPosition) {
    std::lock_guard guard(mCameraVisualizationMutex);
    mGroundTruthCameraPath.push_back(aPosition);
}

void PangolinPlotter::addPredictedCameraPosition(const Eigen::Vector3d& aPosition) {
    std::lock_guard guard(mCameraVisualizationMutex);
    mPredictedCameraPath.push_back(aPosition);
}

void PangolinPlotter::drawCameraPaths(bool aShowGroundTruth, bool aShowPredicted) {
    glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_LINE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);

    const auto drawPath = [](const std::vector<Eigen::Vector3d>& path,
                             const Color& color, float width,
                             const std::shared_ptr<tft::RigidTransform3D>& currentPose = nullptr) {
        if (path.empty() || (path.size() < 2 && !currentPose)) return;
        setColor(color);
        glLineWidth(width);
        glBegin(GL_LINE_STRIP);
        for (const auto& position : path) {
            glVertex3d(position.x(), position.y(), position.z());
        }
        // The current frustum may show a motion prediction while tracking is
        // still running. Connect to it without committing it to the accepted path.
        if (currentPose) {
            const auto& position = currentPose->mTranslation;
            glVertex3d(position.x(), position.y(), position.z());
        }
        glEnd();
    };

    if (aShowGroundTruth) drawPath(mGroundTruthCameraPath, mTheme.groundTruth, 3.f);
    if (aShowPredicted) drawPath(mPredictedCameraPath, mTheme.predicted, 2.f, mPredictedFrustumPose);
    glPopAttrib();
}

pangolin::OpenGlMatrix
PangolinPlotter::GetPangolinModelMatrix(const Eigen::Matrix3d &R,
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

void PangolinPlotter::plotTransform(
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
    if (showFrameName) {
        drawOutlinedWorldText(transform->mTarget, Eigen::Vector3d::Zero(),
                              mTheme.text, mTheme.textOutline);
    }
    glPopMatrix();
}

void PangolinPlotter::drawAxes(const double radius, const double length) {
    glPushMatrix();

    // X axis
    setColor(mTheme.axisX);
    glPushMatrix();
    glRotatef(90.f, 0.f, 1.f, 0.f); // Z → X
    drawCylinder(radius, length);
    glPopMatrix();

    // Y axis
    setColor(mTheme.axisY);
    glPushMatrix();
    glRotatef(-90.f, 1.f, 0.f, 0.f); // Z → Y
    drawCylinder(radius, length);
    glPopMatrix();

    // Z axis
    setColor(mTheme.axisZ);
    drawCylinder(radius, length);

    glPopMatrix();
}

void PangolinPlotter::plotMatches(const cv::Mat &oldFrame, const cv::Mat &newFrame,
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
    const cv::Scalar markerColor = toBgr(mTheme.matchMarker);

    for (size_t i = 0; i < std::min(features1.size(), features2.size()); ++i) {
        cv::Point2d pt1{static_cast<double>(features1[i]->getImgX()), static_cast<double>(features1[i]->getImgY()+ halfSize) };
        cv::Point2d pt2{static_cast<double>(features2[i]->getImgX()), static_cast<double>(features2[i]->getImgY() + offset - halfSize)};
        const cv::Scalar lineColor = toBgr(mTheme.matchLines[i % mTheme.matchLines.size()]);
        cv::line(colorResult, pt1, pt2, lineColor, 1, cv::LINE_AA);
    }

    for (const auto &kp : features1) {
        cv::Point pt{static_cast<int>(kp->getImgX()), static_cast<int>(kp->getImgY())};
        cv::rectangle(colorResult,
                      {pt.x - halfSize, pt.y - halfSize},
                      {pt.x + halfSize, pt.y + halfSize},
                      markerColor, 1);
    }

    for (const auto &kp : features2) {
        cv::Point pt{static_cast<int>(kp->getImgX()), static_cast<int>(kp->getImgY() + offset)};
        cv::rectangle(colorResult,
                      {pt.x - halfSize, pt.y - halfSize},
                      {pt.x + halfSize, pt.y + halfSize},
                      markerColor, 1);
    }

    setMatcherTexture(colorResult);
}

void PangolinPlotter::drawCylinder(double radius, double length, int slices) {
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

void PangolinPlotter::drawCameras(bool showGroundTruthPath, bool showPredictedPath,
                         bool showCam0Frustum, bool showPredictedFrustum, double alpha) {
    std::lock_guard guard(mCameraVisualizationMutex);
    drawCameraPaths(showGroundTruthPath, showPredictedPath);
    drawFrustums(showCam0Frustum, showPredictedFrustum, alpha);
}

void PangolinPlotter::drawFrustums(bool aShowCam0, bool aShowPredicted, double aAlpha) {
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
        setColor(mTheme.groundTruth);
        glLineWidth(3.f);
        plotFrustum(mFrustum->mImage, mFrustum->mIntrinsics, mFrustumPose, *m3DImageTexture, aAlpha);
    }
    if (aShowPredicted && mPredictedFrustumPose) {
        setColor(mTheme.predicted);
        glLineWidth(2.f);
        plotFrustum(mFrustum->mImage, mFrustum->mIntrinsics, mPredictedFrustumPose, *m3DImageTexture, aAlpha);
    }
    glPopAttrib();
}

void PangolinPlotter::drawMap(bool showKeyFrameFrustums, bool showFrameNames, double alpha) {
    MapSnapshot snapshot;
    std::shared_ptr<tft::RigidTransform3D> mapToWorld;
    std::uint64_t mapRevision;
    {
        std::lock_guard guard(mMapMutex);
        if (mpMap) snapshot = mpMap->getSnapshot();
        mapToWorld = mMapToWorld;
        mapRevision = mMapRevision;
    }
    mDisplayedKeyframeCount = snapshot.keyFrames.size();
    mDisplayedMapPointCount = snapshot.points.size();
    if (mapRevision != mKeyFrameTextureMapRevision) {
        mKeyFrameTextures.clear();
        mKeyFrameTextureMapRevision = mapRevision;
    }
    if (!mapToWorld) return;

    glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT | GL_TEXTURE_BIT | GL_LINE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glPushAttrib(GL_POINT_BIT);
    glPointSize(5.f);
    glBegin(GL_POINTS);
    for (const auto& point : snapshot.points) {
        const Eigen::Vector3d worldPoint = mapToWorld->mRotation * point + mapToWorld->mTranslation;
        const float depth = static_cast<float>(std::clamp(worldPoint.z() / 2.0, 0.0, 1.0));
        setColor(blend(mTheme.mapPointNear, mTheme.mapPointFar, depth));
        glVertex3d(worldPoint.x(), worldPoint.y(), worldPoint.z());
    }
    glEnd();
    glPopAttrib();

    if (!showKeyFrameFrustums) {
        glPopAttrib();
        return;
    }
    glLineWidth(2.5f);
    for (const auto& keyFrame : snapshot.keyFrames) {
        const auto& image = keyFrame.imageData;
        if (image.empty()) continue;
        auto& texture = mKeyFrameTextures[keyFrame.id];
        if (!texture) {
            GLint internalFormat;
            GLenum pixelFormat;
            switch (image.channels()) {
                case 1: internalFormat = GL_LUMINANCE8; pixelFormat = GL_LUMINANCE; break;
                case 3: internalFormat = GL_RGB8; pixelFormat = GL_BGR; break;
                case 4: internalFormat = GL_RGBA8; pixelFormat = GL_BGRA; break;
                default: throw std::invalid_argument("Unsupported keyframe image channel count");
            }
            texture = std::make_unique<pangolin::GlTexture>(
                image.cols, image.rows, internalFormat, true, 0, pixelFormat, GL_UNSIGNED_BYTE);
            const cv::Mat pixels = image.isContinuous() ? image : image.clone();
            glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            texture->Upload(pixels.data, pixelFormat, GL_UNSIGNED_BYTE);
            glPopClientAttrib();
        }
        const auto cameraToWorld = mapToWorld * keyFrame.mapToCamera.inverse();

        glDisable(GL_TEXTURE_2D);
        setColor(mTheme.keyframe);
        plotFrustum(image, keyFrame.intrinsics, cameraToWorld, *texture, alpha);
        if (showFrameNames) {
            const auto& position = cameraToWorld->mTranslation;
            drawOutlinedWorldText("kf " + std::to_string(keyFrame.id), position,
                                  mTheme.keyframeText, mTheme.textOutline);
        }
    }
    glPopAttrib();
}

void PangolinPlotter::plotFrustum(const cv::Mat& image, const Eigen::Matrix3d& intrinsics,
                         const std::shared_ptr<tft::RigidTransform3D>& transform,
                         pangolin::GlTexture& texture, double alpha) {
    glPushMatrix();

    pangolin::OpenGlMatrix Twc = GetPangolinModelMatrix(
        transform->mRotation, transform->mTranslation);
    glMultMatrixd(Twc.m);

    auto kInv = intrinsics.inverse().eval();
    double scale = 0.1f; // The depth at which to draw

    glDepthMask(GL_FALSE);
    glEnable(GL_TEXTURE_2D);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    texture.Bind();

    glPushAttrib(GL_CURRENT_BIT);
    glColor4f(mTheme.textureTint.red / 255.f, mTheme.textureTint.green / 255.f,
              mTheme.textureTint.blue / 255.f, static_cast<float>(alpha));
    glBegin(GL_QUADS);

    // Define the 4 corners in Camera Space using K-inverse
    glTexCoord2f(0, 1);
    Eigen::Vector3d bl = kInv * Eigen::Vector3d(0, image.rows, 1) * scale;
    glVertex3d(bl[0], bl[1], bl[2]);
    glTexCoord2f(1, 1);
    Eigen::Vector3d br = kInv * Eigen::Vector3d(image.cols, image.rows, 1) * scale;
    glVertex3d(br[0], br[1], br[2]);
    glTexCoord2f(1, 0);
    Eigen::Vector3d tr = kInv * Eigen::Vector3d(image.cols, 0, 1) * scale;
    glVertex3d(tr[0], tr[1], tr[2]);
    glTexCoord2f(0, 0);
    Eigen::Vector3d tl = kInv * Eigen::Vector3d(0, 0, 1) * scale;
    glVertex3d(tl[0], tl[1], tl[2]);

    glEnd();

    texture.Unbind();
    glDisable(GL_TEXTURE_2D);
    glPopAttrib();
    glDepthMask(GL_TRUE);
    // The wireframe is drawn after the image so the keyframe accent stays visible.
    pangolin::glDrawFrustum(kInv, image.cols, image.rows, scale);
    glPopMatrix();
}

void PangolinPlotter::plotFeatures(const cv::Mat &aImage, const std::vector<std::shared_ptr<KeyPoint>>& aFeatures) {
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
            cv::circle(colorResult, cv::Point(targetX, targetY), 2, toBgr(mTheme.featureMarker), -1);
        }
    }

    setFeatureTexture(colorResult);
}

void PangolinPlotter::setFeatureTexture(const cv::Mat &aImage) {
    mFeatureImage = aImage;
    mFeatureImageChanged = true;
}

void PangolinPlotter::setMatcherTexture(const cv::Mat &aImage) {
    mMatcherImage = aImage;
    mMatcherImageChanged = true;
}

void PangolinPlotter::showFeatures() {
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

    setColor(mTheme.textureTint);
    mFeatureImageTexture->RenderToViewportFlipY();
}

void PangolinPlotter::showMatches() {
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

    setColor(mTheme.textureTint);
    mMatcherImageTexture->RenderToViewportFlipY();
}

void PangolinPlotter::showStatus() {
    std::string timestamp;
    {
        std::lock_guard guard(mTimestampMutex);
        timestamp = mTimestampText;
    }
    double averageMs = 0.0;
    std::size_t frameCount = 0;
    {
        std::lock_guard guard(mFrameTimeMutex);
        frameCount = mRecentFrameTimesMs.size();
        if (frameCount != 0) averageMs = mRecentFrameTimeSumMs / frameCount;
    }

    std::ostringstream frameTimeText;
    frameTimeText << "Avg frame time (last " << frameCount << "): ";
    if (frameCount != 0) {
        frameTimeText << std::fixed << std::setprecision(1) << averageMs << " ms";
    } else {
        frameTimeText << "-- ms";
    }
    const std::string mapText = "Keyframes: " + std::to_string(mDisplayedKeyframeCount)
        + "  Map points: " + std::to_string(mDisplayedMapPointCount);

    glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_VIEWPORT_BIT);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    pangolin::DisplayBase().ActivatePixelOrthographic();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    drawRect(kSidebarWidth, 0, pangolin::DisplayBase().v.w - kSidebarWidth, 80,
             mTheme.statusBackground);
    setColor(mTheme.statusText);
    if (!timestamp.empty()) pangolin::default_font().Text(timestamp).DrawWindow(kSidebarWidth + 10, 10);
    pangolin::default_font().Text(frameTimeText.str()).DrawWindow(kSidebarWidth + 10, 32);
    pangolin::default_font().Text(mapText).DrawWindow(kSidebarWidth + 10, 54);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPopAttrib();
}


void PangolinPlotter::run() {
    pangolin::BindToContext("3D Visualizer");
    // enable depth
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLint samples = 0;
    glGetIntegerv(GL_SAMPLES, &samples);

    // 1. Camera setup
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(752, 480, mCameraIntrinsics(0, 0), mCameraIntrinsics(1, 1), mCameraIntrinsics(0, 2), mCameraIntrinsics(1, 2), 0.01,
                                   1000),
        pangolin::ModelViewLookAt(0, -150, -100, 0, 0, 50, mZUP ? pangolin::AxisZ : pangolin::AxisNegY));

    // 2. 3D viewport
    pangolin::Handler3D handler(s_cam, mZUP ? pangolin::AxisZ : pangolin::AxisNegY);
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, 1920.0f / 1080.0f)
            .SetHandler(&handler);


    pangolin::View &featureView = pangolin::CreateDisplay()
            .SetBounds(0.6, 1.0, 0.6, 1.0, mImageAspectRatio)
            .SetLock(pangolin::LockRight, pangolin::LockTop);

    pangolin::View &matcherView = pangolin::CreateDisplay()
            .SetBounds(0.0, 0.6, 0.6, 1.0, mImageAspectRatio / 2)
            .SetLock(pangolin::LockRight, pangolin::LockBottom);

    glClearColor(mTheme.background.red / 255.f, mTheme.background.green / 255.f,
                 mTheme.background.blue / 255.f, 1.0f);

    Sidebar sidebar(mTheme);
    const auto viewCamera = [this, &s_cam](const std::shared_ptr<tft::RigidTransform3D>& cameraPose) {
        if (!cameraPose) return;
        const auto worldToCamera = cameraPose->inverse();
        // Camera axes are right/down/forward; OpenGL is right/up/backward.
        const Eigen::Matrix3d cameraToOpenGL = Eigen::Vector3d(1.0, -1.0, -1.0).asDiagonal();
        s_cam.SetModelViewMatrix(GetPangolinModelMatrix(
            cameraToOpenGL * worldToCamera->mRotation,
            cameraToOpenGL * worldToCamera->mTranslation));
    };
    sidebar.viewCam0 = [this, viewCamera] {
        std::shared_ptr<tft::RigidTransform3D> cam0Pose;
        {
            std::lock_guard guard(mCameraVisualizationMutex);
            if (mFrustumPose && mFrustumPose->mSource == "gt/cam0") {
                cam0Pose = mFrustumPose;
            }
        }
        viewCamera(cam0Pose);
    };
    sidebar.viewPredicted = [this, viewCamera] {
        std::shared_ptr<tft::RigidTransform3D> predictedPose;
        {
            std::lock_guard guard(mCameraVisualizationMutex);
            predictedPose = mPredictedFrustumPose;
        }
        viewCamera(predictedPose);
    };
    auto& sidebarView = pangolin::CreateDisplay();
    sidebarView.zorder = 1;
    sidebarView.SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(kSidebarWidth))
        .SetHandler(&sidebar)
        .SetDrawFunction([&sidebar](pangolin::View& view) { sidebar.Render(view); });

    auto& statusView = pangolin::CreateDisplay();
    statusView.zorder = 2;
    statusView.SetBounds(0.0, pangolin::Attach::Pix(80), pangolin::Attach::Pix(kSidebarWidth), 1.0)
        .SetDrawFunction([this](pangolin::View& view) {
            view.Activate();
            showStatus();
        });
    pangolin::DisplayBase().ResizeChildren();

    // 3. Main loop
    double orbitAngle = 0.0;
    auto previousFrameTime = std::chrono::steady_clock::now();
    while (!pangolin::ShouldQuit()) {
        const auto frameTime = std::chrono::steady_clock::now();
        const double elapsedSeconds = std::chrono::duration<double>(frameTime - previousFrameTime).count();
        previousFrameTime = frameTime;
        if (sidebar.enabled(Sidebar::OrbitCamera)) {
            constexpr double kOrbitRadiansPerSecond = 0.15;
            constexpr double kSqrtHalf = 0.7071067811865476;
            orbitAngle = std::remainder(orbitAngle + kOrbitRadiansPerSecond * elapsedSeconds,
                                        2.0 * 3.14159265358979323846);
            const double horizontalRadius = sidebar.orbitDistance * kSqrtHalf;
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(
                horizontalRadius * std::cos(orbitAngle),
                horizontalRadius * std::sin(orbitAngle),
                sidebar.orbitDistance * kSqrtHalf,
                0.0, 0.0, 0.0, pangolin::AxisZ));
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // Draw grid
        glLineWidth(1.0f);
        DrawGrid(100, 1.0f);

        drawMap(sidebar.enabled(Sidebar::KeyframeFrustums), sidebar.enabled(Sidebar::FrameNames),
                sidebar.imageAlpha);

        drawCameras(sidebar.enabled(Sidebar::GroundTruthPath), sidebar.enabled(Sidebar::PredictedPath),
                    sidebar.enabled(Sidebar::Cam0Frustum), sidebar.enabled(Sidebar::PredictedFrustum),
                    sidebar.imageAlpha);

        {
            std::lock_guard guard(mPauseMutex);
            mPause = sidebar.enabled(Sidebar::Pause);
        }

        // Draw all transforms
        if (sidebar.enabled(Sidebar::Frames)) {
            for (auto &transform: mpTransformer->getRootedTransforms()) {
                plotTransform(transform, 0.005, 0.1, sidebar.enabled(Sidebar::FrameNames));
            }
        }

        featureView.Activate();
        if (sidebar.enabled(Sidebar::Features)) {
            showFeatures();
        }

        matcherView.Activate();
        if (sidebar.enabled(Sidebar::Matches)) {
            showMatches();
        }


        pangolin::FinishFrame();
    }
    mKeyFrameTextures.clear();
}
