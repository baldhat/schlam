#include "FFMPEGDataset.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <new>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using SteadyClock = std::chrono::steady_clock;
constexpr auto frameTimeout = std::chrono::seconds(5);

std::string ffmpegError(int status) {
    char message[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(status, message, sizeof(message));
    return message;
}

void checkFFmpeg(int status, const std::string& operation) {
    if (status < 0) {
        throw std::runtime_error(operation + ": " + ffmpegError(status));
    }
}

// FFmpeg frees several of its objects through pointer-to-pointer APIs.
template <typename T, void (*Free)(T**)>
using FFmpegPtr = std::unique_ptr<T, decltype([](T* value) { Free(&value); })>;
}

struct FFMPEGDataset::Stream {
    SteadyClock::time_point deadline;
    FFmpegPtr<AVFormatContext, avformat_close_input> format;
    FFmpegPtr<AVCodecContext, avcodec_free_context> decoder;
    FFmpegPtr<AVPacket, av_packet_free> packet{av_packet_alloc()};
    FFmpegPtr<AVFrame, av_frame_free> frame{av_frame_alloc()};
    std::unique_ptr<SwsContext, decltype(&sws_freeContext)> scaler{nullptr, sws_freeContext};
    int videoIndex = -1;
    const utils::TTimestamp clockOrigin = std::chrono::system_clock::now();
    const SteadyClock::time_point steadyOrigin = SteadyClock::now();
    bool sdp = false;
    bool useMediaTimestamps = false;
    std::chrono::microseconds frameInterval{1};
    std::optional<std::int64_t> firstVideoPts;
    utils::TTimestamp firstVideoTime;
    std::optional<utils::TTimestamp> lastVideoTime;

    explicit Stream(const std::string& source) {
        const auto* protocol = avio_find_protocol_name(source.c_str());
        const bool localFile = protocol && std::string_view(protocol) == "file";
        AVDictionary* options = nullptr;
        FFmpegPtr<AVDictionary, av_dict_free> inputOptions;
        if (localFile) {
            // A local SDP description can open nested RTP and UDP inputs.
            const int status = av_dict_set(&options, "protocol_whitelist", "file,crypto,data,udp,rtp", 0);
            inputOptions.reset(options);
            checkFFmpeg(status, "Cannot configure FFmpeg file protocols");
        }

        auto* input = avformat_alloc_context();
        if (!input || !packet || !frame) {
            avformat_free_context(input);
            throw std::bad_alloc();
        }
        input->interrupt_callback = {[](void* opaque) -> int {
            return SteadyClock::now() >= static_cast<Stream*>(opaque)->deadline;
        }, this};
        deadline = SteadyClock::now() + frameTimeout;
        options = inputOptions.release();
        const int openStatus = avformat_open_input(&input, source.c_str(), nullptr, &options);
        inputOptions.reset(options);
        checkFFmpeg(openStatus, "Cannot open FFmpeg input");
        format.reset(input);

        // SDP provides the RTP payload mapping without waiting for video packets.
        sdp = std::string_view(format->iformat->name) == "sdp";
        if (!sdp) {
            deadline = SteadyClock::now() + frameTimeout;
            checkFFmpeg(avformat_find_stream_info(format.get(), nullptr), "Cannot read FFmpeg stream information");
        }
        const AVCodec* codec = nullptr;
        videoIndex = av_find_best_stream(format.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
        checkFFmpeg(videoIndex, "Cannot select an FFmpeg video stream");
        decoder.reset(avcodec_alloc_context3(codec));
        if (!decoder) {
            throw std::bad_alloc();
        }
        auto* video = format->streams[videoIndex];
        checkFFmpeg(avcodec_parameters_to_context(decoder.get(), video->codecpar),
                    "Cannot initialize FFmpeg decoder parameters");
        decoder->pkt_timebase = video->time_base;
        decoder->thread_count = 1;
        checkFFmpeg(avcodec_open2(decoder.get(), codec, nullptr), "Cannot open FFmpeg video decoder");

        useMediaTimestamps = localFile && !sdp;
        if (useMediaTimestamps) {
            const auto rate = av_guess_frame_rate(format.get(), video, nullptr);
            if (rate.num > 0 && rate.den > 0) {
                frameInterval = std::max(frameInterval, std::chrono::microseconds(
                    av_rescale_q(1, av_inv_q(rate), AVRational{1, AV_TIME_BASE})));
            }
        }
    }

    utils::TTimestamp timestamp() {
        if (!useMediaTimestamps) {
            return clockOrigin + std::chrono::duration_cast<utils::TTimestamp::duration>(
                SteadyClock::now() - steadyOrigin);
        }

        auto time = clockOrigin;
        if (lastVideoTime) {
            time = *lastVideoTime + std::min<utils::TTimestamp::duration>(
                frameInterval, utils::TTimestamp::max() - *lastVideoTime);
        }
        const auto pts = frame->best_effort_timestamp;
        if (pts != AV_NOPTS_VALUE) {
            if (!firstVideoPts) {
                firstVideoPts = pts;
                firstVideoTime = time;
            }
            const auto elapsed = av_rescale_q(av_sat_sub64(pts, *firstVideoPts),
                decoder->pkt_timebase, AVRational{1, AV_TIME_BASE});
            const auto maxElapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                utils::TTimestamp::max() - firstVideoTime).count();
            if (elapsed >= 0 && elapsed <= maxElapsed) {
                const auto mediaTime = firstVideoTime + std::chrono::microseconds(elapsed);
                if (!lastVideoTime || mediaTime > *lastVideoTime) {
                    time = mediaTime;
                }
            }
        }
        lastVideoTime = time;
        return time;
    }

    int readFrame() {
        deadline = SteadyClock::now() + frameTimeout;
        while (SteadyClock::now() < deadline) {
            int status = avcodec_receive_frame(decoder.get(), frame.get());
            if (status != AVERROR(EAGAIN) && status != AVERROR_INVALIDDATA) {
                return status;
            }

            status = av_read_frame(format.get(), packet.get());
            if (status == AVERROR_EOF) {
                status = avcodec_send_packet(decoder.get(), nullptr);
                if (status < 0) {
                    return status;
                }
                return avcodec_receive_frame(decoder.get(), frame.get());
            }
            if (status == AVERROR(EAGAIN)) {
                continue;
            }
            if (status < 0) {
                return SteadyClock::now() >= deadline ? AVERROR(ETIMEDOUT) : status;
            }
            if (packet->stream_index == videoIndex) {
                status = avcodec_send_packet(decoder.get(), packet.get());
            }
            av_packet_unref(packet.get());
            if (status < 0 && status != AVERROR_INVALIDDATA) {
                return status;
            }
        }
        return AVERROR(ETIMEDOUT);
    }

    cv::Mat grayscale() {
        scaler.reset(sws_getCachedContext(scaler.release(),
            frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
            frame->width, frame->height, AV_PIX_FMT_GRAY8, SWS_BILINEAR, nullptr, nullptr, nullptr));
        if (!scaler) {
            return {};
        }

        cv::Mat gray(frame->height, frame->width, CV_8UC1);
        std::uint8_t* planes[4]{gray.data};
        const int strides[4]{static_cast<int>(gray.step[0])};
        if (sws_scale(scaler.get(), frame->data, frame->linesize, 0, frame->height,
                      planes, strides) != frame->height) {
            return {};
        }
        return gray;
    }
};

FFMPEGDataset::FFMPEGDataset(const std::filesystem::path& calibrationFile, const std::string& source) {
    const auto filename = calibrationFile.string();
    try {
        const auto camera = YAML::LoadFile(filename);
        const auto intrinsics = camera["intrinsics"].as<std::vector<double>>();
        const auto resolution = camera["resolution"].as<std::vector<int>>();
        const auto distortion = camera["distortion_coefficients"].as<std::vector<double>>();
        const auto allFinite = [](const std::vector<double>& values) {
            return std::ranges::all_of(values, [](double value) { return std::isfinite(value); });
        };

        if (camera["camera_model"].as<std::string>("pinhole") != "pinhole" ||
            camera["distortion_model"].as<std::string>("rad-tan") != "rad-tan") {
            throw std::invalid_argument("use a pinhole camera with rad-tan distortion");
        }
        if (intrinsics.size() != 4 || !allFinite(intrinsics) || intrinsics[0] <= 0 || intrinsics[1] <= 0) {
            throw std::invalid_argument("intrinsics must be [fx, fy, cx, cy] with positive focal lengths");
        }
        if (resolution.size() != 2 || resolution[0] <= 0 || resolution[1] <= 0) {
            throw std::invalid_argument("resolution must be [width, height] in positive pixels");
        }
        constexpr int maxRemapDimension = 32766;
        if (resolution[0] > maxRemapDimension || resolution[1] > maxRemapDimension) {
            throw std::invalid_argument("resolution exceeds OpenCV's remap size limit");
        }
        const auto count = distortion.size();
        if ((count != 0 && count != 4 && count != 5 && count != 8) || !allFinite(distortion)) {
            throw std::invalid_argument("distortion_coefficients must contain 0, 4, 5, or 8 finite values");
        }

        const auto [fx, fy, cx, cy] = std::array{intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]};
        intrinsics_ << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        imageSize_ = {resolution[0], resolution[1]};
        if (std::ranges::any_of(distortion, [](double value) { return value != 0; })) {
            const cv::Matx33d cameraMatrix(fx, 0, cx, 0, fy, cy, 0, 0, 1);
            cv::initUndistortRectifyMap(cameraMatrix, cv::Mat(distortion), cv::noArray(), cameraMatrix,
                                        imageSize_, CV_16SC2, undistortMap1_, undistortMap2_);
        }
    } catch (const std::exception& error) {
        throw std::runtime_error("Invalid FFmpeg calibration file " + filename + ": " + error.what());
    }
    stream_ = std::make_unique<Stream>(source);
}

FFMPEGDataset::~FFMPEGDataset() = default;

double FFMPEGDataset::getImageAspectRatio() {
    return static_cast<double>(imageSize_.width) / imageSize_.height;
}

std::shared_ptr<ImageData> FFMPEGDataset::getNextImageData() {
    if (!stream_) {
        return nullptr;
    }
    const int status = stream_->readFrame();
    if (status < 0) {
        if (status == AVERROR(ETIMEDOUT)) {
            spdlog::warn("FFmpeg input stopped after {} seconds without a decoded frame", frameTimeout.count());
        } else if (status != AVERROR_EOF) {
            spdlog::error("Cannot read FFmpeg frame: {}", ffmpegError(status));
        }
        stream_.reset();
        return nullptr;
    }
    if (stream_->frame->width != imageSize_.width || stream_->frame->height != imageSize_.height) {
        spdlog::error("FFmpeg frame size {}x{} differs from calibration size {}x{}",
                      stream_->frame->width, stream_->frame->height, imageSize_.width, imageSize_.height);
        stream_.reset();
        return nullptr;
    }

    const auto timestamp = stream_->timestamp();
    auto image = stream_->grayscale();
    if (image.empty()) {
        spdlog::error("Cannot convert FFmpeg frame to grayscale");
        stream_.reset();
        return nullptr;
    }
    if (!undistortMap1_.empty()) {
        cv::Mat undistorted;
        cv::remap(image, undistorted, undistortMap1_, undistortMap2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        image = std::move(undistorted);
    }
    return std::make_shared<ImageData>(timestamp, image, intrinsics_,
                                      stream_->sdp ? "rtp/cam0" : "ffmpeg/cam0");
}
