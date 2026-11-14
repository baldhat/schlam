//
// Created by baldhat on 11/14/26.
//

#ifndef SCHLAM_IFEATUREDETECTOR_H
#define SCHLAM_IFEATUREDETECTOR_H

class Frame;

class IFeatureDetector {
public:
    virtual ~IFeatureDetector() = default;

    virtual void detectFeatures(Frame& frame) = 0;
};

#endif //SCHLAM_IFEATUREDETECTOR_H
