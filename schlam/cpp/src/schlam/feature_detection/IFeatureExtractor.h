//
// Created by baldhat on 11/14/26.
//

#ifndef SCHLAM_IFEATUREDETECTOR_H
#define SCHLAM_IFEATUREDETECTOR_H

class Frame;

class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;

    virtual void extractFeatures(Frame& frame) = 0;
};

#endif //SCHLAM_IFEATUREDETECTOR_H
