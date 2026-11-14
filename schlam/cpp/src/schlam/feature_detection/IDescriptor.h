//
// Created by baldhat on 11/14/26.
//

#ifndef SCHLAM_IDESCRIPTOR_H
#define SCHLAM_IDESCRIPTOR_H
#include <memory>

class IDescriptor;
using DescriptorPtr = std::shared_ptr<IDescriptor>;

class IDescriptor {
public:
    virtual ~IDescriptor() = default;

    [[nodiscard]] virtual double getDistance(const DescriptorPtr& aDescriptor) const = 0;
};

#endif //SCHLAM_IDESCRIPTOR_H
