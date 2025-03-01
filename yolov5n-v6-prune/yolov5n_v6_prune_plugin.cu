#include "yolov5n_v6_prune_plugin.h"
#include "stdio.h"
#include <iostream>
#include <cassert>
#include <memory>
#include<math.h>

#ifndef CUDA_CHECK

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif

#include <assert.h>
#include <vector>
#include <iostream>

namespace Tn
{
    template<typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template<typename T>
    void read(const char*& buffer, T& val)
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }
}

namespace nvinfer1
{
    YoloLayerPlugin6::YoloLayerPlugin6(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<YoloKernel>& vYoloKernel)
    {
        mClassCount = classCount;
        mYoloV5NetWidth = netWidth;
        mYoloV5NetHeight = netHeight;
        mMaxOutObject = maxOut;
        mYoloKernel = vYoloKernel;
        mKernelCount = vYoloKernel.size();

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* 3 * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
    }
    YoloLayerPlugin6::~YoloLayerPlugin6()
    {
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaFree(mAnchor[ii]));
        }
        CUDA_CHECK(cudaFreeHost(mAnchor));
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin6::YoloLayerPlugin6(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        read(d, mYoloV5NetWidth);
        read(d, mYoloV5NetHeight);
        read(d, mMaxOutObject);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(mYoloKernel.data(), d, kernelSize);
        d += kernelSize;
        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* 3 * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
        assert(d == a + length);
    }

    void YoloLayerPlugin6::serialize(void* buffer) const TRT_NOEXCEPT
{
    using namespace Tn;
    char* d = static_cast<char*>(buffer), *a = d;
    write(d, mClassCount);
    write(d, mThreadCount);
    write(d, mKernelCount);
    write(d, mYoloV5NetWidth);
    write(d, mYoloV5NetHeight);
    write(d, mMaxOutObject);
    auto kernelSize = mKernelCount * sizeof(YoloKernel);
    memcpy(d, mYoloKernel.data(), kernelSize);
    d += kernelSize;

    assert(d == a + getSerializationSize());
}

size_t YoloLayerPlugin6::getSerializationSize() const TRT_NOEXCEPT
{
return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(YoloKernel) * mYoloKernel.size() + sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject);
}

int YoloLayerPlugin6::initialize() TRT_NOEXCEPT
{
return 0;
}

Dims YoloLayerPlugin6::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT
{
//output the result to channel
int totalsize = mMaxOutObject * sizeof(DetectRes) / sizeof(float);

return Dims3(totalsize + 1, 1, 1);
}

// Set plugin namespace
void YoloLayerPlugin6::setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT
{
mPluginNamespace = pluginNamespace;
}

const char* YoloLayerPlugin6::getPluginNamespace() const TRT_NOEXCEPT
{
return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType YoloLayerPlugin6::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT
{
return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool YoloLayerPlugin6::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT
{
return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool YoloLayerPlugin6::canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT
{
return false;
}

void YoloLayerPlugin6::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT
{
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void YoloLayerPlugin6::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT
{
}

// Detach the plugin object from its execution context.
void YoloLayerPlugin6::detachFromContext() TRT_NOEXCEPT {}

const char* YoloLayerPlugin6::getPluginType() const TRT_NOEXCEPT
{
return "YoloLayer6_TRT";
}

const char* YoloLayerPlugin6::getPluginVersion() const TRT_NOEXCEPT
{
return "1";
}

void YoloLayerPlugin6::destroy() TRT_NOEXCEPT
{
delete this;
}

// Clone the plugin
IPluginV2IOExt* YoloLayerPlugin6::clone() const TRT_NOEXCEPT
{
YoloLayerPlugin6* p = new YoloLayerPlugin6(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, mYoloKernel);
p->setPluginNamespace(mPluginNamespace);
return p;
}

__device__ float Logist6(float data) { return 1.0f / (1.0f + expf(-data)); };

__global__ void CalDetection(const float *input, float *output, int noElements,
                             const int netwidth, const int netheight, int maxoutobject, int yoloWidth, int yoloHeight, const float anchors[3 * 2], int classes, int outputElem)
{

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= noElements) return;

    int total_grid = yoloWidth * yoloHeight;
    int bnIdx = idx / total_grid;
    idx = idx - total_grid * bnIdx;
    int info_len_i = 5 + classes;
    const float* curInput = input + bnIdx * (info_len_i * total_grid * 3);

    for (int k = 0; k < 3; ++k) {
        float box_prob = Logist6(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
        if (box_prob < 0.1) continue;
        int class_id = 0;
        float max_cls_prob = 0.0;
        for (int i = 5; i < info_len_i; ++i) {
            float p = Logist6(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
            if (p > max_cls_prob) {
                max_cls_prob = p;
                class_id = i - 5;
            }
        }
        float *res_count = output + bnIdx * outputElem;
        int count = (int)atomicAdd(res_count, 1);
        if (count >= maxoutobject) return;
        char *data = (char*)res_count + sizeof(float) + count * sizeof(DetectRes);
        DetectRes *det = (DetectRes*)(data);

        int row = idx / yoloWidth;
        int col = idx % yoloWidth;

        //Location
        // pytorch:
        //  y = x[i].sigmoid()
        //  y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
        //  y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
        //  X: (sigmoid(tx) + cx)/FeaturemapW *  netwidth
        det->bbox[0] = (col - 0.5f + 2.0f * Logist6(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
        det->bbox[1] = (row - 0.5f + 2.0f * Logist6(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netheight / yoloHeight;

        // W: (Pw * e^tw) / FeaturemapW * netwidth
        // v5: https://github.com/ultralytics/yolov5/issues/471
        det->bbox[2] = 2.0f * Logist6(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
        det->bbox[2] = det->bbox[2] * det->bbox[2] * anchors[2 * k];
        det->bbox[3] = 2.0f * Logist6(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
        det->bbox[3] = det->bbox[3] * det->bbox[3] * anchors[2 * k + 1];
        det->conf = box_prob * max_cls_prob;
        det->class_id = class_id;
    }
}

void YoloLayerPlugin6::forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize)
{
    int outputElem = 1 + mMaxOutObject * sizeof(DetectRes) / sizeof(float);
    for (int idx = 0; idx < batchSize; ++idx) {
        CUDA_CHECK(cudaMemsetAsync(output + idx * outputElem, 0, sizeof(float), stream));
    }
    int numElem = 0;
    for (unsigned int i = 0; i < mYoloKernel.size(); ++i) {
        const auto& yolo = mYoloKernel[i];
        numElem = yolo.width * yolo.height * batchSize;
        if (numElem < mThreadCount) mThreadCount = numElem;

        //printf("Net: %d  %d vs %d %d\n", mYoloV5NetWidth, mYoloV5NetHeight, yolo.width, yolo.height);
        CalDetection << < (numElem + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream >> >
                                                                                        (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float*)mAnchor[i], mClassCount, outputElem);
    }
}


int YoloLayerPlugin6::enqueue(int batchSize, const void* const* inputs, void* TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT
{
forwardGpu((const float* const*)inputs, (float*)outputs[0], stream, batchSize);
return 0;
}

PluginFieldCollection YoloPluginCreator6::mFC{};
std::vector<PluginField> YoloPluginCreator6::mPluginAttributes;

YoloPluginCreator6::YoloPluginCreator6()
{
    mPluginAttributes.clear();

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* YoloPluginCreator6::getPluginName() const TRT_NOEXCEPT
{
return "YoloLayer6_TRT";
}

const char* YoloPluginCreator6::getPluginVersion() const TRT_NOEXCEPT
{
return "1";
}

const PluginFieldCollection* YoloPluginCreator6::getFieldNames() TRT_NOEXCEPT
{
return &mFC;
}

IPluginV2IOExt* YoloPluginCreator6::createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT
{
    assert(fc->nbFields == 2);
    assert(strcmp(fc->fields[0].name, "netinfo") == 0);
    assert(strcmp(fc->fields[1].name, "kernels") == 0);
    int *p_netinfo = (int*)(fc->fields[0].data);
    int class_count = p_netinfo[0];
    int input_w = p_netinfo[1];
    int input_h = p_netinfo[2];
    int max_output_object_count = p_netinfo[3];
    //printf("netinfo: %d %d %d\n", class_count, input_w, input_h);
    std::vector<YoloKernel> kernels(fc->fields[1].length);
    memcpy(&kernels[0], fc->fields[1].data, kernels.size() * sizeof(YoloKernel));
    YoloLayerPlugin6* obj = new YoloLayerPlugin6(class_count, input_w, input_h, max_output_object_count, kernels);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2IOExt* YoloPluginCreator6::deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT
{
    // This object will be deleted when the network is destroyed, which will
    // call YoloLayerPlugin6::destroy()
    YoloLayerPlugin6* obj = new YoloLayerPlugin6(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
}

