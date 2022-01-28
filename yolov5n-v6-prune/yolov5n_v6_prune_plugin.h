#ifndef _YOLO_LAYER6_H
#define _YOLO_LAYER6_H

#include <vector>
#include <string>
#include <NvInfer.h>
#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "macros.h"

struct WeightsData {
    nvinfer1::Weights wts;
    unsigned int dim1;
    unsigned int dim2;
};

struct YoloKernel
{
    int width;
    int height;
    float anchors[3*2];
};

struct alignas(float) DetectRes {
    //center_x center_y w h
    float bbox[4];
    float conf;  // bbox_conf * cls_conf
    float class_id;
};


namespace nvinfer1
{
    class API YoloLayerPlugin6 : public IPluginV2IOExt
{
    public:
    YoloLayerPlugin6(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<YoloKernel>& vYoloKernel);
    YoloLayerPlugin6(const void* data, size_t length);
    ~YoloLayerPlugin6();

    int getNbOutputs() const TRT_NOEXCEPT override
            {
                    return 1;
            }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT override;

    int initialize() TRT_NOEXCEPT override;

    virtual void terminate() TRT_NOEXCEPT override {};

    virtual size_t getWorkspaceSize(int maxBatchSize) const TRT_NOEXCEPT override { return 0; }

    virtual int enqueue(int batchSize, const void* const* inputs, void*TRT_CONST_ENQUEUE* outputs, void* workspace, cudaStream_t stream) TRT_NOEXCEPT override;

    virtual size_t getSerializationSize() const TRT_NOEXCEPT override;

    virtual void serialize(void* buffer) const TRT_NOEXCEPT override;

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const TRT_NOEXCEPT override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
    }

    const char* getPluginType() const TRT_NOEXCEPT override;

    const char* getPluginVersion() const TRT_NOEXCEPT override;

    void destroy() TRT_NOEXCEPT override;

    IPluginV2IOExt* clone() const TRT_NOEXCEPT override;

    void setPluginNamespace(const char* pluginNamespace) TRT_NOEXCEPT override;

    const char* getPluginNamespace() const TRT_NOEXCEPT override;

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const TRT_NOEXCEPT override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const TRT_NOEXCEPT override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const TRT_NOEXCEPT override;

    void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) TRT_NOEXCEPT override;

    void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) TRT_NOEXCEPT override;

    void detachFromContext() TRT_NOEXCEPT override;

    private:
    void forwardGpu(const float* const* inputs, float *output, cudaStream_t stream, int batchSize = 1);
    int mThreadCount = 256;
    const char* mPluginNamespace;
    int mKernelCount;
    int mClassCount;
    int mYoloV5NetWidth;
    int mYoloV5NetHeight;
    int mMaxOutObject;
    std::vector<YoloKernel> mYoloKernel;
    void** mAnchor;
};

class API YoloPluginCreator6 : public IPluginCreator
{
public:
YoloPluginCreator6();

~YoloPluginCreator6() override = default;

const char* getPluginName() const TRT_NOEXCEPT override;

const char* getPluginVersion() const TRT_NOEXCEPT override;

const PluginFieldCollection* getFieldNames() TRT_NOEXCEPT override;

IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) TRT_NOEXCEPT override;

IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) TRT_NOEXCEPT override;

void setPluginNamespace(const char* libNamespace) TRT_NOEXCEPT override
        {
                mNamespace = libNamespace;
        }

const char* getPluginNamespace() const TRT_NOEXCEPT override
        {
                return mNamespace.c_str();
        }

private:
std::string mNamespace;
static PluginFieldCollection mFC;
static std::vector<PluginField> mPluginAttributes;
};
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator6);
};

#endif  // _YOLO_LAYER_H
