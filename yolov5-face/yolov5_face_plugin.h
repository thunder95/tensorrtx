#ifndef _DECODE_CU_H
#define _DECODE_CU_H

#include <string>
#include <vector>
#include "NvInfer.h"

namespace yolov5FaceConfig
{
    struct alignas(float) FaceBox{
        float bbox[5];  //x1 y1 x2 y2 s
        float landmarks[10];
        float conf;
    };
    static const int INPUT_H = 384;
    static const int INPUT_W = 640;
    static const int MAX_OUT = 100;

    static const float CONF_THRESH = 0.3;
    static const float NMS_THRESH = 0.5;
}

namespace nvinfer1
{
    class yolov5FacePlugin: public IPluginV2IOExt
    {
        public:
            yolov5FacePlugin();
            yolov5FacePlugin(const void* data, size_t length);

            ~yolov5FacePlugin();

            int getNbOutputs() const override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

            int initialize() override;

            virtual void terminate() override {};

            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

            virtual size_t getSerializationSize() const override;

            virtual void serialize(void* buffer) const override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const override;

            const char* getPluginVersion() const override;

            void destroy() override;

            IPluginV2IOExt* clone() const override;

            void setPluginNamespace(const char* pluginNamespace) override;

            const char* getPluginNamespace() const override;

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const override;

            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

            void detachFromContext() override;

            int input_size_;
        private:
            void forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize = 1);
            int thread_count_ = 256;
            int refer_rows_1 = 0; //(4800 + 1200 + 300)*3
            int refer_rows_2 = 0;
            int refer_rows_3 = 0;
            int block_grid_size = (refer_rows_3 + thread_count_ - 1) / thread_count_;
            float conf_thresh = 0;
            const char* mPluginNamespace;

            float* refer_matrix   = NULL; //参考矩阵
    };

    class yolov5FacePluginCreator : public IPluginCreator
    {
        public:
            yolov5FacePluginCreator();

            ~yolov5FacePluginCreator() override = default;

            const char* getPluginName() const override;

            const char* getPluginVersion() const override;

            const PluginFieldCollection* getFieldNames() override;

            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

            void setPluginNamespace(const char* libNamespace) override
            {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const override
            {
                return mNamespace.c_str();
            }

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(yolov5FacePluginCreator);
};

extern "C"  void yolov5FacePreprocess(const unsigned char*src,int srcWidth,int srcHeight,int srcPitch, float* dst,int dstWidth,
                        int dstHeight, int write_x, int write_y, float resize_w, float resize_h, cudaStream_t stream);

#endif 
