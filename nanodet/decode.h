#ifndef _DECODE_CU_H
#define _DECODE_CU_H

#include <string>
#include <vector>
#include "NvInfer.h"

namespace decodeplugin
{
    struct alignas(float) Detection{
        float bbox[5];  //x1 y1 x2 y2 s
        float conf;
        float class_id;
    };
    static const int INPUT_H = 320;
    static const int INPUT_W = 320;
    static const int MAX_OUT = 300;
    //static const int STRIDE_STEPS[3] = {8, 16, 32};
    static const float CONF_THRESH = 0.35;
    static const float NMS_THRESH = 0.6;
    static const int CLASS_NUM = 1;
    static const int REG_MAX = 7;
}

namespace nvinfer1
{
    class DecodePlugin: public IPluginV2IOExt
    {
        public:
            DecodePlugin();
            DecodePlugin(const void* data, size_t length);

            ~DecodePlugin();

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
            int refer_rows = 2100;
            int block_grid_size = (refer_rows + thread_count_ - 1) / thread_count_; //2100=refer_rows 1600+400+100
            float conf_thresh = 0;
            int row_num = 0;
            int loc_len = 0;
            const char* mPluginNamespace;
    };

    class DecodePluginCreator : public IPluginCreator
    {
        public:
            DecodePluginCreator();

            ~DecodePluginCreator() override = default;

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
    REGISTER_TENSORRT_PLUGIN(DecodePluginCreator);
};

#endif 
