#include "decode.h"
#include "stdio.h"
#include <iostream>
#include <cassert>

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

namespace nvinfer1
{
    DecodePlugin::DecodePlugin()
    {
        conf_thresh = -log(1 / decodeplugin::CONF_THRESH - 1);
        loc_len = decodeplugin::REG_MAX + 1;
        row_num = decodeplugin::CLASS_NUM + loc_len * 4;

        //std::cout<<"conf_thresh: "<<conf_thresh<<std::endl;
    }

    DecodePlugin::~DecodePlugin()
    {
    }

    // create the plugin at runtime from a byte stream
    DecodePlugin::DecodePlugin(const void* data, size_t length)
    {
    }

    void DecodePlugin::serialize(void* buffer) const
    {
    }

    size_t DecodePlugin::getSerializationSize() const
    {  
        return 0;
    }

    int DecodePlugin::initialize()
    { 
        return 0;
    }

    Dims DecodePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalsize = decodeplugin::MAX_OUT * sizeof(decodeplugin::Detection) / sizeof(float);
        return Dims3(totalsize + 1, 1, 1);

    }

    // Set plugin namespace
    void DecodePlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* DecodePlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType DecodePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool DecodePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool DecodePlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void DecodePlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void DecodePlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void DecodePlugin::detachFromContext() {}

    const char* DecodePlugin::getPluginType() const
    {
        return "NANODET_TRT";
    }

    const char* DecodePlugin::getPluginVersion() const
    {
        return "1";
    }

    void DecodePlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* DecodePlugin::clone() const
    {
        DecodePlugin *p = new DecodePlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data){ return 1./(1. + expf(-data)); }; //sigmoid func

    __global__ void CalDetection(const float *input, float *output, int total_grid, int row_num, int num_elem,
        int output_elem, const int loc_len, const float obj_thresh) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        const float* curInput = input + idx * row_num; //输入的行
        int bn_idx = idx / total_grid;      //batch_id
        idx %= total_grid;                  //offset idx in batch_id

        int stride = 8;
        int offset = 0;
        if (idx >= 1600) {
            stride = (idx >=2000) ? 32 : 16;
            offset = stride == 16 ? 1600: 2000;
        }

        int max_classes = 0;
        float max_class_confidence = -1;
        for (int j = 0; j < decodeplugin::CLASS_NUM; ++j, ++curInput) {
            if (*curInput > max_class_confidence) {
                max_classes = j;
                max_class_confidence = *curInput;
            }
        }

        if (max_class_confidence < obj_thresh) //obj_thresh, decodeplugin::CONF_THRESH
            return;

        //printf("conf: %f, thresh: %f\n", max_class_confidence, decodeplugin::CONF_THRESH);
        float *res_count = output + bn_idx * output_elem;
        int count = (int)atomicAdd(res_count, 1);
        if (count >= decodeplugin::MAX_OUT) return;

        //todo construct detection
        int width =  decodeplugin::INPUT_W / stride;
        int cell_idx = idx - offset;
        int celly = cell_idx / width;
        int cellx = cell_idx % width;
        float ct_x = (cellx + 0.5) * stride;
        float ct_y = (celly + 0.5) * stride;

        float* dis_pred = new float[4];
        for (int i = 0; i < 4; i++) {
            const float* ptr = curInput + i * loc_len;
            const float* ptrmax = ptr;
            float alpha = *ptrmax;
            for (int j = 1; j < loc_len; ++j, ++ptrmax) {
                if (*ptrmax > alpha) {
                    alpha = *ptrmax;
                }
            } //计算最大值

            float denominator = 0;
            float dis = 0;
            for (int j = 0; j < loc_len; ++j) {
                float tmp_dis = exp(ptr[j] - alpha);
                denominator += tmp_dis;
                dis += j * tmp_dis;
            } //softmax分母

            dis /= denominator;
            dis *= stride;
            dis_pred[i] = dis;
        }

        char* data = (char *)res_count + sizeof(float) + count * sizeof(decodeplugin::Detection);
        decodeplugin::Detection* det = (decodeplugin::Detection*)(data);

        det->bbox[0] = (ct_x - dis_pred[0]); //x1
        det->bbox[1] = (ct_y - dis_pred[1]); //y1
        det->bbox[2] = (ct_x + dis_pred[2]); //x2
        det->bbox[3] = (ct_y + dis_pred[3]); //y2
        det->bbox[4] = (dis_pred[2] + dis_pred[0]) * (dis_pred[3] + dis_pred[1]); //s
        delete[] dis_pred;
        det->class_id = max_classes;
        det->conf = Logist(max_class_confidence);
    }

    void DecodePlugin::forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize)
    {

        int outputElem = 1 + decodeplugin::MAX_OUT * sizeof(decodeplugin::Detection) / sizeof(float);
        for (int idx = 0; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx * outputElem, 0, sizeof(float))); //set total_num=0
        }
        int total_num_elem = refer_rows * batchSize;
        std::cout<<"total_num_elem: "<<total_num_elem<<std::endl;

        CalDetection << < (total_num_elem + thread_count_ - 1) / thread_count_, thread_count_ , 0, stream >> > (inputs[0],
            output, refer_rows, row_num, total_num_elem, outputElem, loc_len, conf_thresh);
    }

    int DecodePlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float *)outputs[0], stream, batchSize);
        return 0;
    };

    PluginFieldCollection DecodePluginCreator::mFC{};
    std::vector<PluginField> DecodePluginCreator::mPluginAttributes;

    DecodePluginCreator::DecodePluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* DecodePluginCreator::getPluginName() const
    {
        return "NANODET_TRT";
    }

    const char* DecodePluginCreator::getPluginVersion() const
    {
        return "1";
    }

    const PluginFieldCollection* DecodePluginCreator::getFieldNames()
    {
        return &mFC;
    }

    IPluginV2IOExt* DecodePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        DecodePlugin* obj = new DecodePlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* DecodePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call PReluPlugin::destroy()
        DecodePlugin* obj = new DecodePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
