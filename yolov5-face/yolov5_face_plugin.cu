#include "yolov5_face_plugin.h"
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

namespace nvinfer1
{
    yolov5FacePlugin::yolov5FacePlugin()
    {
        conf_thresh = yolov5FaceConfig::CONF_THRESH;
        refer_rows_1 = 3 * yolov5FaceConfig::INPUT_H * yolov5FaceConfig::INPUT_W / (8.0 * 8.0);
        refer_rows_2 = refer_rows_1 + 3 * yolov5FaceConfig::INPUT_H * yolov5FaceConfig::INPUT_W / (16.0 * 16.0);
        refer_rows_3 = refer_rows_2 + 3 * yolov5FaceConfig::INPUT_H * yolov5FaceConfig::INPUT_W / (32.0 * 32.0);
        std::cout<<"init decode plugin" <<std::endl;
    }

    yolov5FacePlugin::~yolov5FacePlugin()
    {
        std::cout<<"destroy yolov5_face plugin"<<std::endl;
    }

    // create the plugin at runtime from a byte stream
    yolov5FacePlugin::yolov5FacePlugin(const void* data, size_t length)
    {
    }

    void yolov5FacePlugin::serialize(void* buffer) const
    {
    }

    size_t yolov5FacePlugin::getSerializationSize() const
    {
        return 0;
    }

    int yolov5FacePlugin::initialize()
    {
        return 0;
    }

    Dims yolov5FacePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalsize = yolov5FaceConfig::MAX_OUT * sizeof(yolov5FaceConfig::FaceBox) / sizeof(float);
        return Dims3(totalsize + 1, 1, 1);

    }

    // Set plugin namespace
    void yolov5FacePlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* yolov5FacePlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType yolov5FacePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool yolov5FacePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool yolov5FacePlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void yolov5FacePlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void yolov5FacePlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void yolov5FacePlugin::detachFromContext() {}

    const char* yolov5FacePlugin::getPluginType() const
    {
        return "YOLOV5FACE_TRT";
    }

    const char* yolov5FacePlugin::getPluginVersion() const
    {
        return "1";
    }

    void yolov5FacePlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* yolov5FacePlugin::clone() const
    {
        yolov5FacePlugin *p = new yolov5FacePlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data){ return 1./(1. + expf(-data)); }; //sigmoid func
    __device__ float dev_anchors_grid[3][6] = {4,5,  8,10,  13,16, 23,29,  43,55,  73,105, 146,217,  231,300,  335,433};
    __device__ int dev_strides[3] = {8, 16, 32};

    //计算, 输入大小 18900 * 16
    __global__ void CalDetection(const float *input, float *output, int refer_rows_1, int refer_rows_2,
            int refer_rows_3, int num_elem, int output_elem) {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx >= num_elem) return;

            const float* curInput = input + idx * 16; //每行第一个, 16暂时写死
            int bn_idx = idx / refer_rows_3;      //batch_id, total_grid=18900
            idx %= refer_rows_3;                  //offset idx in batch_id

            // 过滤置信度
            float cur_conf = Logist(*(curInput + 4));
            if (cur_conf <= yolov5FaceConfig::CONF_THRESH) //0.3
                return;
            //printf("ori_val: %f, conf: %f, thresh: %f\n", *(curInput + 4), cur_conf, yolov5FaceConfig::CONF_THRESH);

            float *res_count = output + bn_idx * output_elem;
            int count = (int)atomicAdd(res_count, 1); //总数累加
            if (count >= yolov5FaceConfig::MAX_OUT) return;

            //判断位于哪个层
            int layer_idx = 0;
            if (idx > refer_rows_2) {
                layer_idx = 2;
                idx -= refer_rows_2;
            } else if (idx > refer_rows_1) {
                layer_idx = 1;
                idx -= refer_rows_1;
            }

            int cur_stride = dev_strides[layer_idx];
            int cur_w = yolov5FaceConfig::INPUT_W / cur_stride;
            int cur_h = yolov5FaceConfig::INPUT_H / cur_stride;
            int h_idx_all = idx / cur_w; //h方向堆叠的全局idx
            int h_idx = h_idx_all % cur_h; //grid范围内的索引
            int anchor_idx = (h_idx_all / cur_h) % 3; //考虑归属哪一个anchor
            int w_idx = idx % cur_w; //横向对应的索引

            //构造检测框
            char* data = (char *)res_count + sizeof(float) + count * sizeof(yolov5FaceConfig::FaceBox);
            yolov5FaceConfig::FaceBox* det = (yolov5FaceConfig::FaceBox*)(data);

            //xywhs c
            float bw = pow((Logist(*(curInput + 2)) * 2), 2) * dev_anchors_grid[layer_idx][2 * anchor_idx]; //w
            float bh = pow((Logist(*(curInput + 3)) * 2), 2) * dev_anchors_grid[layer_idx][2 * anchor_idx + 1]; //h
            det->bbox[0] = (Logist(*(curInput)) * 2. - 0.5 + w_idx) * cur_stride - bw / 2.0; //x1
            det->bbox[1] = (Logist(*(curInput + 1)) * 2. - 0.5 + h_idx) * cur_stride - bh / 2.0; //y1
            det->bbox[2] = det->bbox[0] + bw; //x2
            det->bbox[3] = det->bbox[1] + bh; //y2
            det->bbox[4] = cur_conf * Logist(*(curInput+ 15)); //score
            det->conf = cur_conf;

            //landmarks x1 y1 -> x5 y5
            det->landmarks[0] = (*(curInput+ 5)) * dev_anchors_grid[layer_idx][2 * anchor_idx] + w_idx * cur_stride;
            det->landmarks[1] = (*(curInput+ 6)) * dev_anchors_grid[layer_idx][2 * anchor_idx + 1] + h_idx * cur_stride;
            det->landmarks[2] = (*(curInput+ 7)) * dev_anchors_grid[layer_idx][2 * anchor_idx] + w_idx * cur_stride;
            det->landmarks[3] = (*(curInput+ 8)) * dev_anchors_grid[layer_idx][2 * anchor_idx + 1] + h_idx * cur_stride;
            det->landmarks[4] = (*(curInput+ 9)) * dev_anchors_grid[layer_idx][2 * anchor_idx] + w_idx * cur_stride;
            det->landmarks[5] = (*(curInput+ 10)) * dev_anchors_grid[layer_idx][2 * anchor_idx + 1] + h_idx * cur_stride;
            det->landmarks[6] = (*(curInput+ 11)) * dev_anchors_grid[layer_idx][2 * anchor_idx] + w_idx * cur_stride;
            det->landmarks[7] = (*(curInput+ 12)) * dev_anchors_grid[layer_idx][2 * anchor_idx + 1] + h_idx * cur_stride;
            det->landmarks[8] = (*(curInput+ 13)) * dev_anchors_grid[layer_idx][2 * anchor_idx] + w_idx * cur_stride;
            det->landmarks[9] = (*(curInput+ 14)) * dev_anchors_grid[layer_idx][2 * anchor_idx + 1] + h_idx * cur_stride;
        }

    void yolov5FacePlugin::forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize)
    {

        int outputElem = 1 + yolov5FaceConfig::MAX_OUT * sizeof(yolov5FaceConfig::FaceBox) / sizeof(float);

        for (int idx = 0; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx * outputElem, 0, sizeof(float))); //set total_num=0
        }

        int total_num_elem = refer_rows_3 * batchSize;
        //std::cout<<"total_num_elem: "<<total_num_elem << "row num: "<<row_num<<" batchsize:" << batchSize <<std::endl;

        CalDetection << < (total_num_elem + thread_count_ - 1) / thread_count_, thread_count_ , 0, stream >> > (inputs[0],
                output, refer_rows_1, refer_rows_2, refer_rows_3, total_num_elem, outputElem);
    }

    int yolov5FacePlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float *)outputs[0], stream, batchSize);
        return 0;
    };

    PluginFieldCollection yolov5FacePluginCreator::mFC{};
    std::vector<PluginField> yolov5FacePluginCreator::mPluginAttributes;

    yolov5FacePluginCreator::yolov5FacePluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* yolov5FacePluginCreator::getPluginName() const
    {
        return "YOLOV5FACE_TRT";
    }

    const char* yolov5FacePluginCreator::getPluginVersion() const
    {
        return "1";
    }

    const PluginFieldCollection* yolov5FacePluginCreator::getFieldNames()
    {
        return &mFC;
    }

    IPluginV2IOExt* yolov5FacePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        yolov5FacePlugin* obj = new yolov5FacePlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* yolov5FacePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call PReluPlugin::destroy()
        yolov5FacePlugin* obj = new yolov5FacePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
