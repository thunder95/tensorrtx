#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include "yolov5_face_plugin.h"
#include <stdio.h>
#include "memory.h"
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << " ---> Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)


static const int OUTPUT_SIZE = 1601;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

void debug_print(ITensor* input_tensor){
    std::cout<<"DEBUG: "<<input_tensor->getName()<<": ";
    for (int i=0; i<input_tensor->getDimensions().nbDims; i++) {
        std::cout<<input_tensor->getDimensions().d[i]<<" ";
    }
    std::cout<<std::endl;
}

int getOutSize(ITensor* input_tensor){
    return input_tensor->getDimensions().d[0] * input_tensor->getDimensions().d[1] * input_tensor->getDimensions().d[2];
}

static float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

static bool cmp(yolov5FaceConfig::FaceBox& a, yolov5FaceConfig::FaceBox& b) {
    return a.bbox[4] > b.bbox[4];
}

static void nms(std::vector<yolov5FaceConfig::FaceBox>& res, float *output, float conf_thresh = 0.3, float nms_thresh = 0.5) {
    int det_size = sizeof(yolov5FaceConfig::FaceBox) / sizeof(float);
    std::vector<yolov5FaceConfig::FaceBox> dets;
    for (int i = 0; i < output[0] && i < yolov5FaceConfig::MAX_OUT; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        yolov5FaceConfig::FaceBox det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        dets.push_back(det);
    }

    std::sort(dets.begin(), dets.end(), cmp);
    for (size_t m = 0; m < dets.size(); ++m) {
        auto& item = dets[m];
        res.push_back(item);
        for (size_t n = m + 1; n < dets.size(); ++n) {
            if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                dets.erase(dets.begin()+n);
                --n;
            }
        }
    }
}

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

//basic bocks
ILayer* addSilu(INetworkDefinition *network, ITensor& input) {
    auto sig = network->addActivation(input, ActivationType::kSIGMOID);
    assert(sig);
    auto ew = network->addElementWise(input, *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
//    std::cout << "len " << len << " eps: " << eps << " "<< lname + ".running_var"<< std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnSilu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int num_filters, int k, int s, int p, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv = network->addConvolutionNd(input, num_filters, DimsHW{k, k}, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv);
    conv->setStrideNd(DimsHW{s, s});
    conv->setPaddingNd(DimsHW{p, p});
    conv->setNbGroups(g);
    auto bn = addBatchNorm2d(network, weightMap, *conv->getOutput(0), lname + ".bn", 1e-3);
    ILayer* silu = addSilu(network, *bn->getOutput(0));
    return silu;
}

//stemBock
ILayer* StemBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input){
    ILayer* stem_1 = convBnSilu(network, weightMap, input, 16, 3, 2, 1, 1, "model.0.stem_1");
    ILayer* stem_2a = convBnSilu(network, weightMap, *stem_1->getOutput(0), 8, 1, 1, 0, 1, "model.0.stem_2a");
    ILayer* stem_2b = convBnSilu(network, weightMap, *stem_2a->getOutput(0), 16, 3, 2, 1, 1, "model.0.stem_2b");
    ILayer* stem_2p = network->addPoolingNd(*stem_1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2}); //stride??

    ITensor* inputTensors1[] = {stem_2b->getOutput(0), stem_2p->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors1, 2);
    assert(cat1);

    ILayer* stem_3 = convBnSilu(network, weightMap, *cat1->getOutput(0), 16, 1, 1, 0, 1, "model.0.stem_3");
    return stem_3;
}

//ShuffleV2Block
ILayer* ShuffleV2Block(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inch, int outch, int s) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int branch_features = outch / 2;
    ITensor *x1, *x2i, *x2o;

    //branch 1
    if (s > 1) {
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, inch, DimsHW{3, 3},
                weightMap[lname + "branch1.0.weight"], weightMap[lname + "branch1.2.bias"]);
        assert(conv1);
        conv1->setStrideNd(DimsHW{s, s});
        conv1->setPaddingNd(DimsHW{1, 1});
        conv1->setNbGroups(inch);
        IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "branch1.1", 1e-3);
        IConvolutionLayer* conv2 = network->addConvolutionNd(*bn1->getOutput(0), branch_features, DimsHW{1, 1},
                weightMap[lname + "branch1.2.weight"], weightMap[lname + "branch1.2.bias"]);
        assert(conv2);
        IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "branch1.3", 1e-3);
        ILayer* silu = addSilu(network, *bn2->getOutput(0));
        x1 = silu->getOutput(0);
        x2i = &input;

    } else {

        Dims d = input.getDimensions();
        ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ d.d[0] / 2, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
        ISliceLayer *s2 = network->addSlice(input, Dims3{ d.d[0] / 2, 0, 0 }, Dims3{ d.d[0] / 2, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
        x1 = s1->getOutput(0);
        x2i = s2->getOutput(0);
    }

    //branch 2
    IConvolutionLayer* conv3 = network->addConvolutionNd(*x2i, branch_features, DimsHW{1, 1},
            weightMap[lname + "branch2.0.weight"], weightMap[lname + "branch2.0.bias"]);
    assert(conv3);
    IScaleLayer *bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "branch2.1", 1e-3);
    ILayer* silu3 = addSilu(network, *bn3->getOutput(0));

    IConvolutionLayer* conv4 = network->addConvolutionNd(*silu3->getOutput(0), branch_features, DimsHW{3, 3},
            weightMap[lname + "branch2.3.weight"], weightMap[lname + "branch2.3.bias"]);
    assert(conv4);
    conv4->setStrideNd(DimsHW{s, s});
    conv4->setPaddingNd(DimsHW{1, 1});
    conv4->setNbGroups(branch_features);
    IScaleLayer *bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "branch2.4", 1e-3);
    IConvolutionLayer* conv5 = network->addConvolutionNd(*bn4->getOutput(0), branch_features, DimsHW{1, 1},
            weightMap[lname + "branch2.5.weight"], weightMap[lname + "branch2.5.bias"]);
    assert(conv5);
    IScaleLayer *bn5 = addBatchNorm2d(network, weightMap, *conv5->getOutput(0), lname + "branch2.6", 1e-3);
    ILayer* silu5 = addSilu(network, *bn5->getOutput(0));

    ITensor* inputTensors1[] = {x1, silu5->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors1, 2);
    assert(cat1);

    Dims dims = cat1->getOutput(0)->getDimensions();

    IShuffleLayer *sf1 = network->addShuffle(*cat1->getOutput(0));
    assert(sf1);
    sf1->setReshapeDimensions(Dims4(2, dims.d[0] / 2, dims.d[1], dims.d[2]));
    sf1->setSecondTranspose(Permutation{1, 0, 2, 3});

    IShuffleLayer *sf2 = network->addShuffle(*sf1->getOutput(0));
    assert(sf2);
    sf2->setReshapeDimensions(Dims3(dims.d[0], dims.d[1], dims.d[2]));
    return sf2;
}

ILayer *C3(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int c2, std::string lname) {
    int c_ = (int) (c2 * 0.5);
    auto cv1 = convBnSilu(network, weightMap, input, c_, 1, 1, 0, 1, lname + ".cv1");
    auto cv2 = convBnSilu(network, weightMap, input, c_, 1, 1, 0, 1, lname + ".cv2");

    auto bottle = convBnSilu(network, weightMap, *cv1->getOutput(0), c_, 1, 1, 0, 1, lname + ".m.0.cv1");
    bottle = convBnSilu(network, weightMap, *bottle->getOutput(0), c_, 3, 1, 1, 1, lname + ".m.0.cv2");

    ITensor *inputTensors[] = {bottle->getOutput(0), cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBnSilu(network, weightMap, *cat->getOutput(0), c2, 1, 1, 0, 1, lname + ".cv3");
    return cv3;
}


//UPSAMPLE + CONCCAT_+ C3
ILayer* UpsampleBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,
        ITensor* concat_source, int num_filters, std::string lname, bool do_upsample) {
    //concat
    IConcatenationLayer* cat;
    if (do_upsample) {
        //upsample
        auto upsample = network->addResize(input);
        assert(upsample);
        upsample->setResizeMode(ResizeMode::kNEAREST);
        Dims up_dims  = input.getDimensions();
        up_dims.d[1] *= 2;
        up_dims.d[2] *= 2;
        upsample->setOutputDimensions(up_dims);
        upsample->setAlignCorners(false);

        ITensor* tensors[] = {upsample->getOutput(0), concat_source};
        cat = network->addConcatenation(tensors, 2);

    } else {
        ITensor* tensors[] = {&input, concat_source};
        cat = network->addConcatenation(tensors, 2);
    }

    //c3
    auto c3 = C3(network, weightMap, *cat->getOutput(0), num_filters, lname);
    return c3;
}

//Detect head dimensions
IShuffleLayer* DetectResize(INetworkDefinition *network,  ITensor* input) {
    IShuffleLayer *sf = network->addShuffle(*input); // 48 * 60 * 80
    assert(sf);
    auto dims = input->getDimensions();
    sf->setReshapeDimensions(Dims4(3, 16, dims.d[1], dims.d[2]));  // 3 * 16 * 60 * 80
    sf->setSecondTranspose(Permutation{0, 2, 3, 1});  // 3 * 60 * 80  * 16

    IShuffleLayer *sf2 = network->addShuffle(*sf->getOutput(0));
    assert(sf2);
    sf2->setReshapeDimensions(Dims2(dims.d[1] *dims.d[2] * 3, 16));
    return sf2;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    //todo 支持32倍数的任意图片大小 + 预处理

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, yolov5FaceConfig::INPUT_H, yolov5FaceConfig::INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../yolov5_face.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    ILayer* steam_block = StemBlock(network, weightMap, *data); //p2

    //ShuffleV2Block
    ILayer* ir1 = ShuffleV2Block(network, weightMap, *steam_block->getOutput(0), "model.1.", 16, 64, 2); //p2

    ir1 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.2.0.", 64, 64, 1);
    ir1 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.2.1.", 64, 64, 1);
    ILayer* p3 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.2.2.", 64, 64, 1);//p3

    ir1 = ShuffleV2Block(network, weightMap, *p3->getOutput(0), "model.3.", 64, 128, 2);

    ir1 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.4.0.", 128, 128, 1);
    ir1 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.4.1.", 128, 128, 1);
    ir1 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.4.2.", 128, 128, 1);
    ir1 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.4.3.", 128, 128, 1);
    ir1 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.4.4.", 128, 128, 1);
    ir1 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.4.5.", 128, 128, 1);
    ILayer* p4 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.4.6.", 128, 128, 1);//p4

    ILayer* p5 = ShuffleV2Block(network, weightMap, *p4->getOutput(0), "model.5.", 128, 256, 2);

    ir1 = ShuffleV2Block(network, weightMap, *p5->getOutput(0), "model.6.0.", 256, 256, 1);
    ir1 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.6.1.", 256, 256, 1);
    ir1 = ShuffleV2Block(network, weightMap, *ir1->getOutput(0), "model.6.2.", 256, 256, 1);


    //upsample + concat
    ILayer* up1_conv = convBnSilu(network, weightMap, *ir1->getOutput(0), 64, 1, 1, 0, 1, "model.7");
    ILayer* up1 = UpsampleBlock(network, weightMap, *up1_conv->getOutput(0), p4->getOutput(0), 64, "model.10", true);

    ILayer* up2_conv = convBnSilu(network, weightMap, *up1->getOutput(0), 64,  1, 1, 0, 1, "model.11");
    ILayer* up2 = UpsampleBlock(network, weightMap, *up2_conv->getOutput(0), p3->getOutput(0), 64, "model.14", true); //p3

    ILayer* up3_conv = convBnSilu(network, weightMap, *up2->getOutput(0), 64,  3, 2, 1, 1, "model.15");
    ILayer* up3 = UpsampleBlock(network, weightMap, *up3_conv->getOutput(0), up2_conv->getOutput(0), 64, "model.17", false); //p4

    ILayer* up4_conv = convBnSilu(network, weightMap, *up3->getOutput(0), 64,  3, 2, 1, 1, "model.18");
    ILayer* up4 = UpsampleBlock(network, weightMap, *up4_conv->getOutput(0), up1_conv->getOutput(0), 64, "model.20", false); //p5

    //detect head (48, 60, 80) (48, 30, 40) (48, 15, 20)
    IConvolutionLayer* det1 = network->addConvolutionNd(*up2->getOutput(0), 48, DimsHW{1, 1},
            weightMap["model.21.m.0.weight"], weightMap["model.21.m.0.bias"]); // 48 = (1+5+10) * (anchors_len // 2)

    IConvolutionLayer* det2 = network->addConvolutionNd(*up3->getOutput(0), 48, DimsHW{1, 1},
            weightMap["model.21.m.1.weight"], weightMap["model.21.m.1.bias"]); // 48 = (1+5+10) * (anchors_len // 2)

    IConvolutionLayer* det3 = network->addConvolutionNd(*up4->getOutput(0), 48, DimsHW{1, 1},
            weightMap["model.21.m.2.weight"], weightMap["model.21.m.2.bias"]); // 48 = (1+5+10) * (anchors_len // 2)
    ITensor* det_inputs[] = {
        DetectResize(network, det1->getOutput(0))->getOutput(0),
        DetectResize(network, det2->getOutput(0))->getOutput(0),
        DetectResize(network, det3->getOutput(0))->getOutput(0)
    };

    IConcatenationLayer* detect_head = network->addConcatenation(det_inputs, 3);
//    debug_print(detect_head->getOutput(0));

//    auto xx = DetectResize(network, det1->getOutput(0));
    //decode
    auto creator = getPluginRegistry()->getPluginCreator("YOLOV5FACE_TRT", "1");
    IPluginV2 *plugin_obj = creator->createPlugin("yoloface", NULL);
    ITensor* decode_input = detect_head->getOutput(0);
    auto decode = network->addPluginV2(&decode_input, 1, *plugin_obj);

    decode->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out up1" << std::endl;
    network->markOutput(*decode->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{

    cudaError_t error = cudaGetLastError();
    printf("CUDA error111: %s\n", cudaGetErrorString(error));
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * yolov5FaceConfig::INPUT_H * yolov5FaceConfig::INPUT_W * sizeof(float)));
    error = cudaGetLastError();
    printf("CUDA error222: %s\n", cudaGetErrorString(error));

    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
    error = cudaGetLastError();
    printf("CUDA error333: %s\n", cudaGetErrorString(error));


    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * yolov5FaceConfig::INPUT_H * yolov5FaceConfig::INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    error = cudaGetLastError();
    printf("CUDA error444: %s\n", cudaGetErrorString(error));

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5_face -s   // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5_face -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("yolov5_face.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
//        return 1;

        std::ifstream file("yolov5_face.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }

    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("yolov5_face.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }

    //测试输入或输出
//    std::map<std::string, Weights> input_wts = loadWeights("/d/hulei/pd_match/yolov5-face/input_img.wts");
//    float *var = (float*)input_wts["img"].values;
//    std::cout<<"img floats num: "<<input_wts["img"].count;
//
////    float sum = 0.0;
////    std::cout << "\ninput:\n\n";
////    for (unsigned int i = 0; i < 192; i++) {
////        std::cout<<prob[i*30*40]<<", ";
////    }
//
//    static float data[3 * 480 * 640];
//    for (int i = 0; i < 3 * 480 * 640; i++) {
////        data[i] = 1.0;
//        data[i] = var[i];
//    }
//
//
//    IRuntime* runtime = createInferRuntime(gLogger);
//    assert(runtime != nullptr);
//    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
//    assert(engine != nullptr);
//    IExecutionContext* context = engine->createExecutionContext();
//    assert(context != nullptr);
//    delete[] trtModelStream;
//
//    // Run inference
//    static float prob[OUTPUT_SIZE];
//    for (int i = 0; i < 10; i++) {
//        auto start = std::chrono::system_clock::now();
//        doInference(*context, data, prob, 1);
//        auto end = std::chrono::system_clock::now();
//        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
//        break;
//    }
//
//    // Destroy the engine
//    context->destroy();
//    engine->destroy();
//    runtime->destroy();
//
//    // Print histogram of the output distribution
//    std::cout << "\nOutput:\n\n";
//    std::cout << "\ninput:\n\n";
//    std::cout<<"real_output_size: "<<real_output_size<<std::endl;
//    for (unsigned int i = 0; i < 32; i++) {
//        if (i % 16 == 0)
//            std::cout<<std::endl;
//        std::cout<<prob[i]<<", ";
//    }
////    std::cout<<std::endl;
////    for (unsigned int i = real_output_size-20; i < real_output_size; i++) {
////        std::cout<<prob[i]<<", ";
////    }
////    std::cout<<std::endl;
//
//    std::vector<yolov5FaceConfig::FaceBox> fboxes;
//    nms(fboxes, prob);
//    printf("output size: %f, boxes: %d\n", prob[0], fboxes.size());
//    for (int i = 0; i < fboxes.size(); i++) {
//        printf("boxes: %f %f %f %f, score: %f\n", fboxes[i].bbox[0], fboxes[i].bbox[1], fboxes[i].bbox[2],
//                fboxes[i].bbox[3], fboxes[i].bbox[4]);
//    }
//
//
////    std::map<std::string, Weights> weightMap = loadWeights("../yolov5_face.wts");
////    int len = weightMap["model.10.cv1.conv.weight"].count;
////    float *var = (float*)weightMap["model.10.cv1.conv.weight"].values;
////    float sum = 0.0;
////    std::cout << "\ninput:\n\n";
////    for (unsigned int i = 0; i < 192; i++) {
////        std::cout<<prob[i*30*40]<<", ";
////    }
////    std::cout<<std::endl;
////
////    std::cout << "\nweight:\n\n";
////    for (unsigned int i = 0; i < 192; i++) {
////        std::cout<<var[i]<<", ";
////    }
////    std::cout<<std::endl;
//
//
////    for (unsigned int i = 0; i < 192; i++) {
////        sum += prob[i*30*40] * var[i];
////    }
////    std::cout<<"sum: "<<sum<<std::endl;
//
//
////    for (unsigned int i = 0; i < 1000; i++)
////    {
////        std::cout << prob[i] << ", ";
////        if (i % 10 == 0) std::cout << i / 10 << std::endl;
////    }
////    for (unsigned int i = 0; i < 12000; i++)
////    {
////
////        if (i % 1200 == 0) std::cout << prob[i] << ", ";
////    }
//    std::cout << std::endl;

    //测试opencv预处理
    float *data;
    int in_h = yolov5FaceConfig::INPUT_H;
    cudaMalloc<float>(&data, 2000 * 2000 * 3 * sizeof(float));
    cv::Mat img = cv::imread("../people.jpg");
    int w, h, x, y;
    float r_w = 640 / (img.cols * 1.0);
    float r_h = in_h / (img.rows * 1.0);
    float r_b;
    if (r_h > r_w) {
        w = 640;
        h = r_w * img.rows;
        x = 0;
        y = (in_h - h) / 2;
        r_b = r_w;
    } else {
        w = r_h * img.cols;
        h = in_h;
        x = (640 - w) / 2;
        y = 0;
        r_b = r_h;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(in_h, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    //split channels
    out.convertTo(out, CV_32FC3, 1. / 255.);
    cv::Mat input_channels[3];
    cv::split(out, input_channels);
    for (int j = 0; j < 3; j++) {
        cudaMemcpyAsync(data + 640 * in_h * j,
                        input_channels[2 - j].data, 640 * in_h * sizeof(float), cudaMemcpyHostToDevice);
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    static float prob[OUTPUT_SIZE];
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1000; i++) {
        doInference(*context, data, prob, 1);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    std::vector<yolov5FaceConfig::FaceBox> fboxes;
    nms(fboxes, prob);
    printf("output size: %f, boxes: %d, img_w: %d, img_h: %d\n", prob[0], fboxes.size(), img.cols, img.rows);
    for (int i = 0; i < fboxes.size(); i++) {
        printf("boxes: %f %f %f %f, score: %f\n", fboxes[i].bbox[0], fboxes[i].bbox[1], fboxes[i].bbox[2],
                fboxes[i].bbox[3], fboxes[i].bbox[4]);

        for (int k = 0; k<4; k++) {

            if (k % 2 == 0)
                fboxes[i].bbox[k] -= x;
            else
                fboxes[i].bbox[k] -= y;
            fboxes[i].bbox[k] /= r_b;
            printf("box:%d ->  %f\n", k, fboxes[i].bbox[k]);
        }

        for (int k = 0; k<10; k++) {
            if (k % 2 == 0)
                fboxes[i].landmarks[k] -= x;
            else
                fboxes[i].landmarks[k] -= y;
            fboxes[i].landmarks[k] /= r_b;
            printf("landmarks:%d ->  %f\n", k, fboxes[i].landmarks[k]);
        }

        //画框
        cv::rectangle(img, cv::Point(fboxes[i].bbox[0], fboxes[i].bbox[1]),
                      cv::Point(fboxes[i].bbox[2], fboxes[i].bbox[3]), cv::Scalar(0, 0, 255), 3);
        cv::circle(img, cv::Point(fboxes[i].landmarks[0], fboxes[i].landmarks[1]), 2,
                   cv::Scalar(255, 0, 0), 2);
        cv::circle(img, cv::Point(fboxes[i].landmarks[2], fboxes[i].landmarks[3]), 2,
                   cv::Scalar(255, 0, 0), 2);
        cv::circle(img, cv::Point(fboxes[i].landmarks[4], fboxes[i].landmarks[5]), 2,
                   cv::Scalar(255, 0, 0), 2);
        cv::circle(img, cv::Point(fboxes[i].landmarks[6], fboxes[i].landmarks[7]), 2,
                   cv::Scalar(255, 0, 0), 2);
        cv::circle(img, cv::Point(fboxes[i].landmarks[8], fboxes[i].landmarks[9]), 2,
                   cv::Scalar(255, 0, 0), 2);

    }

    cv::imshow("a", img);
    auto key = cv::waitKey(0);
    return 0;
}
