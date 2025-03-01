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
#include <numeric>
#include "opencv2/opencv.hpp"
#include "decode.h"

#define  CLASSES_NUM 1

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 320;
static const int INPUT_W = 320;
static const int reg_max_ = 7;
static const int GPU_OUTPUT_SIZE = decodeplugin::MAX_OUT * sizeof(decodeplugin::Detection) / sizeof(float) + 1;

const char* INPUT_BLOB_NAME = "data"; //data input.1
const std::string OUTPUT_BLOB_NAME[]={"cls_big","dst_big","cls_mid","dst_mid","cls_tiny","dist_tiny"};
//const std::string OUTPUT_BLOB_NAME[]={"789","790","804","805","819","820"};
const int out_size[] = {1600*CLASSES_NUM, 1600*32, 400*CLASSES_NUM, 400*32, 100*CLASSES_NUM, 100*32};
//const std::string OUTPUT_BLOB_NAME[]={"stage2_out","stage3_out","stage4_out"};
//const int out_size[] = {96*40*40,  232*20*20, 464*10*10};

using namespace nvinfer1;

static Logger gLogger;

//dims5
class Dims5 : public Dims
{
public:
    Dims5()
    {
        nbDims = 5;
        d[0] = d[1] = d[2] = d[3] = d[4] = 0;
    }

    Dims5(int d0, int d1, int d2, int d3, int d4)
    {
        nbDims = 5;
        d[0] = d0;
        d[1] = d1;
        d[2] = d2;
        d[3] = d3;
        d[4] = d4;
    }
};


//加载权重
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
    std::cout<<"count: "<<count<<std::endl;
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

//添加BN层:
/* 权重文件输出示例：
 *  keys:  backbone.stage2.0.branch1.3.weight , val shape: torch.Size([58])
    keys:  backbone.stage2.0.branch1.3.bias , val shape: torch.Size([58])
    keys:  backbone.stage2.0.branch1.3.running_mean , val shape: torch.Size([58])
    keys:  backbone.stage2.0.branch1.3.running_var , val shape: torch.Size([58])
 * */
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    //std::cout<<"layer: "<<lname<<std::endl;
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

//    printf("debug gamma: %s, len: %d, ", (lname + ".weight").c_str(), len);
//    for (int k=0; k<weightMap[lname + ".weight"].count; k++) {
//        printf("%f ",  gamma[k]);
//    }
//    printf("\n");
//
//    printf("debug bias: %s, ", (lname + ".bias").c_str());
//    for (int k=0; k<weightMap[lname + ".bias"].count; k++) {
//        printf("%f ",  beta[k]);
//    }
//    printf("\n");
//
//    printf("debug mean: %s, ", (lname + ".running_mean").c_str());
//    for (int k=0; k<weightMap[lname + ".running_mean"].count; k++) {
//        printf("%f ",  mean[k]);
//    }
//    printf("\n");
//
//    printf("debug var: %s, ", (lname + ".running_var").c_str());
//    for (int k=0; k<weightMap[lname + ".running_var"].count; k++) {
//        printf("%f ",  var[k]);
//    }
//    printf("\n");


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
    //std::cout<<scale_1->getOutput(0)->getName()<<std::endl;
    assert(scale_1);
    return scale_1;
}


void debug_print(ITensor* input_tensor){
    std::cout<<"DEBUG: "<<input_tensor->getName()<<": ";
    for (int i=0; i<input_tensor->getDimensions().nbDims; i++) {
        std::cout<<input_tensor->getDimensions().d[i]<<" ";
    }
    std::cout<<std::endl;
}


//conv+bn+leary_relu lname:head.cls_convs.0
ILayer* head_cls_block(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int idx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    std::string cls_lname = "head.cls_convs." + std::to_string(idx);

    std::string lname = cls_lname + ".0"; //stack 0
    //depthwise
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, 96, DimsHW{3, 3}, weightMap[lname + ".depthwise.weight"], emptywts);
    conv1->setStrideNd(DimsHW{1, 1});
    conv1->setPaddingNd(DimsHW{1, 1});
    conv1->setNbGroups(96);
    assert(conv1);

    IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".dwnorm", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    relu1->setAlpha(0.1);
    assert(relu1);

    //pointwise
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), 96, DimsHW{1, 1}, weightMap[lname + ".pointwise.weight"], emptywts); //todo
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setPaddingNd(DimsHW{0, 0});
    assert(conv2);

    IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".pwnorm", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
    relu2->setAlpha(0.1);
    assert(relu2);

    lname = cls_lname + ".1"; //stack 1
    //depthwise
    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), 96, DimsHW{3, 3}, weightMap[lname + ".depthwise.weight"], emptywts);
    conv3->setStrideNd(DimsHW{1, 1});
    conv3->setPaddingNd(DimsHW{1, 1});
    conv3->setNbGroups(96);
    assert(conv3);

    IScaleLayer *bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".dwnorm", 1e-5);
    IActivationLayer* relu3 = network->addActivation(*bn3->getOutput(0), ActivationType::kLEAKY_RELU);
    relu3->setAlpha(0.1);
    assert(relu3);

    //pointwise
    IConvolutionLayer* conv4 = network->addConvolutionNd(*relu3->getOutput(0), 96, DimsHW{1, 1}, weightMap[lname + ".pointwise.weight"], emptywts); //todo
    conv4->setStrideNd(DimsHW{1, 1});
    conv4->setPaddingNd(DimsHW{0, 0});
    assert(conv4);

    IScaleLayer *bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + ".pwnorm", 1e-5);
    IActivationLayer* relu4 = network->addActivation(*bn4->getOutput(0), ActivationType::kLEAKY_RELU);
    relu4->setAlpha(0.1);
    assert(relu4);

    //gfl
    lname = "head.gfl_cls." + std::to_string(idx);
    int output_num = CLASSES_NUM + 32; // 4 * (regmax + 1)
    IConvolutionLayer* conv5 = network->addConvolutionNd(*relu4->getOutput(0), output_num, DimsHW{1, 1}, weightMap[lname + ".weight"], weightMap[lname + ".bias"]);
    conv5->setStrideNd(DimsHW{1, 1});
    conv5->setPaddingNd(DimsHW{0, 0});

    return conv5;
}

//head cls,  lname:head.cls_convs.0
void  head_module(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int idx) {
    std::cout<<idx<<std::endl;
    debug_print(&input);
    //cls
    auto x = head_cls_block(network, weightMap, input, idx);
    debug_print(x->getOutput(0));

    IShuffleLayer *head_shuffle = network->addShuffle(*x->getOutput(0));
    assert(head_shuffle);
    int output_num = CLASSES_NUM + 32; // 4 * (regmax + 1)
    head_shuffle->setReshapeDimensions(Dims2(output_num, -1));
    head_shuffle->setSecondTranspose(Permutation{1, 0});
    debug_print(head_shuffle->getOutput(0));
    //return head_shuffle;

    //split
    Dims d = x->getOutput(0)->getDimensions();
    ISliceLayer *score = network->addSlice(*x->getOutput(0), Dims3{ 0, 0, 0 }, Dims3{ CLASSES_NUM, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
    ISliceLayer *bbox = network->addSlice(*x->getOutput(0), Dims3{ CLASSES_NUM, 0, 0 }, Dims3{32, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
    debug_print(score->getOutput(0));
    debug_print(bbox->getOutput(0));

    //score
    IActivationLayer* score_sigmoid = network->addActivation(*score->getOutput(0), ActivationType::kSIGMOID);
    debug_print(score_sigmoid->getOutput(0));
    IShuffleLayer *score_shuffle = network->addShuffle(*score_sigmoid->getOutput(0));
    assert(score_shuffle);
    score_shuffle->setReshapeDimensions(Dims3(1, CLASSES_NUM, -1));
    score_shuffle->setSecondTranspose(Permutation{0, 2, 1});
    score_shuffle->getOutput(0)->setName(OUTPUT_BLOB_NAME[2*idx].c_str());
    network->markOutput(*score_shuffle->getOutput(0));
//    debug_print(score_shuffle->getOutput(0));

    //bbox
    IShuffleLayer *bbox_shuffle = network->addShuffle(*bbox->getOutput(0));
    assert(bbox_shuffle);
    bbox_shuffle->setReshapeDimensions(Dims3(1, 32, -1));
    bbox_shuffle->setSecondTranspose(Permutation{0, 2, 1});
    bbox_shuffle->getOutput(0)->setName(OUTPUT_BLOB_NAME[2*idx+1].c_str());
    network->markOutput(*bbox_shuffle->getOutput(0));
    debug_print(bbox_shuffle->getOutput(0));

    //checked




    //to del: for concat, failed to create linear
//    Weights emptywts{DataType::kFLOAT, nullptr, 0};
//    auto soft_max = network->addSoftMax(*bbox_shuffle->getOutput(0));
//    soft_max->setAxes(2);
//    debug_print(soft_max->getOutput(0));
//    auto linear = network->addFullyConnected(*soft_max->getOutput(0), 1, weightMap["head.linear.weight"], emptywts);
//    linear->setKernelWeights(weightMap["head.linear.weight"]);
//    linear->setNbOutputChannels(1);
//    debug_print(linear->getOutput(0));
//    return x;
}



/*
 * shufflenet的叠加模块invertedRes, 有两种变体:
 *  1. split+右侧分支
 *  2. 对输入进行左右分支处理
 *
 *  deprecated
 * */
ILayer* invertedRes_dynamic(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inch, int outch, int s) {
    std::cout<<"layer: "<<lname<<std::endl;
    debug_print(&input);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int branch_features = outch; //nanodet里似乎不需要除2
    ITensor *x1, *x2i, *x2o;
    if (s > 1) {
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, inch, DimsHW{3, 3}, weightMap[lname + "branch1.0.weight"], emptywts);
        assert(conv1);
        conv1->setStrideNd(DimsHW{s, s});
        conv1->setPaddingNd(DimsHW{1, 1});
        conv1->setNbGroups(inch);
        IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "branch1.1", 1e-5);
        IConvolutionLayer* conv2 = network->addConvolutionNd(*bn1->getOutput(0), branch_features, DimsHW{1, 1}, weightMap[lname + "branch1.2.weight"], emptywts);
        assert(conv2);
        IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "branch1.3", 1e-5);
        IActivationLayer* relu1 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
        assert(relu1);
        x1 = relu1->getOutput(0);
        x2i = &input;
    } else {
        Dims d = input.getDimensions();
        ISliceLayer *s1 = network->addSlice(input, Dims4{d.d[0], 0, 0, 0 }, Dims4{ d.d[0], d.d[1] / 2, d.d[2], d.d[3] }, Dims4{1, 1, 1, 1 });
        ISliceLayer *s2 = network->addSlice(input, Dims4{d.d[0], d.d[1] / 2, 0, 0 }, Dims4{d.d[0],  d.d[1] / 2, d.d[2], d.d[3] }, Dims4{1, 1, 1, 1 });

//        ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{  d.d[1] / 2, d.d[2], d.d[3] }, Dims3{1, 1, 1 });
//        ISliceLayer *s2 = network->addSlice(input, Dims3{d.d[1] / 2, 0, 0 }, Dims3{ d.d[1] / 2, d.d[2], d.d[3] }, Dims3{1, 1, 1 });

//        ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ d.d[0] / 2, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
//        ISliceLayer *s2 = network->addSlice(input, Dims3{ d.d[0] / 2, 0, 0 }, Dims3{ d.d[0] / 2, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
        x1 = s1->getOutput(0);
        debug_print(x1);
        x2i = s2->getOutput(0);
        debug_print(x2i);
    }

    IConvolutionLayer* conv3 = network->addConvolutionNd(*x2i, branch_features, DimsHW{1, 1}, weightMap[lname + "branch2.0.weight"], emptywts);
    debug_print(conv3->getOutput(0));
    assert(conv3);
    IScaleLayer *bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "branch2.1", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn3->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu2);
    IConvolutionLayer* conv4 = network->addConvolutionNd(*relu2->getOutput(0), branch_features, DimsHW{3, 3}, weightMap[lname + "branch2.3.weight"], emptywts);
    assert(conv4);
    conv4->setStrideNd(DimsHW{s, s});
    conv4->setPaddingNd(DimsHW{1, 1});
    conv4->setNbGroups(branch_features);
    //std::cout<<weightMap[lname + "branch2.3.weight"].count<<std::endl;
    IScaleLayer *bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "branch2.4", 1e-5);
    IConvolutionLayer* conv5 = network->addConvolutionNd(*bn4->getOutput(0), branch_features, DimsHW{1, 1}, weightMap[lname + "branch2.5.weight"], emptywts);

    debug_print(conv5->getOutput(0));
    //std::cout<<weightMap[lname + "branch2.5.weight"].count<<std::endl;
    assert(conv5);
    IScaleLayer *bn5 = addBatchNorm2d(network, weightMap, *conv5->getOutput(0), lname + "branch2.6", 1e-5);
    IActivationLayer* relu3 = network->addActivation(*bn5->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu3);

    ITensor* inputTensors1[] = {x1, relu3->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors1, 2);
    assert(cat1);
    debug_print(cat1->getOutput(0));
    Dims dims = cat1->getOutput(0)->getDimensions();
//    std::cout << cat1->getOutput(0)->getName() << " dims: ";
//    for (int i = 0; i < dims.nbDims; i++) {
//        std::cout << dims.d[i] << ", ";
//    }
//    std::cout << std::endl;

    //reshape
    IShuffleLayer *sf1 = network->addShuffle(*cat1->getOutput(0));
    assert(sf1);

    sf1->setReshapeDimensions(Dims5(-1, 2, dims.d[1] / 2, dims.d[2], dims.d[3]));
    debug_print(sf1->getOutput(0));
    //transpose
    sf1->setSecondTranspose(Permutation{0, 2, 1, 3, 4});
    debug_print(sf1->getOutput(0));
//    Dims dims1 = sf1->getOutput(0)->getDimensions();
//    std::cout << sf1->getOutput(0)->getName() << " dims: ";
//    for (int i = 0; i < dims1.nbDims; i++) {
//        std::cout << dims1.d[i] << ", ";
//    }
//    std::cout << std::endl;

    //reshape2
    IShuffleLayer *sf2 = network->addShuffle(*sf1->getOutput(0));
    assert(sf2);
    debug_print(sf2->getOutput(0));
    sf2->setReshapeDimensions(Dims4(dims.d[0], dims.d[1], dims.d[2], dims.d[3]));
    debug_print(sf2->getOutput(0));
//    Dims dims2 = sf2->getOutput(0)->getDimensions();
//    std::cout << sf2->getOutput(0)->getName() << " dims: ";
//    for (int i = 0; i < dims2.nbDims; i++) {
//        std::cout << dims2.d[i] << ", ";
//    }
//    std::cout << std::endl;

    return sf2;
}

ILayer* invertedRes(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, int inch, int outch, int s) {
    std::cout<<"layer: "<<lname<<std::endl;
    debug_print(&input);

    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int branch_features = outch; //nanodet里似乎不需要除2
    ITensor *x1, *x2i, *x2o;
    if (s > 1) {
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, inch, DimsHW{3, 3}, weightMap[lname + "branch1.0.weight"], emptywts);
        conv1->setStrideNd(DimsHW{2, 2});
        conv1->setPaddingNd(DimsHW{1, 1});
        conv1->setNbGroups(inch);

        IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "branch1.1", 1e-5);

        IConvolutionLayer* conv2 = network->addConvolutionNd(*bn1->getOutput(0), branch_features, DimsHW{1, 1},
                weightMap[lname + "branch1.2.weight"], emptywts);
        conv2->setStrideNd(DimsHW{1, 1});
        conv2->setPaddingNd(DimsHW{0, 0});
        conv2->setNbGroups(1);
        assert(conv2);

        IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "branch1.3", 1e-5);
        IActivationLayer* relu1 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
        relu1->setAlpha(0.1);
        assert(relu1);
        x1 = relu1->getOutput(0);
        x2i = &input;

    } else {
        Dims d = input.getDimensions();
        ISliceLayer *s1 = network->addSlice(input, Dims3{ 0, 0, 0 }, Dims3{ d.d[0] / 2, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
        ISliceLayer *s2 = network->addSlice(input, Dims3{ d.d[0] / 2, 0, 0 }, Dims3{ d.d[0] / 2, d.d[1], d.d[2] }, Dims3{ 1, 1, 1 });
        x1 = s1->getOutput(0);
        //debug_print(x1);
        x2i = s2->getOutput(0);
        //debug_print(x2i);
    }

    IConvolutionLayer* conv3 = network->addConvolutionNd(*x2i, branch_features, DimsHW{1, 1}, weightMap[lname + "branch2.0.weight"], emptywts);
//    debug_print(conv3->getOutput(0));
    assert(conv3);
    IScaleLayer *bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "branch2.1", 1e-5);
    IActivationLayer* relu2 = network->addActivation(*bn3->getOutput(0), ActivationType::kLEAKY_RELU);
    relu2->setAlpha(0.1);
    assert(relu2);

    IConvolutionLayer* conv4 = network->addConvolutionNd(*relu2->getOutput(0), branch_features, DimsHW{3, 3}, weightMap[lname + "branch2.3.weight"], emptywts);
    assert(conv4);
    conv4->setStrideNd(DimsHW{s, s});
    conv4->setPaddingNd(DimsHW{1, 1});
    conv4->setNbGroups(branch_features);

    IScaleLayer *bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "branch2.4", 1e-5);

    IConvolutionLayer* conv5 = network->addConvolutionNd(*bn4->getOutput(0), branch_features, DimsHW{1, 1}, weightMap[lname + "branch2.5.weight"], emptywts);
    assert(conv5);
    IScaleLayer *bn5 = addBatchNorm2d(network, weightMap, *conv5->getOutput(0), lname + "branch2.6", 1e-5);
    IActivationLayer* relu3 = network->addActivation(*bn5->getOutput(0), ActivationType::kLEAKY_RELU);
    relu3->setAlpha(0.1);
    assert(relu3);
    debug_print(relu3->getOutput(0));
    ITensor* inputTensors1[] = {x1, relu3->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(inputTensors1, 2);
    assert(cat1);

    debug_print(cat1->getOutput(0));
    Dims dims = cat1->getOutput(0)->getDimensions(); //dim:

    //reshape
    IShuffleLayer *sf1 = network->addShuffle(*cat1->getOutput(0));
    assert(sf1);

    sf1->setReshapeDimensions(Dims4(2, dims.d[0] / 2, dims.d[1], dims.d[2]));
    debug_print(sf1->getOutput(0));
    //transpose
    sf1->setSecondTranspose(Permutation{1, 0, 2, 3});
    debug_print(sf1->getOutput(0));
    //reshape2
    IShuffleLayer *sf2 = network->addShuffle(*sf1->getOutput(0));
    assert(sf2);
    //debug_print(sf2->getOutput(0));
    sf2->setReshapeDimensions(Dims3(dims.d[0], dims.d[1], dims.d[2]));
    //debug_print(sf2->getOutput(0));
    return sf2;
}


// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{

    INetworkDefinition* network = builder->createNetworkV2(0U);
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
//    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 116, 40, 40 });
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../jyj_320.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

//    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//    INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
//    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{-1, 3, INPUT_H, INPUT_W});
//    assert(data);

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 24, DimsHW{3, 3}, weightMap["backbone.conv1.0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.conv1.1", 1e-5);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU); //这里变了
    relu1->setAlpha(0.1);
    assert(relu1);

    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1}); //maxpool
    debug_print(pool1->getOutput(0));


    //stage2
    ILayer* ir1 = invertedRes(network, weightMap, *pool1->getOutput(0), "backbone.stage2.0.", 24, 58, 2);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage2.1.", 58, 58, 1);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage2.2.", 58, 58, 1);
    ILayer* stage2_out = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage2.3.", 58, 58, 1); //checked

//    Dims dims = data->getDimensions();
//    IShuffleLayer *sf1 = network->addShuffle(*cat1->getOutput(0));
//    sf1->setReshapeDimensions(Dims4(2, dims.d[1] / 2, dims.d[2], dims.d[3]));
//    sf1->setSecondTranspose(Permutation{1, 0, 2, 3});
//    IShuffleLayer *sf2 = network->addShuffle(*sf1->getOutput(0));
//    sf2->setReshapeDimensions(Dims3(dims.d[0], dims.d[1], dims.d[2]));

    //stage3
    ir1 = invertedRes(network, weightMap, *stage2_out->getOutput(0), "backbone.stage3.0.", 116, 116, 2);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage3.1.", 116, 116, 1);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage3.2.", 116, 116, 1);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage3.3.", 116, 116, 1);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage3.4.", 116, 116, 1);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage3.5.", 116, 116, 1);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage3.6.", 116, 116, 1);
    ILayer* stage3_out = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage3.7.", 116, 116, 1);

    //stage4
    ir1 = invertedRes(network, weightMap, *stage3_out->getOutput(0), "backbone.stage4.0.", 232, 232, 2);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage4.1.", 232, 232, 1);
    ir1 = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage4.2.", 232, 232, 1);
    ILayer* stage4_out = invertedRes(network, weightMap, *ir1->getOutput(0), "backbone.stage4.3.", 232, 232, 1); //checked



    //check_step_1: shufflenet_v2 done

    // ------------- FPN ---------------
    /*
     * 带bias的卷积
     *  keys:  fpn.lateral_convs.0.conv.weight , val shape: torch.Size([96, 116, 1, 1])
        keys:  fpn.lateral_convs.0.conv.bias
     * */
    IConvolutionLayer* fpn_conv1 = network->addConvolutionNd(*stage2_out->getOutput(0), 96, DimsHW{1, 1},
            weightMap["fpn.lateral_convs.0.conv.weight"], weightMap["fpn.lateral_convs.0.conv.bias"]);
    assert(fpn_conv1);
    debug_print(fpn_conv1->getOutput(0));
    IConvolutionLayer* fpn_conv2 = network->addConvolutionNd(*stage3_out->getOutput(0), 96, DimsHW{1, 1},
            weightMap["fpn.lateral_convs.1.conv.weight"], weightMap["fpn.lateral_convs.1.conv.bias"]);
    assert(fpn_conv2);
    debug_print(fpn_conv2->getOutput(0));
    IConvolutionLayer* fpn_conv3 = network->addConvolutionNd(*stage4_out->getOutput(0), 96, DimsHW{1, 1},
            weightMap["fpn.lateral_convs.2.conv.weight"], weightMap["fpn.lateral_convs.2.conv.bias"]);
    assert(fpn_conv3);
    debug_print(fpn_conv3->getOutput(0));



    //top-down: 3_resize+2, 2_resize+1
    IResizeLayer *fpn_conv3_resize = network->addResize(*fpn_conv3->getOutput(0)); //resize fpn3 to fpn2
    fpn_conv3_resize->setOutputDimensions(fpn_conv2->getOutput(0)->getDimensions());
    fpn_conv3_resize->setResizeMode(ResizeMode::kLINEAR );
    fpn_conv3_resize->setAlignCorners(false);
    assert(fpn_conv3_resize);
    debug_print(fpn_conv3_resize->getOutput(0));

    IElementWiseLayer *add_fpn32 = network->addElementWise(*fpn_conv2->getOutput(0), *fpn_conv3_resize->getOutput(0),
            ElementWiseOperation::kSUM); //merged fpn2
    debug_print(add_fpn32->getOutput(0));



    IResizeLayer *fpn_conv2_resize = network->addResize(*add_fpn32->getOutput(0)); //resize merged fpn2 to fpn1
    fpn_conv2_resize->setOutputDimensions(fpn_conv1->getOutput(0)->getDimensions());
    fpn_conv2_resize->setResizeMode(ResizeMode::kLINEAR);
    fpn_conv2_resize->setAlignCorners(false);
    assert(fpn_conv2_resize);
    debug_print(fpn_conv2_resize->getOutput(0));

    IElementWiseLayer *add_fpn21 = network->addElementWise(*fpn_conv1->getOutput(0), *fpn_conv2_resize->getOutput(0),
                                                           ElementWiseOperation::kSUM); //merged fpn1

    debug_print(add_fpn21->getOutput(0)); //checked align_corner=false

    //bottom-top: merged_fpn1, merged_fpn1 + merged_fpn2 (new_merged_fpn2), new_merged_fpn2 + fpn_conv3 (merged_fpn3)
    IResizeLayer *fpn_conv2_2_resize = network->addResize(*add_fpn21->getOutput(0)); //resize merged_fpn1 to merged_fpn2
    fpn_conv2_2_resize->setOutputDimensions(add_fpn32->getOutput(0)->getDimensions());
    fpn_conv2_2_resize->setResizeMode(ResizeMode::kLINEAR );
    fpn_conv2_2_resize->setAlignCorners(false);
    assert(fpn_conv2_2_resize);
    debug_print(fpn_conv2_2_resize->getOutput(0));
    IElementWiseLayer *add_fpn212 = network->addElementWise(*add_fpn32->getOutput(0), *fpn_conv2_2_resize->getOutput(0),
                                                           ElementWiseOperation::kSUM); //new merged fpn2
    debug_print(add_fpn212->getOutput(0));


    IResizeLayer *fpn_conv3_2_resize = network->addResize(*add_fpn212->getOutput(0)); //new_merged_fpn2 to fpn3
    fpn_conv3_2_resize->setOutputDimensions(fpn_conv3->getOutput(0)->getDimensions());
    fpn_conv3_2_resize->setResizeMode(ResizeMode::kLINEAR );
    fpn_conv3_2_resize->setAlignCorners(false);
    assert(fpn_conv3_2_resize);
    debug_print(fpn_conv3_2_resize->getOutput(0));
    IElementWiseLayer *add_fpn312 = network->addElementWise(*fpn_conv3->getOutput(0), *fpn_conv3_2_resize->getOutput(0),
                                                            ElementWiseOperation::kSUM); //new merged fpn3
    debug_print(add_fpn312->getOutput(0)); //checked perfect

    //fpn output:add_fpn21, add_fpn212, add_fpn312

    //checking......0
//    add_fpn21->getOutput(0)->setName("stage2_out");
//    network->markOutput(*add_fpn21->getOutput(0));
//    debug_print(add_fpn21->getOutput(0));



    //cls head:
//    auto x = head_cls_block(network, weightMap, *add_fpn21->getOutput(0), "head.cls_convs.0");
//    debug_print(x->getOutput(0));
    head_module(network, weightMap, *add_fpn21->getOutput(0), 0);
    head_module(network, weightMap, *add_fpn212->getOutput(0), 1);
    head_module(network, weightMap, *add_fpn312->getOutput(0), 2);
//    ITensor* catTensors[] = {cat1->getOutput(0), cat2->getOutput(0), cat3->getOutput(0)};
//    IConcatenationLayer* cat = network->addConcatenation(catTensors, 3);
//    debug_print(cat->getOutput(0));



//    IOptimizationProfile *profile = builder->createOptimizationProfile();
//    config->addOptimizationProfile(profile);


//    auto creator = getPluginRegistry()->getPluginCreator("NANODET_TRT", "1");
//    PluginFieldCollection pfc;
//    IPluginV2 *pluginObj = creator->createPlugin("nanodet", &pfc);
//
//    ITensor* inputTensors[] = {cat->getOutput(0)};
//    auto decodelayer = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
//    assert(decodelayer);
//
//    decodelayer->getOutput(0)->setName(OUTPUT_BLOB_NAME[0].c_str());
//    network->markOutput(*decodelayer->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));
//    config->setFlag(BuilderFlag::kFP16); //fp16

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

void doInferenceCPU(IExecutionContext& context, float* input, float* out1, float* out2, float* out3, float* out4, float* out5,  float* out6, int batchSize)
//void doInference(IExecutionContext& context, float* input, float* out1,  int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    printf("input data: ");
    for (int j=0; j<80; j++)
        printf("%f ", input[j]);
    printf("\n");

    std::cout<<"-===> "<<engine.getNbBindings()<<std::endl;
    for (int i=0; i< engine.getNbBindings(); i++) {
        std::cout<<engine.getBindingName(i)<<std::endl;

        nvinfer1::Dims dims = engine.getBindingDimensions(i);
        nvinfer1::DataType dtype = engine.getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        std::cout<<engine.getBindingName(i) << totalSize <<std::endl;
    }




    assert(engine.getNbBindings() == 7);
    void* buffers[7];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int out_idx_1 = engine.getBindingIndex(OUTPUT_BLOB_NAME[0].c_str());
    const int out_idx_2 = engine.getBindingIndex(OUTPUT_BLOB_NAME[1].c_str());
    const int out_idx_3 = engine.getBindingIndex(OUTPUT_BLOB_NAME[2].c_str());
    const int out_idx_4 = engine.getBindingIndex(OUTPUT_BLOB_NAME[3].c_str());
    const int out_idx_5 = engine.getBindingIndex(OUTPUT_BLOB_NAME[4].c_str());
    const int out_idx_6 = engine.getBindingIndex(OUTPUT_BLOB_NAME[5].c_str());

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3*320*320 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[out_idx_1], batchSize * out_size[0] * sizeof(float)));
    CHECK(cudaMalloc(&buffers[out_idx_2], batchSize * out_size[1] * sizeof(float)));
    CHECK(cudaMalloc(&buffers[out_idx_3], batchSize * out_size[2] * sizeof(float)));
    CHECK(cudaMalloc(&buffers[out_idx_4], batchSize * out_size[3] * sizeof(float)));
    CHECK(cudaMalloc(&buffers[out_idx_5], batchSize * out_size[4] * sizeof(float)));
    CHECK(cudaMalloc(&buffers[out_idx_6], batchSize * out_size[5] * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize *  3*320*320  * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(out1, buffers[out_idx_1], batchSize * out_size[0] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(out2, buffers[out_idx_2], batchSize * out_size[1] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(out3, buffers[out_idx_3], batchSize * out_size[2] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(out4, buffers[out_idx_4], batchSize * out_size[3] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(out5, buffers[out_idx_5], batchSize * out_size[4] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(out6, buffers[out_idx_6], batchSize * out_size[5] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[out_idx_1]));
    CHECK(cudaFree(buffers[out_idx_2]));
    CHECK(cudaFree(buffers[out_idx_3]));
    CHECK(cudaFree(buffers[out_idx_4]));
    CHECK(cudaFree(buffers[out_idx_5]));
    CHECK(cudaFree(buffers[out_idx_6]));
}

void doInferenceGPU(IExecutionContext& context, float* input, float* out1, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    printf("input data: ");
    for (int j=0; j<80; j++)
        printf("%f ", input[j]);
    printf("\n");

    std::cout<<"-===> "<<engine.getNbBindings()<<std::endl;
    for (int i=0; i< engine.getNbBindings(); i++) {
        std::cout<<engine.getBindingName(i)<<std::endl;

        nvinfer1::Dims dims = engine.getBindingDimensions(i);
        nvinfer1::DataType dtype = engine.getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        std::cout<<engine.getBindingName(i) << totalSize <<std::endl;
    }

    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int out_idx_1 = engine.getBindingIndex(OUTPUT_BLOB_NAME[0].c_str());

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3*320*320 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[out_idx_1], batchSize * GPU_OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize *  3*320*320  * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
//    auto t_start = std::chrono::high_resolution_clock::now();
//    for (int i=0; i<1000; i++) {
//        context.enqueue(batchSize, buffers, stream, nullptr);
//    }
//    auto t_end = std::chrono::high_resolution_clock::now();
//    float total_inf = std::chrono::duration<float, std::milli>(t_end - t_start).count();
//    std::cout << "1000times Inference take: " << total_inf << " ms." << std::endl;


    CHECK(cudaMemcpyAsync(out1, buffers[out_idx_1], batchSize * GPU_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[out_idx_1]));
}


//cpu softmax
template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

struct BBox {
    //center_x center_y w h
    cv::Rect box;
    float conf;  // bbox_conf * cls_conf
    float class_id;
};

//cpu: parsing box
static cv::Rect decodeBox_nanodet(const float*& pdis_pred, float cellx, float celly, int stride, int reg_max) {

    float ct_x = (cellx + 0.5) * stride;
    float ct_y = (celly + 0.5) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(pdis_pred + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float x = (ct_x - dis_pred[0]);
    float y = (ct_y - dis_pred[1]);
    float r = (ct_x + dis_pred[2]);
    float b = (ct_y + dis_pred[3]);

    return cv::Rect(cv::Point(x, y), cv::Point(r + 1, b + 1));
}

void outPutBox(std::vector<BBox>& objs, const cv::Size& imageSize, const cv::Size& netInputSize, float minsize = 15 * 15) {
    float sw = netInputSize.width / (float)imageSize.width;
    float sh = netInputSize.height / (float)imageSize.height;
    float scale_size = std::min(sw, sh);

    for (int i = 0; i < objs.size(); ++i) {
        objs[i].box.x = std::max(0.0f, std::min(objs[i].box.x / scale_size, imageSize.width - 1.0f));
        objs[i].box.y = std::max(0.0f, std::min(objs[i].box.y / scale_size, imageSize.height - 1.0f));
        objs[i].box.width = std::max(0.0f, std::min(objs[i].box.width / scale_size, imageSize.width - 1.0f));
        objs[i].box.height = std::max(0.0f, std::min(objs[i].box.height / scale_size, imageSize.height - 1.0f));
    }
}

void postProcessCPU(const float* cls_tensor, const float* loc_tensor,
                    int stride, int tensor_height, float threshold, int&total, std::vector<BBox>& boxes) {

    int batchSize = 1;
    int cls_num = CLASSES_NUM;
    int loc_dis = 32;
    int feature_h = 320 / stride;
    int feature_w = 320 / stride;
    for (int idx = 0; idx < tensor_height; idx++) {
        const float* pclasses = cls_tensor + cls_num * idx;
        float max_class_confidence = *pclasses;
        int max_classes = 0;
        for (int k = 0; k < cls_num; ++k, ++pclasses) {
            if (*pclasses > max_class_confidence) {
                max_classes = k;
                max_class_confidence = *pclasses;
            }
        }
        if (idx < 10)
            std::cout<<"stride: "<<stride<<"max_classes: "<<max_class_confidence<<std::endl;
        if (max_class_confidence < threshold)
            continue;
//        std::cout<<"max_classes: "<<max_class_confidence<<std::endl;
        total += 1;
        std::cout<<"max cls conf: "<<max_class_confidence<<std::endl;

        int celly = idx / feature_w;
        int cellx = idx % feature_w;
        const float* pdis_pred = loc_tensor + 32 * idx;

        BBox tmp_box;
        tmp_box.box = decodeBox_nanodet(pdis_pred, cellx, celly, stride, reg_max_);
        if (tmp_box.box.area()>0) {
            tmp_box.class_id = max_classes;
            tmp_box.conf = max_class_confidence;
            boxes.push_back(tmp_box);
        }


    }

}

//cpu iou
float iouOf(const cv::Rect&a, const cv::Rect& b){
    float xmax = fmax(a.x, b.x);
    float ymax = fmax(a.y, b.y);
    float xmin = fmin(a.x+a.width, b.x+b.width);
    float ymin = fmin(a.y+a.height, b.y+b.height);
    float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
    float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
    float iou = uw * uh;
    return iou / (a.area() + b.area() - iou);
}

//cpu nms
std::vector<BBox> nms(std::vector<BBox>& objs, float iou_threshold){

    std::sort(objs.begin(), objs.end(), [](const BBox& a, const BBox& b){
        return a.conf > b.conf;
    });

    std::vector<BBox> out;
    std::vector<int> flags(objs.size());
    for (int i = 0; i < objs.size(); ++i){
        if (flags[i] == 1) continue;

        out.push_back(objs[i]);
        flags[i] = 1;
        for (int k = i + 1; k < objs.size(); ++k){
            if (flags[k] == 0){
                float iouUnion = iouOf(objs[i].box, objs[k].box);
                if (iouUnion > iou_threshold)
                    flags[k] = 1;
            }
        }
    }
    return out;
}

//cpu ioufromgpu
float iouFromGPU(const float* a, const float * b){
    float xmax = a[0] > b[0] ? a[0] : b[0];
    float ymax = a[1] > b[1] ? a[1] : b[1];
    float xmin = a[2] < b[2] ? a[2] : b[2];
    float ymin = a[3] < b[3] ? a[3] : b[3];
    float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
    float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
    float iou = uw * uh;
    return iou / (a[4] + b[4] - iou);
}

//nmsfromgpu
void nmsFromGPU(std::vector<decodeplugin::Detection>& res, float *output, float nms_thresh = 0.5) {
    int det_size = sizeof(decodeplugin::Detection) / sizeof(float);
    std::map<float, std::vector<decodeplugin::Detection>> m;
    for (int i = 0; i < output[0] && i < decodeplugin::MAX_OUT; i++) {
        decodeplugin::Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<decodeplugin::Detection>());
        m[det.class_id].push_back(det);
    }

    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), [](const decodeplugin::Detection& a, const decodeplugin::Detection& b){
            return a.conf > b.conf;
        });

        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iouFromGPU(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

void prepareImage(cv::Mat& src_img, std::unique_ptr<float[]>& data) {
    float ratio = 320 / float(src_img.cols) < 320 / float(src_img.rows) ? 320 / float(src_img.cols) : 320 / float(src_img.rows);
    cv::Mat flt_img = cv::Mat::zeros(cv::Size(320, 320), CV_8UC3);
    cv::Mat rsz_img;
    cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
    rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
    flt_img.convertTo(flt_img, CV_32FC3);

    //HWC TO CHW
    int channelLength = 320 * 320;
    float img_mean[3] = {103.53, 116.28, 123.675};
    float img_std[3] =  {57.375,  57.12,  58.395};

    cv::Mat input_channels[3];
    cv::split(flt_img, input_channels);
    for (int i = 0; i < 3; i++) {
        input_channels[i] = (input_channels[i] - img_mean[i]) / img_std[i];
    }
    for (int i = 0; i < 3; i++) {
        memcpy(data.get() + channelLength * i,
               input_channels[2-i].data, channelLength * sizeof(float));
    }

    std::cout<<"debug: "<<img_mean[0]<< " vs "<<img_std[0]<<std::endl;
    printf("debug preprocess: ");
    for (int i=0; i<40; i++)
        printf("%f ", *(float*)(input_channels[2].data + i));
    printf("%f ", data.get()[3*320*320-5]);
    printf("\n");
}

void prepareImageBatch(std::vector<cv::Mat>& vec_img, std::unique_ptr<float[]>& data) {
    //HWC TO CHW
    int channelLength = 320 * 320;
    float img_mean[3] = {103.53, 116.28, 123.675};
    float img_std[3] =  {57.375,  57.12,  58.395};

    for (int m=0; m<vec_img.size(); m++) {
        float ratio = 320 / float(vec_img[m].cols) < 320 / float(vec_img[m].rows) ? 320 / float(vec_img[m].cols) : 320 / float(vec_img[m].rows);
        cv::Mat flt_img = cv::Mat::zeros(cv::Size(320, 320), CV_8UC3);
        cv::Mat rsz_img;
        cv::resize(vec_img[m], rsz_img, cv::Size(), ratio, ratio);
        rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
        flt_img.convertTo(flt_img, CV_32FC3);

        cv::Mat input_channels[3];
        cv::split(flt_img, input_channels);
        for (int i = 0; i < 3; i++) {
            input_channels[i] = (input_channels[i] - img_mean[i]) / img_std[i];
        }
        for (int i = 0; i < 3; i++) {
            memcpy(data.get() + (m*3 + i) * channelLength,
                   input_channels[2-i].data, channelLength * sizeof(float));
        }
    }
}



int main(int argc, char** argv)
{

    //serialize
    
    IHostMemory* modelStream{nullptr};
    APIToModel(4, &modelStream);
    assert(modelStream != nullptr);

    std::ofstream p("jyj_320.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    std::cout<<"done"<<std::endl;
    //return -1;
    
    //deserialzie
    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("./jyj_320.engine", std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    //1个batch
    std::unique_ptr<float[]> data(new float[3*320*320]);
    cv::Mat input_img = cv::imread("/d/images/25p.jpg");
    prepareImage(input_img, data);

    //batch=3
    int batch_size = 1;
//    std::unique_ptr<float[]> data(new float[3*320*320*batch_size]);
//    //std::vector<cv::Mat> input_imgs = {cv::imread("/d/images/0.jpg"), cv::imread("/d/images/20people.jpg"),cv::imread("/d/images/25p.jpg")};
//    std::vector<cv::Mat> input_imgs = {cv::imread("/d/images/0.jpg")};
//    prepareImageBatch(input_imgs, data);



//    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
//        data.get()[i] = 1.0;

//DEBUG
//    static float data[116*40*40];
//    std::map<std::string, Weights> tmp_input = loadWeights("../tmp.wts");
//    printf("debug: %s, ", "input");
//    const float* tmp_data = reinterpret_cast<const float*>(tmp_input["input"].values);
//    for (int k=0; k<116*40*40; k++) {
//        if (k<80)
//            printf("%f ",  tmp_data[k]);
//        data[k] = tmp_data[k];
//    }
//    printf("\n");

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // Run inference
    std::unique_ptr<float[]> out1(new float[out_size[0]]);
    std::unique_ptr<float[]> out2(new float[out_size[1]]);
    std::unique_ptr<float[]> out3(new float[out_size[2]]);
    std::unique_ptr<float[]> out4(new float[out_size[3]]);
    std::unique_ptr<float[]> out5(new float[out_size[4]]);
    std::unique_ptr<float[]> out6(new float[out_size[5]]);

//    static float out1[out_size[0]];
//    static float out2[out_size[1]];
//    static float out3[out_size[2]];
//    static float out4[out_size[3]];
//    static float out5[out_size[4]];
//    static float out6[out_size[5]];

    //gpu后处理

//    std::unique_ptr<float[]> out1(new float[GPU_OUTPUT_SIZE * batch_size]);

    for (int i = 0; i < 1; i++) {
        auto start = std::chrono::system_clock::now();
        doInferenceCPU(*context, data.get(), out1.get(), out2.get(), out3.get(), out4.get(), out5.get(), out6.get(), 1);
//        doInferenceGPU(*context, data.get(), out1.get(), batch_size);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    }

//    for (int m=0; m<batch_size; m++) {
//        std::cout<<"out num: "<<(out1.get()+m * GPU_OUTPUT_SIZE)[0]<<std::endl;
//        std::vector<decodeplugin::Detection> res;
//        nmsFromGPU(res, out1.get() + m * GPU_OUTPUT_SIZE, 0.6) ;
//        std::cout<<"res num: "<<res.size()<<std::endl;
//
//        float sw = decodeplugin::INPUT_W / (float)input_imgs[m].cols;
//        float sh = decodeplugin::INPUT_H / (float)input_imgs[m].rows;
//        float scale_size = std::min(sw, sh);
//
//        for (int i = 0; i < res.size(); ++i) {
//            res[i].bbox[0] = std::max(0.0f, std::min(res[i].bbox[0] / scale_size, input_imgs[m].cols - 1.0f));
//            res[i].bbox[2] = std::max(0.0f, std::min(res[i].bbox[2] / scale_size, input_imgs[m].cols - 1.0f));
//
//            res[i].bbox[1] = std::max(0.0f, std::min(res[i].bbox[1] / scale_size, input_imgs[m].rows - 1.0f));
//            res[i].bbox[3] = std::max(0.0f, std::min(res[i].bbox[3] / scale_size, input_imgs[m].rows - 1.0f));
//
//            cv::rectangle(input_imgs[m], cv::Rect(res[i].bbox[0], res[i].bbox[1], res[i].bbox[2] - res[i].bbox[0],
//                                              res[i].bbox[3] - res[i].bbox[1]), cv::Scalar(0x27, 0xC1, 0x36), 2);
//
//            std::cout<<"cls id: "<< res[i].class_id << "with conf: " << res[i].conf<< std::endl;
//        }
//
//        cv::imshow("test", input_imgs[m]);
//        cv::imwrite("gpu.jpg", input_imgs[m]);
//        cv::waitKey(0);
//    }


    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

//    std::cout << "\nOutput:\n\n";
//    for (unsigned int i = 0; i < out_size[0]; i++)
//    {
//        if (i % 10 == 0) std::cout << out1.get()[i]<< std::endl;
//    }
//    std::cout << std::endl;

    //post processing....
//
//
//
//
//
////------------------------------cpu-----------------------------------//
//    //792 => 1600 => 8
//
//    std::cout<<out1.get()[58*40*40-2]<<std::endl;
//    printf("1first row cls: ");
//    float max_val=0.0;
//    for (int j=0; j<80; j++) {
//        printf("%f ", out1.get()[j]);
//        if (out1.get()[j] > max_val)
//            max_val = out1.get()[j];
//    }
//    printf("\n");
//    printf("max_val: %f\n", max_val);
//
//    printf("3first row cls: ");
//    max_val=0.0;
//    for (int j=0; j<80; j++) {
//        printf("%f ", out3.get()[j]);
//        if (out3.get()[j] > max_val)
//            max_val = out3.get()[j];
//    }
//    printf("\n");
//    printf("max_val: %f\n", max_val);
//
//    printf("5first row cls: ");
//    max_val=0.0;
//    for (int j=0; j<80; j++) {
//        printf("%f ", out5.get()[j]);
//        if (out5.get()[j] > max_val)
//            max_val = out5.get()[j];
//    }
//    printf("\n");
//    printf("max_val: %f\n", max_val);
//
    int total = 0;
    std::vector<BBox> boxes;
    postProcessCPU(out1.get(), out2.get(), 8, 1600, 0.3, total, boxes);
    postProcessCPU(out3.get(), out4.get(), 16, 400, 0.3, total, boxes);
    postProcessCPU(out5.get(), out6.get(), 32, 100, 0.3, total, boxes);
    std::cout<<"total: "<<boxes.size()<<std::endl;

    //nms
    boxes = nms(boxes, 0.6);
    std::cout<<"nms: "<<boxes.size()<<std::endl;
    outPutBox(boxes, input_img.size(), cv::Size(320, 320));
    for(int k=0; k<boxes.size(); k++){
        cv::rectangle(input_img, boxes[k].box, cv::Scalar(0x27, 0xC1, 0x36), 2);
    }
    cv::imshow("test", input_img);
    cv::waitKey(0);




//
//    printf("2first row cls: ");
//    for (int j=0; j<80; j++)
//        printf("%f ", out3.get()[j]);
//    printf("\n");
//    postProcessCPU(out3.get(), out4.get(), 16, 400, 0.3, total);
//
//    printf("3first row cls: ");
//    for (int j=0; j<80; j++)
//        printf("%f ", out5.get()[j]);
//    printf("\n");
//    postProcessCPU(out5.get(), out6.get(), 32, 100, 0.3, total);
//    std::cout<<"total: "<<total<<std::endl;



//    if (argc != 2) {
//        std::cerr << "arguments not right!" << std::endl;
//        std::cerr << "./shufflenet -s   // serialize model to plan file" << std::endl;
//        std::cerr << "./shufflenet -d   // deserialize plan file and run inference" << std::endl;
//        return -1;
//    }
//
//    // create a model using the API directly and serialize it to a stream
//    char *trtModelStream{nullptr};
//    size_t size{0};
//
//    if (std::string(argv[1]) == "-s") {
//        IHostMemory* modelStream{nullptr};
//        APIToModel(1, &modelStream);
//        assert(modelStream != nullptr);
//
//        std::ofstream p("shufflenet.engine", std::ios::binary);
//        if (!p) {
//            std::cerr << "could not open plan output file" << std::endl;
//            return -1;
//        }
//        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
//        modelStream->destroy();
//        return 1;
//    } else if (std::string(argv[1]) == "-d") {
//        std::ifstream file("shufflenet.engine", std::ios::binary);
//        if (file.good()) {
//            file.seekg(0, file.end);
//            size = file.tellg();
//            file.seekg(0, file.beg);
//            trtModelStream = new char[size];
//            assert(trtModelStream);
//            file.read(trtModelStream, size);
//            file.close();
//        }
//    } else {
//        return -1;
//    }
//
//    static float data[3 * INPUT_H * INPUT_W];
//    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
//        data[i] = 1.0;
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
//    }
//
//    // Destroy the engine
//    context->destroy();
//    engine->destroy();
//    runtime->destroy();
//
//    // Print histogram of the output distribution
//    std::cout << "\nOutput:\n\n";
//    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
//    {
//        std::cout << prob[i] << ", ";
//        if (i % 10 == 0) std::cout << i / 10 << std::endl;
//    }
//    std::cout << std::endl;

    return 0;
}
