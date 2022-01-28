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
#include "yolov5n_v6_prune_plugin.h"
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



const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
//const char* INPUT_MODEL_WEIGHTS = "../yolov5n_v6_pruned.wts";
const char* INPUT_MODEL_WEIGHTS = "../generated.wts";
const int BATCH_SIZE = 1;
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 384;
const int CLASS_NUM = 2;
const int MAX_OUTPUT_BOXES = 100;
const int OUTPUT_SIZE = MAX_OUTPUT_BOXES * sizeof(DetectRes) / sizeof(float) + 1;

using namespace nvinfer1;

sample::Logger gLogger;

//opencv 图像预处理
void ImagePrepare(cv::Mat& image, std::unique_ptr<float>& input_data) {
    int channelLength = INPUT_WIDTH * INPUT_HEIGHT;
    int w, h, x, y;

    float r_w = INPUT_WIDTH / (image.cols * 1.0);
    float r_h = INPUT_HEIGHT / (image.rows * 1.0);
    if (r_h > r_w) {
        w = INPUT_WIDTH;
        h = r_w * image.rows;
        x = 0;
        y = (INPUT_HEIGHT - h) / 2;
    } else {
        w = r_h * image.cols;
        h = INPUT_HEIGHT;
        x = (INPUT_WIDTH - w) / 2;
        y = 0;
    }

    cv::Mat re(h, w, CV_8UC3);
    cv::resize(image, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    //split channels
    out.convertTo(out, CV_32FC3, 1. / 255.);
    cv::Mat input_channels[3];
    cv::split(out, input_channels);
    for (int j = 0; j < 3; j++) {
        memcpy(input_data.get() + channelLength * j, input_channels[2 - j].data, channelLength * sizeof(float));
    }
}


//todo cuda 图像预处理

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

static bool cmp(DetectRes& a, DetectRes& b) {
    return a.conf > b.conf;
}

static void nms(std::vector<DetectRes>& res, float *output, float conf_thresh, float nms_thresh = 0.5) {
    int det_size = sizeof(DetectRes) / sizeof(float);
    std::map<float, std::vector<DetectRes>> m;
    for (int i = 0; i < output[0] && i < MAX_OUTPUT_BOXES; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        DetectRes det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<DetectRes>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
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
}

static void get_rect(cv::Rect& rect, const int img_w, const int img_h, float bbox[4], int input_w, int input_h) {
    int l, r, t, b;
    float r_w = input_w / (img_w * 1.0);
    float r_h = input_h / (img_h * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (input_h - r_w * img_h) / 2;
        b = bbox[1] + bbox[3]/2.f - (input_h - r_w * img_h) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (input_w - r_h * input_w) / 2;
        r = bbox[0] + bbox[2]/2.f - (input_w - r_h * input_w) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    rect.x = l;
    rect.y = t;
    rect.width = r - l;
    rect.height = b - t;
}



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

static std::vector<std::vector<float>> getAnchors(std::map<std::string, WeightsData> &weightMap, std::string lname) {
    std::vector<std::vector<float>> anchors;
    Weights wts = weightMap[lname + ".anchor_grid"].wts;
    int anchor_len = 3 * 2;
    for (int i = 0; i < wts.count / anchor_len; i++) {
        auto *p = (const float *) wts.values + i * anchor_len;
        std::vector<float> anchor(p, p + anchor_len);
        anchors.push_back(anchor);
    }
    return anchors;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}


static IScaleLayer *addBatchNorm2d6(INetworkDefinition *network, std::map<std::string, WeightsData> &weightMap, ITensor &input,
                                    std::string lname, float eps) {
    float *gamma = (float *) weightMap[lname + ".weight"].wts.values;
    float *beta = (float *) weightMap[lname + ".bias"].wts.values;
    float *mean = (float *) weightMap[lname + ".running_mean"].wts.values;
    float *var = (float *) weightMap[lname + ".running_var"].wts.values;
    int len = weightMap[lname + ".running_var"].wts.count;

    float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};

    float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"].wts = scale;
    weightMap[lname + ".shift"].wts = shift;
    weightMap[lname + ".power"].wts = power;
    IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

static ILayer *
convBlock(INetworkDefinition *network, std::map<std::string, WeightsData> &weightMap, ITensor &input, int ksize,
          int s, int g, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    int p = ksize / 3;
    int outch = weightMap[lname + ".conv.weight"].dim1;

    IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize},
                                                         weightMap[lname + ".conv.weight"].wts, emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    conv1->setNbGroups(g);
    IScaleLayer *bn1 = addBatchNorm2d6(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // silu = x * sigmoid
    auto sig = network->addActivation(*bn1->getOutput(0), ActivationType::kSIGMOID);
    assert(sig);
    auto ew = network->addElementWise(*bn1->getOutput(0), *sig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

static ILayer *
bottleneck6(INetworkDefinition *network, std::map<std::string, WeightsData> &weightMap, ITensor &input,
            bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), 3, 1, g, lname + ".cv2");

    if (shortcut && weightMap[lname + ".cv1"].dim1 == weightMap[lname + ".cv2"].dim1) {
        auto ew = network->addElementWise(input, *cv2->getOutput(0), ElementWiseOperation::kSUM);
        return ew;
    }
    return cv2;
}

static ILayer *
C3(INetworkDefinition *network, std::map<std::string, WeightsData> &weightMap, ITensor &input, int n,
   bool shortcut, int g, float e, std::string lname) {
    auto cv1 = convBlock(network, weightMap, input, 1, 1, 1, lname + ".cv1");
    auto cv2 = convBlock(network, weightMap, input, 1, 1, 1, lname + ".cv2");
    ITensor *y1 = cv1->getOutput(0);
    for (int i = 0; i < n; i++) {
        auto b = bottleneck6(network, weightMap, *y1, shortcut, g, 1.0, lname + ".m." + std::to_string(i));
        y1 = b->getOutput(0);
    }

    ITensor *inputTensors[] = {y1, cv2->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 2);

    auto cv3 = convBlock(network, weightMap, *cat->getOutput(0), 1, 1, 1, lname + ".cv3");
    return cv3;
}

static ILayer *
SPPF(INetworkDefinition *network, std::map<std::string, WeightsData> &weightMap, ITensor &input,  int k,
     std::string lname) {

    auto cv1 = convBlock(network, weightMap, input, 1, 1, 1, lname + ".cv1");

    auto pool1 = network->addPoolingNd(*cv1->getOutput(0), PoolingType::kMAX, DimsHW{k, k});
    pool1->setPaddingNd(DimsHW{k / 2, k / 2});
    pool1->setStrideNd(DimsHW{1, 1});
    auto pool2 = network->addPoolingNd(*pool1->getOutput(0), PoolingType::kMAX, DimsHW{k, k});
    pool2->setPaddingNd(DimsHW{k / 2, k / 2});
    pool2->setStrideNd(DimsHW{1, 1});
    auto pool3 = network->addPoolingNd(*pool2->getOutput(0), PoolingType::kMAX, DimsHW{k, k});
    pool3->setPaddingNd(DimsHW{k / 2, k / 2});
    pool3->setStrideNd(DimsHW{1, 1});
    ITensor *inputTensors[] = {cv1->getOutput(0), pool1->getOutput(0), pool2->getOutput(0), pool3->getOutput(0)};
    auto cat = network->addConcatenation(inputTensors, 4);
    auto cv2 = convBlock(network, weightMap, *cat->getOutput(0), 1, 1, 1, lname + ".cv2");
    return cv2;
}

static IPluginV2Layer *addYoLoLayer(INetworkDefinition *network, std::map<std::string, WeightsData> &weightMap, std::string lname,
                                    std::vector<IConvolutionLayer *> dets, int cls_num, int input_w, int input_h) {
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer6_TRT", "1");
    auto anchors = getAnchors(weightMap, lname);
    PluginField plugin_fields[2];
    int netinfo[4] = {cls_num, input_w, input_h, MAX_OUTPUT_BOXES};
    plugin_fields[0].data = netinfo;
    plugin_fields[0].length = 4;
    plugin_fields[0].name = "netinfo";
    plugin_fields[0].type = PluginFieldType::kFLOAT32;
    int scale = 8;
    std::vector<YoloKernel> kernels;
    for (size_t i = 0; i < anchors.size(); i++) {
        YoloKernel kernel;
        kernel.width = input_w / scale;
        kernel.height = input_h/ scale;
        memcpy(kernel.anchors, &anchors[i][0], anchors[i].size() * sizeof(float));
        kernels.push_back(kernel);
        scale *= 2;
    }
    plugin_fields[1].data = &kernels[0];
    plugin_fields[1].length = kernels.size();
    plugin_fields[1].name = "kernels";
    plugin_fields[1].type = PluginFieldType::kFLOAT32;
    PluginFieldCollection plugin_data;
    plugin_data.nbFields = 2;
    plugin_data.fields = plugin_fields;
    IPluginV2 *plugin_obj = creator->createPlugin("yololayer", &plugin_data);
    std::vector<ITensor *> input_tensors;
    for (auto det: dets) {
        input_tensors.push_back(det->getOutput(0));
    }
    auto yolo = network->addPluginV2(&input_tensors[0], input_tensors.size(), *plugin_obj);
    return yolo;
}


// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] [dim1] [dim2] <data x size in hex>
std::map<std::string, WeightsData> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, WeightsData> weightMap;

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
        uint32_t dim1;
        uint32_t dim2;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size >> std::dec >> dim1 >> std::dec >> dim2;

        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;

        WeightsData wd;
        wd.wts = wt;
        wd.dim1 = dim1;
        wd.dim2 = dim2;
//        std::cout<<name<< " " << dim1 << " " << dim2 << " "<< size << std::endl;
        weightMap[name] = wd;
    }

    return weightMap;
}


static nvinfer1::ICudaEngine *  //用的默认的build_engine
createEngine(const char* input_model_path, int m_classNum, int m_inputWidth, int m_inputHeight,
                   int maxBatchSize, IBuilder* builder, IBuilderConfig* config, bool use_fp16 = true, char net_model = 'n') {

    auto gd = 0.0f; //深度由网络类型决定, 通道宽度可变, 跟裁剪有关,设置从模型文件中读取
    if (net_model == 'n') {
        gd = 0.33;
    } else if (net_model == 's') {
        gd = 0.33;
    } else if (net_model == 'm') {
        gd = 0.67;
    } else if (net_model == 'l') {
        gd = 1.0;
    } else if (net_model == 'x') {
        gd = 1.33;
    }

    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims3{ 3, m_inputHeight, m_inputWidth });
    assert(data);
    std::map<std::string, WeightsData> weightMap = loadWeights(input_model_path);

    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data, 6, 2, 1,  "model.0");
    assert(conv0);

    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), 3, 2, 1, "model.7");
    auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), 5, "model.9");

    /* ------ yolov5 head ------ */
    auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), 1, 1, 1, "model.10");
    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    auto csp6_dim = bottleneck_csp6->getOutput(0)->getDimensions();
    auto conv10_dim = conv10->getOutput(0)->getDimensions();
    csp6_dim.d[0] = conv10_dim.d[0];
    upsample11->setOutputDimensions(csp6_dim);

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };

    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    auto csp4_dim = bottleneck_csp4->getOutput(0)->getDimensions();
    auto conv14_dim = conv14->getOutput(0)->getDimensions();
    csp4_dim.d[0] = conv14_dim.d[0];
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(csp4_dim);

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_depth(3, gd), false, 1, 0.5, "model.17");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (m_classNum + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"].wts, weightMap["model.24.m.0.bias"].wts);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (m_classNum + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"].wts, weightMap["model.24.m.1.bias"].wts);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (m_classNum + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"].wts, weightMap["model.24.m.2.bias"].wts);

    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2}, m_classNum, m_inputWidth, m_inputHeight);
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.wts.values));
    }

    return engine;
}


void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(INPUT_MODEL_WEIGHTS, CLASS_NUM, INPUT_WIDTH, INPUT_HEIGHT, BATCH_SIZE, builder, config);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

//void doInference(IExecutionContext& context, float* input, float* output, int batchSize, bool )
//{
//
//    cudaError_t error = cudaGetLastError();
//    const ICudaEngine& engine = context.getEngine();
//
//    // Pointers to input and output device buffers to pass to engine.
//    // Engine requires exactly IEngine::getNbBindings() number of buffers.
//    assert(engine.getNbBindings() == 2);
//    void* buffers[2];
//
//    // In order to bind the buffers, we need to know the names of the input and output tensors.
//    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
//    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
//    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
//
//    // Create GPU buffers on device
//    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float)));
//    error = cudaGetLastError();
//
//    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
//    error = cudaGetLastError();
//
//    // Create stream
//    cudaStream_t stream;
//    CHECK(cudaStreamCreate(&stream));
//
//    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_HEIGHT * INUPT_WIDTH * sizeof(float), cudaMemcpyHostToDevice, stream));
//    context.enqueue(batchSize, buffers, stream, nullptr);
//    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
//    cudaStreamSynchronize(stream);
//
//    error = cudaGetLastError();
//
//    // Release stream and buffers
//    cudaStreamDestroy(stream);
//    CHECK(cudaFree(buffers[inputIndex]));
//    CHECK(cudaFree(buffers[outputIndex]));
//}

void run_video(IExecutionContext& context, std::string& video_path, bool is_show=true) {
    const ICudaEngine& engine = context.getEngine();
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    cv::VideoCapture cap;
    cap.open(video_path);
    cv::Mat img;
    std::unique_ptr<float> input_data(new float[BATCH_SIZE * 3 * INPUT_HEIGHT * INPUT_WIDTH]);
    std::unique_ptr<float> output_data(new float[BATCH_SIZE * OUTPUT_SIZE]);
    while (true) {
        if (!cap.read(img)) {
            std::cout << "read cam error" << std::endl;
            break;
        }

        //预处理
        ImagePrepare(img, input_data);

        //模型推理
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input_data.get(), BATCH_SIZE * 3 * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(BATCH_SIZE, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output_data.get(), buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        //todo 后处理
//        std::cout<<"num: "<< output_data.get()[0]<<std::endl;

        std::vector<DetectRes> batch_res;
        batch_res.clear();

        nms(batch_res, output_data.get(), 0.5, 0.5);

        //修复边界框坐标
        size_t obj_num = batch_res.size();
        for (size_t j = 0; j < obj_num; j++) {
            cv::Rect tmp_rect;
            get_rect(tmp_rect, img.cols, img.rows, batch_res[j].bbox, INPUT_WIDTH, INPUT_HEIGHT);

            cv::putText(img, std::to_string((int)batch_res[j].class_id), cv::Point(batch_res[j].bbox[0], batch_res[j].bbox[1] - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            cv::rectangle(img, cv::Point(batch_res[j].bbox[0], batch_res[j].bbox[1]),  cv::Point(batch_res[j].bbox[2], batch_res[j].bbox[3]), cv::Scalar(0, 0, 255), 3);
        }



        cv::imshow("demo", img);
        auto key = cv::waitKey(20);
        if (key == 'q') {
            break;
        }

    }
    cap.release();
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);


//        std::vector<yolov5FaceConfig::FaceBox> fboxes;
//    nms(fboxes, prob);
//    printf("output size: %f, boxes: %d, img_w: %d, img_h: %d\n", prob[0], fboxes.size(), img.cols, img.rows);
//    for (int i = 0; i < fboxes.size(); i++) {
//        //画框
//        cv::putText(tmp, std::to_string((int)dr[j].class_id), cv::Point(dr[j].rect.x, dr[j].rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
//        cv::rectangle(img, cv::Point(fboxes[i].bbox[0], fboxes[i].bbox[1]),
//                      cv::Point(fboxes[i].bbox[2], fboxes[i].bbox[3]), cv::Scalar(0, 0, 255), 3);
//
//    }



}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5n_v6_prune -s   // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5n_v6_prune -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);

        assert(modelStream != nullptr);

        std::ofstream p("yolov5n_v6_prune.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
//        return 1;

        std::ifstream file("yolov5n_v6_prune.engine", std::ios::binary);
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
        std::ifstream file("yolov5n_v6_prune.engine", std::ios::binary);
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

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    std::string vpath = "/d/涛哥专用/dui/censong_0908_dui4.m4v";
    run_video(*context, vpath);


    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();



    return 0;
}
