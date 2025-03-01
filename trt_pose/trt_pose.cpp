//
// Created by hl on 22-3-1.
//
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "post_process.h"

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

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;


const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME_1 = "cmap";
static const int OUTPUT_SIZE_1 = 18*56*56; //18*56*56
const char* OUTPUT_BLOB_NAME_2 = "paf";
static const int OUTPUT_SIZE_2 = 42*56*56; //42*56*56

//const char* OUTPUT_BLOB_TMP = "tmp";
//static const int OUTPUT_SIZE_TMP = 56448; //512*7*7

static int topology[84] = {
        0, 1, 15, 13, 2, 3, 13, 11, 4, 5, 16, 14, 6, 7, 14, 12, 8, 9,
        11, 12, 10, 11, 5, 7, 12, 13, 6, 8, 14, 15, 7, 9, 16, 17, 8, 10,
        18, 19, 1, 2, 20, 21, 0, 1, 22, 23, 0, 2, 24, 25, 1, 3, 26, 27,
        2, 4, 28, 29, 3, 5, 30, 31, 4, 6, 32, 33, 17, 0, 34, 35, 17, 5,
        36, 37, 17, 6, 38, 39, 17, 11, 40, 41, 17, 12
};
static float threshold = 0.1;

using namespace nvinfer1;

static Logger gLogger;

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
        printf("wts: %s %d\n", name.c_str(), size);
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + "._mean"].values;
    float *var = (float*)weightMap[lname + "._variance"].values;
    int len = weightMap[lname + "._variance"].count;
    std::cout << "len " << len << std::endl;

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


IActivationLayer* transCBR(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,
        const int& idx, const std::string& lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto conv1 = network->addDeconvolutionNd(input, 256, DimsHW{4, 4},
            weightMap[lname + std::to_string(idx)+ ".weight"], weightMap[lname + std::to_string(idx)+ ".bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + std::to_string(idx+1), 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    return relu1;
}


IActivationLayer* UpsampleCBR(INetworkDefinition *network, std::map<std::string, Weights>& weightMap,
        ITensor& input, const std::string& lname) {
    auto block1 = transCBR(network, weightMap, input, 0, lname);
    auto block2 = transCBR(network, weightMap, *block1->getOutput(0), 3, lname);
    auto block3 = transCBR(network, weightMap, *block2->getOutput(0), 6, lname);
    return block3;
}

IActivationLayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{stride, stride});
        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape { 3, INPUT_H, INPUT_W } with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../trt_pose.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["0.resnet.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "0.resnet.bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStrideNd(DimsHW{2, 2});
    pool1->setPaddingNd(DimsHW{1, 1});

    IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "0.resnet.layer1.0.");
    IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "0.resnet.layer1.1.");

    IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "0.resnet.layer2.0.");
    IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "0.resnet.layer2.1.");

    IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "0.resnet.layer3.0.");
    IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "0.resnet.layer3.1.");

    IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "0.resnet.layer4.0.");
    IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "0.resnet.layer4.1."); //fp16有明显差异


    //cmap
    auto upsample_cmap = UpsampleCBR(network, weightMap, *relu9->getOutput(0), "1.cmap_up.");
    IConvolutionLayer* cmap_att = network->addConvolutionNd(*upsample_cmap->getOutput(0), 256, DimsHW{3, 3},
            weightMap["1.cmap_att.weight"], weightMap["1.cmap_att.bias"]);
    cmap_att->setPaddingNd(DimsHW{1, 1});
    assert(cmap_att);

    auto *sigmoid = network->addActivation(*cmap_att->getOutput(0), ActivationType::kSIGMOID);
    auto camp_prod = network->addElementWise(*upsample_cmap->getOutput(0), *sigmoid->getOutput(0),
                                                        ElementWiseOperation::kPROD);

    IConvolutionLayer* cmap_att_conv = network->addConvolutionNd(*camp_prod->getOutput(0), 18, DimsHW{1, 1},
            weightMap["1.cmap_conv.weight"], weightMap["1.cmap_conv.bias"]);
    assert(cmap_att_conv);

    cmap_att_conv->getOutput(0)->setName(OUTPUT_BLOB_NAME_1);
    network->markOutput(*cmap_att_conv->getOutput(0));

    //paf
    auto upsample_paf = UpsampleCBR(network, weightMap, *relu9->getOutput(0), "1.paf_up.");
    IConvolutionLayer* paf_att = network->addConvolutionNd(*upsample_paf->getOutput(0), 256, DimsHW{3, 3},
            weightMap["1.paf_att.weight"], weightMap["1.paf_att.bias"]);
    paf_att->setPaddingNd(DimsHW{1, 1});
    assert(paf_att);
    auto *tanh = network->addActivation(*paf_att->getOutput(0), ActivationType::kTANH);
    auto paf_prod = network->addElementWise(*upsample_paf->getOutput(0), *tanh->getOutput(0),
                                             ElementWiseOperation::kPROD);
    IConvolutionLayer* paf_att_conv = network->addConvolutionNd(*paf_prod->getOutput(0), 42, DimsHW{1, 1},
            weightMap["1.paf_conv.weight"], weightMap["1.paf_conv.bias"]);
    assert(paf_att_conv);
    paf_att_conv->getOutput(0)->setName(OUTPUT_BLOB_NAME_2);
    network->markOutput(*paf_att_conv->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    config->setFlag(BuilderFlag::kFP16); //fp16还是有一定精度损失
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

void doInference(IExecutionContext& context, float* input, float* cmap_output, float* paf_output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME_1);
    const int outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME_2);
//    const int outputIndex_tmp = engine.getBindingIndex(OUTPUT_BLOB_TMP);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * OUTPUT_SIZE_1 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * OUTPUT_SIZE_2 * sizeof(float)));
//    CHECK(cudaMalloc(&buffers[outputIndex_tmp], batchSize * OUTPUT_SIZE_TMP * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(cmap_output, buffers[outputIndex1], batchSize * OUTPUT_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    CHECK(cudaMemcpyAsync(paf_output, buffers[outputIndex2], batchSize * OUTPUT_SIZE_2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

//    CHECK(cudaMemcpyAsync(cmap_output, buffers[outputIndex_tmp], batchSize * OUTPUT_SIZE_TMP * sizeof(float), cudaMemcpyDeviceToHost, stream));
//    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex1]));
    CHECK(cudaFree(buffers[outputIndex2]));
//    CHECK(cudaFree(buffers[outputIndex_tmp]));

}

void preprocess_image(const cv::Mat& img, float* data) {
    cv::Mat resizeimage;
    cv::resize(img, resizeimage, cv::Size(224, 224), cv::INTER_NEAREST); //不进行resize效果更好,  图像有变形
//    int w, h, x, y;
//    float r_w = 224 / (img.cols * 1.0);
//    float r_h = 224 / (img.rows * 1.0);
//    float r_b;
//    if (r_h > r_w) {
//        w = 224;
//        h = r_w * img.rows;
//        x = 0;
//        y = (224 - h) / 2;
//        r_b = r_w;
//    } else {
//        w = r_h * img.cols;
//        h = 224;
//        x = (224 - w) / 2;
//        y = 0;
//        r_b = r_h;
//    }
//    cv::Mat re(h, w, CV_8UC3);
//    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
//    cv::Mat out(224, 224, CV_8UC3);
//    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
//    cv::cvtColor(out, resizeimage, cv::COLOR_BGR2RGB);

    resizeimage.convertTo(resizeimage, CV_32FC3, 1. / 255.0);
    cv::Mat input_channels[3];
    cv::split(resizeimage, input_channels);
    int channelLength = 224 * 224;
    input_channels[0] = (input_channels[0] - 0.485) / 0.229;
    input_channels[1] = (input_channels[1] - 0.456) / 0.224;
    input_channels[2] = (input_channels[2] - 0.406) / 0.225;
    for (int j = 0; j < 3; j++) {
        memcpy(data + channelLength * j,
               input_channels[j].data, channelLength * sizeof(float));
    }
}

void show_img(cv::Mat &image1, int object_counts, float *peaks, int *objects) {

    cv::Mat image = image1.clone();

    //图片展示
    int h = image.rows;
    int w = image.cols;
    int k = 21;
    int c = 18;

    for (int i = 0; i < object_counts; i++) {
        //描点
        for (int j = 0; j < c; j++) {
            int tmp_k = objects[c * i + j];
            int offset = 30 * 2 * j + i * 2;
            if (tmp_k >= 0) {
                int x = round(peaks[offset + 1] * w);
                int y = round(peaks[offset] * h);
                //std::cout<<"tmp_k: "<<tmp_k<<", offset: "<<offset<<", circle: "<<x<<"---->"<<y<<" vs "<< peaks[offset + 1] <<"...--->" <<  peaks[offset + 0]  <<std::endl;
                cv::Scalar color = cv::Scalar(0, 255, 0);
                if (j == 17)
                    color = cv::Scalar(0, 0, 255);
                cv::circle(image, cv::Point(x, y), 3, color, cv::FILLED);
            }
        }

        //连线
        for (int m = 0; m < k; m++) {
            int c_a = topology[m * 4 + 2];
            int c_b = topology[m * 4 + 3];
            int obj_1 = objects[c * i + c_a];
            int obj_2 = objects[c * i + c_b];
            if (obj_1 >= 0 && obj_2 >= 0) {
                int offset_0 = c_a * 60 + obj_1 * 2;
                int x0 = round(peaks[offset_0 + 1] * w);
                int y0 = round(peaks[offset_0] * h);

                int offset_1 = c_b * 60 + obj_2 * 2;
                int x1 = round(peaks[offset_1 + 1] * w);
                int y1 = round(peaks[offset_1] * h);
                cv::line(image, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 0, 255), 2);
            }

        }
    }
    cv::resize(image, image, cv::Size(640, 480));
    cv::imshow("image", image);
}

void postprocess(cv::Mat img, float* cmap, float* paf, bool show=false, bool is_video=false) {
    // Print histogram of the output distribution
//    std::cout << "\ncamp output:\n\n";
//    for (unsigned int i = 0; i < 10; i++)
//    {
//        std::cout << cmap[i] << ", ";
//    }
//    std::cout << std::endl;
//    for (unsigned int i = 0; i < 10; i++)
//    {
//        std::cout << cmap[OUTPUT_SIZE_1 - 10 + i] << ", ";
//    }
//    std::cout << std::endl;
//
//
//    std::cout << "\npaf output:\n\n";
//    for (unsigned int i = 0; i < 10; i++)
//    {
//        std::cout << paf[i] << ", ";
//    }
//    std::cout << std::endl;
//    for (unsigned int i = 0; i < 10; i++)
//    {
//        std::cout << paf[OUTPUT_SIZE_2 - 10 + i] << ", ";
//    }
//    std::cout << std::endl;
//    return ;

//    std::cout<<img.cols<<" vs "<<img.size<<std::endl;

    int objects[30 * 18];
    int object_counts = -1;
    float peaks[2 * 18 * 30] = {0};

    int counts[18] = {0};
    float thresholds[30] = {0};
    try {
        find_peaks(cmap, counts, peaks, thresholds, threshold);
        float score_graph[21 * 30 * 30] = {0};
        paf_score_graph(paf, peaks, score_graph, topology, counts);
        int connections[21 * 2 * 30];
        memset(connections, -1, sizeof(connections));
        assignement(score_graph, topology, counts, connections);

        memset(objects, -1, sizeof(objects));
        connect_parts(connections, topology, counts, &object_counts, objects);
    } catch (const std::exception &e) {
        printf("TrtTrtPose Detect error");
        return ;
    }

//    printf("obj count: %d\n", object_counts);
    if (show) {
        show_img(img, object_counts, peaks, objects);
        if (is_video)
            cv::waitKey(1);
        else
            cv::waitKey(0);
    }



}

void infer_img(IExecutionContext* context){
    // Subtract mean from image
    static float data[3 * INPUT_H * INPUT_W];
//    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
//        data[i] = 1.0;

    cv::Mat img = cv::imread("/d/images/10p.jpeg");
    preprocess_image(img, data);


    // Run inference
    std::unique_ptr<float[]> cmap(new float[OUTPUT_SIZE_1]);
    std::unique_ptr<float[]> paf(new float[OUTPUT_SIZE_2]);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1; i++) {
        doInference(*context, data, cmap.get(), paf.get(), 1);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    //post process
    start = std::chrono::system_clock::now();
    for (int i = 0; i < 1; i++) {
        postprocess(img, cmap.get(), paf.get(), true);
    }

    end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;


}

void infer_video(IExecutionContext* context, bool show=false){
    // Subtract mean from image
    std::unique_ptr<float[]> cmap(new float[OUTPUT_SIZE_1]);
    std::unique_ptr<float[]> paf(new float[OUTPUT_SIZE_2]);

    const ICudaEngine& engine = context->getEngine();
    assert(engine.getNbBindings() == 3);
    void* buffers[3];
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME_1);
    const int outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME_2);
    CHECK(cudaMalloc(&buffers[inputIndex], 1 * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex1], 1 * OUTPUT_SIZE_1 * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex2], 1 * OUTPUT_SIZE_2 * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    std::string v_path = "/f/dataset/action/turn.mp4";
    static float data[3 * INPUT_H * INPUT_W];

    cv::Mat img;
    cv::VideoCapture cap(v_path);
    int frame_no = 0;
    int64_t infer_time = 0;
    int64_t post_time = 0;
    int64_t total_time = 0;
    while (true) {
        if (!cap.read(img)) {
            std::cout << "read video error" << std::endl;
            break;
        }

        auto start0 = std::chrono::system_clock::now();
        preprocess_image(img, data);

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], data, 1 * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        auto start = std::chrono::system_clock::now();
        context->enqueue(1, buffers, stream, nullptr);
        auto end = std::chrono::system_clock::now();
        CHECK(cudaMemcpyAsync(cmap.get(), buffers[outputIndex1], 1 * OUTPUT_SIZE_1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        CHECK(cudaMemcpyAsync(paf.get(), buffers[outputIndex2], 1 * OUTPUT_SIZE_2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        infer_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        auto start2 = std::chrono::system_clock::now();
        postprocess(img, cmap.get(), paf.get(), show, true);
        auto end2 = std::chrono::system_clock::now();
        post_time += std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start0).count();

        frame_no += 1;
    }

    std::cout<<"视频总帧数: "<<frame_no<<std::endl;
    std::cout<<"平均推理耗时(ms): "<<infer_time / float(frame_no) <<std::endl;
    std::cout<<"平均后处理耗时(ms): "<<post_time / float(frame_no) <<std::endl;
    std::cout<<"平均单帧耗时(ms): "<<total_time / float(frame_no) <<std::endl;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./trt_pose -s   // serialize model to plan file" << std::endl;
        std::cerr << "./trt_pose -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("trt_pose.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("trt_pose.engine", std::ios::binary);
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
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

//    infer_img(context);
    infer_video(context, false);


    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
