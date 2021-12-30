//
// Created by haha on 2019/12/17.
//

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>

// stuff we know about the network and the input/output blobs
int OUTPUT_SIZE = 4;
static const int INPUT_DIM = 5268480; //输入变量的维度: 7*15*224*224
std::string INPUT_BLOB_NAME = "image"; //输入节点的名称
std::string OUTPUT_BLOB_NAME = "save_infer_model/scale_0"; //输出节点的名称
std::string wts_file = "../src_models/pd_stnet.wts"; //原权重文件
//std::string engine_file =  "../dest_models/pd_stnet.trtengine"; //生成的trt引擎文件
std::string engine_file =  "~/projects/jetson/test_lxr_emotion/dest_models/pd_stnet.trtengine"; //生成的trt引擎文件
int batchSize = 1; //支持多个batch
int runIters = 1; //测试推理的次数
bool use_fp16 = true;

using namespace nvinfer1;

Logger gLogger;

//检查权重是否存在
bool existsWeightKey(std::map<std::string, Weights>& wts, const std::string& key) {
    if (wts.find(key) == wts.end()) {
        std::cout<<"this key "<<key<<" does not exists"<<std::endl;
        return false; //不存在
    }
    return true;
}

//设置输出
void setOutput(ITensor& output, INetworkDefinition* network){
    Dims d = output.getDimensions();
    OUTPUT_SIZE = 1;

    std::cout<<"output dims: ";
    for (int i=0; i< d.nbDims; i++) {
        OUTPUT_SIZE *= d.d[i];
        std::cout<<d.d[i]<<" ";
    }
    std::cout<<std::endl;

    output.setName(OUTPUT_BLOB_NAME.c_str());
    network->markOutput(output);
}

// 加载权重文件
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    //read input and output_name
    input >> INPUT_BLOB_NAME;
    input >> OUTPUT_BLOB_NAME;

    std::cout<<"input name: "<<INPUT_BLOB_NAME<<std::endl;
    std::cout<<"output name: "<<OUTPUT_BLOB_NAME<<std::endl;

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

// 2d的bn层
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    assert(existsWeightKey(weightMap, lname + "_scale"));
    assert(existsWeightKey(weightMap, lname + "_offset"));
    assert(existsWeightKey(weightMap, lname + "_mean"));
    assert(existsWeightKey(weightMap, lname + "_variance"));

    float *gamma = (float*)weightMap[lname + "_scale"].values;
    float *beta = (float*)weightMap[lname + "_offset"].values;
    float *mean = (float*)weightMap[lname + "_mean"].values;
    float *var = (float*)weightMap[lname + "_variance"].values;
    int len = weightMap[lname + "_variance"].count;

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


//resnet的bottleneck组件
ITensor* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    std::string conv_name;
    //lname = "3a"

    //左侧分支3卷积
    std::string brname = "_branch2a";
    conv_name = "res" + lname + brname + "_weights";
    assert(existsWeightKey(weightMap, conv_name));
    IConvolutionLayer* conv1 = network->addConvolution(input, outch, DimsHW{1, 1}, weightMap[conv_name], emptywts); //res3a_branch2a_weights
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn" + lname + brname, 1e-5); //bn3a_branch2a ===> bn3a_branch2a_scale

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    brname = "_branch2b";
    conv_name = "res" + lname + brname + "_weights";
    assert(existsWeightKey(weightMap, conv_name)); //res3a_branch2b_weigts
    IConvolutionLayer* conv2 = network->addConvolution(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[conv_name], emptywts);
    assert(conv2);
    conv2->setStride(DimsHW{stride, stride});
    conv2->setPadding(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "bn" + lname + brname, 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    brname = "_branch2c";
    conv_name = "res" + lname + brname + "_weights";
    assert(existsWeightKey(weightMap, conv_name));//res3a_branch2c_weigts
    IConvolutionLayer* conv3 = network->addConvolution(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[conv_name], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), "bn" + lname + brname, 1e-5);

    //检查输入输出维度是否匹配
    IElementWiseLayer* ew1;
    //shortcut: ch_out = num_filters*4 && ch_in != ch_out or stride != 1
    if (stride != 1 || inch != outch * 4) {
        brname = "_branch1";
        conv_name = "res" + lname + brname + "_weights";
        assert(existsWeightKey(weightMap, conv_name)); //res3a_branch1_weights
        IConvolutionLayer* conv4 = network->addConvolution(input, outch * 4, DimsHW{1, 1}, weightMap[conv_name], emptywts);
        assert(conv4);
        conv4->setStride(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), "bn" + lname + brname, 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3->getOutput(0);
}


//时域模块的bn层
IScaleLayer* temp_addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    assert(existsWeightKey(weightMap, lname + ".w_0"));
    assert(existsWeightKey(weightMap, lname + ".b_0"));
    assert(existsWeightKey(weightMap, lname + ".w_1"));
    assert(existsWeightKey(weightMap, lname + ".w_2"));

    float *gamma = (float*)weightMap[lname + ".w_0"].values;
    float *beta = (float*)weightMap[lname + ".b_0"].values;
    float *mean = (float*)weightMap[lname + ".w_1"].values;
    float *var = (float*)weightMap[lname + ".w_2"].values;
    int len = weightMap[lname + ".w_2"].count;

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


//时域模块
ITensor* temporalBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,
        int num_filters, std::string conv_name, std::string bn_name) {

    //reshape and permutation, 注意: 此处没有进行reshape, 原因是trt这里只维护一个批次的数据input
    nvinfer1::IShuffleLayer* input_reshape = network->addShuffle(input);
    assert(input_reshape);
    input_reshape->setName((conv_name + " @inshape").c_str());
    input_reshape->setSecondTranspose(Permutation{1, 0, 2, 3});  //512*7*28*28
    assert(input_reshape);

    assert(existsWeightKey(weightMap, conv_name+".w_0"));
    assert(existsWeightKey(weightMap, conv_name+".b_0"));

    //conv3d
    IConvolutionLayer* conv3d = network->addConvolutionNd(*input_reshape->getOutput(0), num_filters, Dims3{3, 1, 1},
            weightMap[conv_name+".w_0"], weightMap[conv_name+".b_0"]);
    conv3d->setStrideNd(Dims3{1, 1, 1});
    conv3d->setPaddingNd(Dims3{1, 0, 0});
    conv3d->setNbGroups(1);
    assert(conv3d); //512*7*28*28

    //relu
    IActivationLayer* relu = network->addActivation(*conv3d->getOutput(0), ActivationType::kRELU);
    assert(relu); //512*7*28*28


    //transpose, 注意：这里将transpose提前了，原因是BN加载权重的维度不匹配, 以后再思考有没有更好的办法
    nvinfer1::IShuffleLayer* transpose = network->addShuffle(*relu->getOutput(0));
    assert(transpose);
    transpose->setName((conv_name + " @outshape").c_str());
    transpose->setSecondTranspose(Permutation{1, 0, 2, 3}); //7*512*28*28

    //BN
    IScaleLayer* bn = temp_addBatchNorm2d(network, weightMap, *transpose->getOutput(0), bn_name, 1e-5);
    assert(bn); //7*512*28*28

    //ADD
    IElementWiseLayer* ew = network->addElementWise(*bn->getOutput(0), input, ElementWiseOperation::kSUM);

    return ew->getOutput(0);
}

/*
 * xception模块
 * input: 7*2048*1
 */
ITensor* xception(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
    //BN
    IScaleLayer* bn = temp_addBatchNorm2d(network, weightMap, input, "xception_bn", 1e-5);
    assert(bn);

    //att_conv
    assert(existsWeightKey(weightMap, "xception_att_conv.b_0"));
    assert(existsWeightKey(weightMap, "xception_att_conv.w_0"));
    IConvolutionLayer* att_conv = network->addConvolution(*bn->getOutput(0), 2048, DimsHW{3, 1}, weightMap["xception_att_conv.w_0"],
            weightMap["xception_att_conv.b_0"]);
    att_conv->setStride(DimsHW{1, 1});
    att_conv->setPadding(DimsHW{1, 0});
    att_conv->setNbGroups(2048);
    assert(att_conv);

    //att_2
    assert(existsWeightKey(weightMap, "xception_att_2.b_0"));
    assert(existsWeightKey(weightMap, "xception_att_2.w_0"));
    IConvolutionLayer* att_2 = network->addConvolution(*att_conv->getOutput(0), 1024, DimsHW{1, 1}, weightMap["xception_att_2.w_0"],
                                                          weightMap["xception_att_2.b_0"]);
    att_2->setStride(DimsHW{1, 1});
    assert(att_2);

    //bndw + relu
    IScaleLayer* bndw = temp_addBatchNorm2d(network, weightMap, *att_2->getOutput(0), "xception_bndw", 1e-5);
    assert(bndw);
    IActivationLayer* relu = network->addActivation(*bndw->getOutput(0), ActivationType::kRELU);
    assert(relu);

    //att1
    assert(existsWeightKey(weightMap, "xception_att1.b_0"));
    assert(existsWeightKey(weightMap, "xception_att1.w_0"));
    IConvolutionLayer* att1 = network->addConvolution(*relu->getOutput(0), 1024, DimsHW{3, 1}, weightMap["xception_att1.w_0"],
                                       weightMap["xception_att1.b_0"]);
    att1->setStride(DimsHW{1, 1});
    att1->setPadding(DimsHW{1, 0});
    att1->setNbGroups(1024);
    assert(att1);

    //att1_2
    assert(existsWeightKey(weightMap, "xception_att1_2.b_0"));
    assert(existsWeightKey(weightMap, "xception_att1_2.w_0"));
    IConvolutionLayer* att1_2 = network->addConvolution(*att1->getOutput(0), 1024, DimsHW{1, 1}, weightMap["xception_att1_2.w_0"],
                                       weightMap["xception_att1_2.b_0"]);
    att1_2->setStride(DimsHW{1, 1});
    assert(att1_2);

    //dw<----bn
    assert(existsWeightKey(weightMap, "xception_dw.b_0"));
    assert(existsWeightKey(weightMap, "xception_dw.w_0"));
    IConvolutionLayer* dw = network->addConvolution(*bn->getOutput(0), 1024, DimsHW{1, 1}, weightMap["xception_dw.w_0"],
                                       weightMap["xception_dw.b_0"]);
    dw->setStride(DimsHW{1, 1});
    assert(dw);

    //add_to = dw+att1_2
    IElementWiseLayer* add_to = network->addElementWise(*dw->getOutput(0), *att1_2->getOutput(0), ElementWiseOperation::kSUM);

    //bn2
    IScaleLayer* bn2 = temp_addBatchNorm2d(network, weightMap, *add_to->getOutput(0), "xception_bn2", 1e-5);
    assert(bn2);

    //relu2
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    return relu2->getOutput(0);
}


// 创建引擎
ICudaEngine* createEngine(IBuilder* builder, DataType dt)
{
    INetworkDefinition* network = builder->createNetwork();

    //load weights
    std::map<std::string, Weights> weightMap = loadWeights(wts_file);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // Create input tensor of shape { 7, 15, 224, 224} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME.c_str(), dt, Dims4{7, 15, 224, 224});
    assert(data);

    // Add convolution layer
    IConvolutionLayer* conv1 = network->addConvolution(*data, 64, DimsHW{7, 7}, weightMap["conv1_weights"], emptywts);
    assert(conv1);
    conv1->setStride(DimsHW{2, 2});
    conv1->setPadding(DimsHW{3, 3});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn_conv1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer* pool1 = network->addPooling(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
    assert(pool1);
    pool1->setStride(DimsHW{2, 2});
    pool1->setPadding(DimsHW{1, 1}); //7*64*56*56

    assert(existsWeightKey(weightMap, "conv1_weights"));
    ITensor* x = bottleneck(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "2a");

    x = bottleneck(network, weightMap, *x, 256, 64, 1, "2b");
    x = bottleneck(network, weightMap, *x, 256, 64, 1, "2c"); //7*256*56*56

    x = bottleneck(network, weightMap, *x, 256, 128, 2, "3a");
    x = bottleneck(network, weightMap, *x, 512, 128, 1, "3b");
    x = bottleneck(network, weightMap, *x, 512, 128, 1, "3c");
    x = bottleneck(network, weightMap, *x, 512, 128, 1, "3d"); //7*512*28*28

    //时域模块
    x = temporalBlock(network, weightMap, *x, 512, "conv3d_0", "batch_norm_24");

    x = bottleneck(network, weightMap, *x, 512, 256, 2, "4a");
    x = bottleneck(network, weightMap, *x, 1024, 256, 1, "4b");
    x = bottleneck(network, weightMap, *x, 1024, 256, 1, "4c");
    x = bottleneck(network, weightMap, *x, 1024, 256, 1, "4d");
    x = bottleneck(network, weightMap, *x, 1024, 256, 1, "4e");
    x = bottleneck(network, weightMap, *x, 1024, 256, 1, "4f");

    //时域模块
    x = temporalBlock(network, weightMap, *x, 1024, "conv3d_1", "batch_norm_44");

    x = bottleneck(network, weightMap, *x, 1024, 512, 2, "5a");
    x = bottleneck(network, weightMap, *x, 2048, 512, 1, "5b");
    x = bottleneck(network, weightMap, *x, 2048, 512, 1, "5c");

    //paddle使用的是全局池化, pool_size = h_in, w_in, 因最后一个x输出长宽为7*7
    IPoolingLayer* pool2 = network->addPooling(*x, PoolingType::kAVERAGE, DimsHW{7, 7});
    assert(pool2);
    pool2->setStride(DimsHW{1, 1}); //7*2048*1*1
    Dims dims1 = pool2->getOutput(0)->getDimensions();

    //reshape和transpose
    nvinfer1::IShuffleLayer* pool_shape = network->addShuffle(*pool2->getOutput(0));
    dims1 = pool_shape->getOutput(0)->getDimensions();
    pool_shape->setReshapeDimensions(Dims3(dims1.d[0], dims1.d[1], dims1.d[2])); //7*2048*1
    pool_shape->setSecondTranspose(Permutation{1,0, 2});  //2048*7*1
    assert(pool_shape);

    //xception
    x = xception(network,  weightMap, *pool_shape->getOutput(0)); //1024*7*1

    IPoolingLayer* pool3 = network->addPooling(*x, PoolingType::kMAX, DimsHW{7, 1});
    pool3->setStride(DimsHW{1, 1}); //1024*1*1
    assert(pool3);

    //fc
    assert(existsWeightKey(weightMap, "fc_0.w_0"));
    assert(existsWeightKey(weightMap, "fc_0.b_0"));
    dims1 = pool3->getOutput(0)->getDimensions();
    std::cout<<"weights count: "<<weightMap["fc_0.w_0"].count<<std::endl;

    //全连接层输入维度至少3层：at least 3 dimensions are required for input
    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool3->getOutput(0), 4, weightMap["fc_0.w_0"], weightMap["fc_0.b_0"]);
    assert(fc1);
    dims1 = pool3->getOutput(0)->getDimensions();


    //softmax
    ISoftMaxLayer* prob = network->addSoftMax(*fc1->getOutput(0));
    assert(prob);
    setOutput(*prob->getOutput(0), network);
    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME.c_str());
    network->markOutput(*prob->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(batchSize); //调节batch的大小
    builder->setMaxWorkspaceSize(1 << 32);

    IBuilderConfig* config = builder->createBuilderConfig();
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
    if (use_fp16){
        config->setFlag(BuilderFlag::kFP16);
    }

    //ICudaEngine* engine = builder->buildCudaEngine(*network);
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

//构造模拟数据
void mock_data(std::unique_ptr<float>& data) {
    std::cout<<"start mocking data..."<<std::endl;
    int dims = INPUT_DIM * batchSize;
    for (int i = 0; i < dims; i++)
        data.get()[i] = 1.0;
}

//真实视频帧
#include <opencv2/opencv.hpp>
int segnum = 7;
int seglen = 5;
int target_size = 224;
void read_data(std::unique_ptr<float>& input_data) {

    std::vector<std::string> file_list;
    //直接从txt中读入列表
    std::ifstream infile;
    infile.open("/home/hl/Downloads/PaddleVideo/test.txt");
    std::string s;
    while(getline(infile,s))
    {
        file_list.push_back(s);
    }
    infile.close();             //关闭文

    //选择35个图片
    int videolen = file_list.size();
    int average_dur = videolen / segnum;
    std::vector<std::string> sel_file_list;
    int idx;
    for(int i=0; i<segnum; i++) {
        idx = 0;
        if (average_dur >= seglen) {
            idx = (average_dur - seglen) / 2;
            idx += i * average_dur;
        } else if (average_dur>=1) {
            idx += i * average_dur;
        } else {
            idx = i;
        }

        for (int j=idx; j<idx+seglen; j++)
            sel_file_list.push_back(file_list[j%videolen]);
    }

    std::cout<<sel_file_list.size()<<std::endl;

    //预处理图片
    cv::Mat img;
    cv::Mat input_channels[3];
    int channelLength = 224 * 224;

    for (int m=0; m<batchSize; m++){
        for (int k=0; k<35; k++) {
            //std::cout<<sel_file_list[k]<<std::endl;
            img = cv::imread("/home/hl/Downloads/PaddleVideo/" + sel_file_list[k]);
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

            //resize(img, img, Size(224, 224)); //=====>resize and crop

            //resize
            int w = img.cols;
            int h = img.rows;
            if ((w <= h && w == target_size) || (h <= w && h == target_size)) {
            } else if (w<h) {
                w = target_size;
                h = (int) target_size * 4.0 / 3.0;
                cv::resize(img, img, cv::Size(w, h));
            } else {
                w = (int) target_size * 4.0 / 3.0;
                h = target_size;
                cv::resize(img, img, cv::Size(w, h));
            }

            //crop
            int x1 = round((w - target_size) / 2.);
            int y1 = round((h - target_size) / 2.);
            img = img(cv::Range(y1, y1+target_size),cv::Range(x1, x1+target_size)); //第一个Range是行，即h， 第二个Range是列，即w
            img.convertTo(img, CV_32FC3);

            cv::split(img, input_channels);
            input_channels[0] = (input_channels[0] / 255.0 - 0.485) / 0.229;
            input_channels[1] = (input_channels[1] / 255.0 - 0.456) / 0.224;
            input_channels[2] = (input_channels[2] / 255.0 - 0.406) / 0.225;

            for (int j = 0; j < 3; j++) {
                memcpy(input_data.get() + (channelLength*3) * k + channelLength * j + INPUT_DIM*m,
                       input_channels[j].data, channelLength * sizeof(float));
            }

        }
    }

}


//打印输出
void print_output(std::unique_ptr<float>& prob){
    int max_idx = 0;
    float max_val = 0.0;
    for (unsigned int j=0; j<batchSize; j++) {
        std::cout << std::endl;
        std::cout << j << " ==== > Output: ";
        for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
        {
            float val = prob.get()[i + j*OUTPUT_SIZE];
            std::cout << val << ", ";
            if (val> max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        std::cout << std::endl;
        std::cout<< "max value: "<< max_val << " with index: "<<max_idx<<std::endl;
    }
}

//测试推理
void test_infer(ICudaEngine& engine, IExecutionContext& context,  std::unique_ptr<float>& data, std::unique_ptr<float>& prob) {
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    std::cout<<INPUT_BLOB_NAME.c_str()<<std::endl;
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME.c_str());
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME.c_str());

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_DIM * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], data.get(), batchSize * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(prob.get(), buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    //打印prob
    print_output(prob);

}

void APIToModel(IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(builder, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    //get context
    IExecutionContext* context = engine->createExecutionContext();

    //测试infer
    std::unique_ptr<float> data(new float[INPUT_DIM * batchSize]);
    mock_data(data);
    std::unique_ptr<float> prob(new float[OUTPUT_SIZE * batchSize]);
    auto start = std::chrono::system_clock::now();
    for (int i=0; i<runIters; i++) {
        test_infer(*engine, *context, data, prob);
    }
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Close everything down
    context->destroy();
    engine->destroy();
    builder->destroy();
}

int main(int argc, char** argv)
{
    std::ifstream file(engine_file, std::ios::binary);
    size_t size{0};
    char *trtModelStream{nullptr};

    if (file.good()) {
        std::cout<<"engine file found, now start inferring ...."<<std::endl;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();

        //初始化引擎
        IRuntime* runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
        assert(engine != nullptr);
        IExecutionContext* context = engine->createExecutionContext();
        assert(context != nullptr);

        //检查batchsize
        int engine_size = engine->getMaxBatchSize();
        if (engine_size != batchSize) {
            std::cout<<"WARNING: ====> "<<batchSize << " is not matched with max batch size of engine "<<engine_size<<std::endl;
            batchSize = engine_size;
        }

        std::unique_ptr<float> data(new float[INPUT_DIM * batchSize]);
        mock_data(data);
        //read_data(data);
        std::unique_ptr<float> prob(new float[OUTPUT_SIZE * batchSize]);
        auto start = std::chrono::system_clock::now();
        for (int i=0; i<runIters; i++) {
            test_infer(*engine, *context, data, prob);
        }
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();


    } else  {
        std::cout<<"no engine file found, now start creating ...."<<std::endl;

        IHostMemory* modelStream{nullptr};
        APIToModel(&modelStream);
        assert(modelStream != nullptr);

        std::ofstream p(engine_file);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());


        modelStream->destroy();
        return 1;

    }

    return 0;
}