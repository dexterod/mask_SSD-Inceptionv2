//
// Created by hrh on 2019-09-02.
//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include <cJSON.h>
#include <sys/stat.h>
#include "SampleDetector.hpp"
#include "ji_utils.h"

#include <string>
#include <memory>
#include <vector>
#include <map>

using namespace InferenceEngine;

/**
 * @brief Gets filename without extension
 * @param filepath - full file name
 * @return filename without extension
 */
static std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

SampleDetector::SampleDetector(double thresh) : mThresh(thresh) {
    LOG(INFO) << "Current config: thresh:" << mThresh;
}

int SampleDetector::init(const char *modelXMLPath) {
    // 在此处添加需要检测的目标名称
    mIDNameMap.insert(std::make_pair<int, std::string>(1, "mask"));
    mIDNameMap.insert(std::make_pair<int, std::string>(2, "head"));
    mIDNameMap.insert(std::make_pair<int, std::string>(3, "back"));
    mIDNameMap.insert(std::make_pair<int, std::string>(4, "mid_mask"));

    // 判断模型文件是否存在，并加载模型
    if (modelXMLPath == nullptr) {
        LOG(ERROR) << "Invalid init args!";
        return ERROR_INVALID_INIT_ARGS;
    }
    struct stat st;
    if (stat(modelXMLPath, &st) != 0) {
        LOG(ERROR) << modelXMLPath << " not found!";
        return ERROR_INVALID_INIT_ARGS;
    }
    LOG(INFO) << "Loading model...";
    Core ie;
    std::string binFileName = fileNameNoExt(modelXMLPath) + ".bin";
    if (stat(binFileName.c_str(), &st) != 0) {
        LOG(ERROR) << binFileName << " not found!";
        return ERROR_INVALID_INIT_ARGS;
    }
    LOG(INFO) << "Loading network files:"
                  "\n\t" << modelXMLPath << "\n\t" << binFileName;
    mNetwork = ie.ReadNetwork(modelXMLPath);

    // 获取输入输出信息
    LOG(INFO) << "Preparing input blobs";
    /** Taking information about all topology inputs **/
    InputsDataMap inputsInfo(mNetwork.getInputsInfo());
    /** SSD network has one input and one output **/
    if (inputsInfo.size() != 1 && inputsInfo.size() != 2) throw std::logic_error("Sample supports topologies only with 1 or 2 inputs");

    SizeVector inputImageDims;
    /** Iterating over all input blobs **/
    for (auto & item : inputsInfo) {
        /** Working with first input tensor that stores image **/
        mImageInputName = item.first;
        mInputInfo = item.second;

        LOG(INFO) << "Batch size is " << std::to_string(mNetwork.getBatchSize());
        /** Creating first input blob **/
        Precision inputPrecision = Precision::U8;
        item.second->setPrecision(inputPrecision);
    }
    if (mInputInfo == nullptr) {
        mInputInfo = inputsInfo.begin()->second;
    }

    LOG(INFO) << "Preparing output blobs";
    OutputsDataMap outputsInfo(mNetwork.getOutputsInfo());

    for (const auto& out : outputsInfo) {
        if (out.second->getCreatorLayer().lock()->type == "DetectionOutput") {
            mOutputName = out.first;
            mOutputInfo = out.second;
        }
    }

    if (mOutputInfo == nullptr) {
        throw std::logic_error("Can't find a DetectionOutput layer in the topology");
    }
    mOutputDims = mOutputInfo->getTensorDesc().getDims();
    mMaxProposalCount = mOutputDims[2];
    mObjectSize = mOutputDims[3];

    if (mObjectSize != 7) {
        throw std::logic_error("Output item should have 7 as a last dimension");
    }
    if (mOutputDims.size() != 4) {
        throw std::logic_error("Incorrect output dimensions for SSD model");
    }
    /** Set the precision of output data provided by the user, should be called before load of the network to the device **/
    mOutputInfo->setPrecision(Precision::FP32);

    LOG(INFO) << "Loading model to the device";
    mExecutableNetwork = ie.LoadNetwork(mNetwork, "CPU", {});
    LOG(INFO) << "Done.";
    return SampleDetector::INIT_OK;
}

void SampleDetector::unInit() {
}

STATUS SampleDetector::processImage(const cv::Mat &cv_image, std::vector<Object> &result) {
    if (cv_image.empty()) {
        LOG(ERROR) << "Invalid input!";
        return ERROR_INVALID_INPUT;
    }
    LOG(INFO) << "Create infer request";
    InferRequest infer_request = mExecutableNetwork.CreateInferRequest();

    std::vector<std::shared_ptr<unsigned char>> imagesData; // 最终输入到推理引擎的数据
    std::vector<size_t> imageWidths, imageHeights;

    imageWidths.push_back(cv_image.cols);
    imageHeights.push_back(cv_image.rows);
    LOG(INFO) << "Input image size:" << cv_image.size();
    size_t original_size = cv_image.size().width * cv_image.size().height * cv_image.channels();
    std::shared_ptr<unsigned char > pOriginalData;
    pOriginalData.reset(new unsigned char[original_size], std::default_delete<unsigned char[]>());
    for (size_t id = 0; id < original_size; ++id) {
        pOriginalData.get()[id] = cv_image.data[id];
    }

    // Resize图像
    cv::Mat resized_img(cv_image);
    cv::resize(cv_image, resized_img,
               cv::Size(mInputInfo->getTensorDesc().getDims()[3], mInputInfo->getTensorDesc().getDims()[2]));
    std::shared_ptr<unsigned char> pImageData;
    size_t resized_size = resized_img.size().width * resized_img.size().height * resized_img.channels();
    pImageData.reset(new unsigned char[resized_size], std::default_delete<unsigned char[]>());
    for (size_t id = 0; id < resized_size; ++id) {
        pImageData.get()[id] = resized_img.data[id];
    }
    imagesData.push_back(pImageData);
    if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

    size_t batchSize = mNetwork.getBatchSize();
    LOG(INFO) << "Batch size is " << std::to_string(batchSize);
    if (batchSize != imagesData.size()) {
        LOG(WARNING) << "Number of images " + std::to_string(imagesData.size()) + \
                " doesn't match batch size " + std::to_string(batchSize);
        batchSize = std::min(batchSize, imagesData.size());
        LOG(WARNING) << "Number of images to be processed is "<< std::to_string(batchSize);
    }

    // 将Resize之后的cv_image填充到输入数据变量中
    Blob::Ptr imageInput = infer_request.GetBlob(mImageInputName);
    /** Filling input tensor with images. First b channel, then g and r channels **/
    MemoryBlob::Ptr mimage = as<MemoryBlob>(imageInput);
    if (!mimage) {
        LOG(ERROR) << "We expect image blob to be inherited from MemoryBlob, but by fact we were not able "
                     "to cast imageInput to MemoryBlob";
        return 1;
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto minputHolder = mimage->wmap();

    size_t num_channels = mimage->getTensorDesc().getDims()[1];
    size_t image_size = mimage->getTensorDesc().getDims()[3] * mimage->getTensorDesc().getDims()[2];

    auto *data = minputHolder.as<unsigned char *>();
    /** Iterate over all input images **/
    for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
        /** Iterate over all pixel in image (b,g,r) **/
        for (size_t pid = 0; pid < image_size; pid++) {
            /** Iterate over all channels **/
            for (size_t ch = 0; ch < num_channels; ++ch) {
                /**          [images stride + channels stride + pixel id ] all in bytes            **/
                data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + ch];
            }
        }
    }

    // 推理
    LOG(INFO) << "Start inference";
    infer_request.Infer();
    LOG(INFO) << "Processing output blobs";

    // 获取输出结果
    const Blob::Ptr output_blob = infer_request.GetBlob(mOutputName);
    MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
    if (!moutput) {
        throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                               "but by fact we were not able to cast output to MemoryBlob");
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto moutputHolder = moutput->rmap();
    const float *detection = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();

    /* Each detection has image_id that denotes processed image */
    for (int curProposal = 0; curProposal < mMaxProposalCount; curProposal++) {
        float confidence = detection[curProposal * mObjectSize + 2];
        if (confidence < mThresh) {
            continue;
        }

        auto image_id = static_cast<int>(detection[curProposal * mObjectSize + 0]);
        if (image_id < 0) {
            break;
        }
        auto label = static_cast<int>(detection[curProposal * mObjectSize + 1]);
        auto xmin = static_cast<int>(detection[curProposal * mObjectSize + 3] * imageWidths[image_id]);
        auto ymin = static_cast<int>(detection[curProposal * mObjectSize + 4] * imageHeights[image_id]);
        auto xmax = static_cast<int>(detection[curProposal * mObjectSize + 5] * imageWidths[image_id]);
        auto ymax = static_cast<int>(detection[curProposal * mObjectSize + 6] * imageHeights[image_id]);

        LOG(INFO)<< "[" << curProposal << "," << label << "] element, prob = " << confidence <<
                  "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << image_id;

        result.emplace_back(SampleDetector::Object({
            confidence,
            mIDNameMap[label],
            cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin)
        }));
    }

    return SampleDetector::PROCESS_OK;
}

bool SampleDetector::setThresh(double thresh) {
    mThresh = thresh;
    return true;
}
