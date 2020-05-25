//
// Created by hrh on 2019-09-02.
//

#ifndef JI_SAMPLEDETECTOR_HPP
#define JI_SAMPLEDETECTOR_HPP
#include <string>
#include <opencv2/core/mat.hpp>
#include <inference_engine.hpp>
#include <map>

#define STATUS int

using namespace InferenceEngine;

/**
 * 使用OpenVINO转换的行人检测模型，模型基于ssd inception v2 coco训练得到，模型转换请参考：
 * https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html#ssd_single_shot_multibox_detector_topologies
 */

class SampleDetector {

public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;

    SampleDetector(double thresh);

    /**
     * 初始化模型
     * @param[in] modelXMLPath OpenVINO IR模型的XML文件路径
     * @return 如果初始化正常，INIT_OK
     */
    STATUS init(const char *modelXMLPath);

    /**
     * 反初始化函数
     */
    void unInit();

    /**
     * 对cv::Mat格式的图片进行分类，并输出预测分数前top排名的目标名称到mProcessResult
     * @param[in] image 输入图片
     * @param[out] detectResults 检测到的结果
     * @return 如果处理正常，则返回PROCESS_OK，否则返回`ERROR_*`
     */
    STATUS processImage(const cv::Mat &image, std::vector<Object> &detectResults);

    bool setThresh(double thresh);


public:
    static const int ERROR_BASE = 0x0100;
    static const int ERROR_INVALID_INPUT = 0x0101;
    static const int ERROR_INVALID_INIT_ARGS = 0x0102;

    static const int PROCESS_OK = 0x1001;
    static const int INIT_OK = 0x1002;

private:
    ExecutableNetwork mExecutableNetwork;
    CNNNetwork mNetwork;

    InputInfo::Ptr mInputInfo{nullptr};
    DataPtr mOutputInfo{nullptr};
    SizeVector mOutputDims;
    std::string mImageInputName;
    std::string mOutputName;
    int mMaxProposalCount{0};
    int mObjectSize{0};

    double mThresh = 0.5;

    std::map<int, std::string> mIDNameMap;
};

#endif //JI_SAMPLEDETECTOR_HPP
