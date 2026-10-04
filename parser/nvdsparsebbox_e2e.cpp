// DeepStream bbox parser for end-to-end YOLO models (NMS already in the graph).
// Expects a single output layer shaped [N, 6] per frame: x1, y1, x2, y2, score, class_id,
// in network-input pixels. nvinfer maps the boxes back to the source frame
// (including maintain-aspect-ratio / symmetric-padding), so no rescaling here.

#include <algorithm>
#include <iostream>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYoloE2E(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.size() != 1) {
        std::cerr << "NvDsInferParseYoloE2E: expected 1 output layer, got "
                  << outputLayersInfo.size() << std::endl;
        return false;
    }
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const NvDsInferDims& dims = layer.inferDims;
    if (layer.dataType != FLOAT || dims.numDims < 2 || dims.d[dims.numDims - 1] != 6) {
        std::cerr << "NvDsInferParseYoloE2E: output must be FP32 [N, 6]" << std::endl;
        return false;
    }

    const int numDets = dims.d[dims.numDims - 2];
    const float* det = static_cast<const float*>(layer.buffer);
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    for (int i = 0; i < numDets; ++i, det += 6) {
        const float score = det[4];
        const int cls = static_cast<int>(det[5]);
        if (cls < 0 || cls >= static_cast<int>(detectionParams.numClassesConfigured))
            continue;
        if (score < detectionParams.perClassPreclusterThreshold[cls])
            continue;

        const float x1 = std::clamp(det[0], 0.f, netW);
        const float y1 = std::clamp(det[1], 0.f, netH);
        const float x2 = std::clamp(det[2], 0.f, netW);
        const float y2 = std::clamp(det[3], 0.f, netH);
        if (x2 - x1 < 1.f || y2 - y1 < 1.f)
            continue;

        NvDsInferObjectDetectionInfo obj{};
        obj.classId = cls;
        obj.detectionConfidence = score;
        obj.left = x1;
        obj.top = y1;
        obj.width = x2 - x1;
        obj.height = y2 - y1;
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloE2E);
