#include "head-segmap.hpp"

#include <numeric>
#include <vector>
#include <cuda_fp16.h>

#include "common/check.hpp"

#include "common/tensorrt.hpp"

static __global__ void apply_threshold_kernel(unsigned int total, const half *logits, float threshold, unsigned char *mask)
{
    int i = cuda_linear_index;

    if (i >= total)
        return;

    float probability = 1.0 / (1.0 + expf(-__half2float(logits[i])));
    mask[i] = probability > threshold ? 1 : 0;
}

namespace bevfusion
{
    namespace head
    {
        namespace segmap
        {

            class SegMapImplement : public SegMap
            {
            public:
                virtual ~SegMapImplement()
                {
                    if (output_)
                        checkRuntime(cudaFree(output_));
                    if (mask_device_)
                        checkRuntime(cudaFree(mask_device_));
                    if (mask_host_)
                        checkRuntime(cudaFree(mask_host_));
                }

                bool init(const SegHeadParameters &param)
                {
                    if (!param.enabled)
                        return false;
                    engine_ = TensorRT::load(param.model);

                    if (engine_ == nullptr)
                        return false;
                    if (engine_->has_dynamic_dim())
                    {
                        printf("Dynamic shapes are not supported for map head.\n");
                        return false;
                    }

                    auto shape = engine_->static_dims("map_logits");
                    Asserts(engine_->dtype("map_logits") == TensorRT::DType::HALF, "Invalid map head output data type.");
                    Asserts(shape.size() == 4 && shape[0] == 1, "Map head output must be NCHW with batch size 1.");
                    view_.classes = shape[1];
                    view_.height = shape[2];
                    view_.width = shape[3];

                    size_t volume = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                    checkRuntime(cudaMalloc(&output_, volume * sizeof(nvtype::half)));
                    view_.data = reinterpret_cast<nvtype::half *>(output_);
                    param_ = param;

                    checkRuntime(cudaMalloc(&mask_device_, volume * sizeof(u_int8_t)));
                    checkRuntime(cudaMalloc(&mask_host_, volume * sizeof(u_int8_t)));

                    return true;
                }

                virtual void print() { engine_->print("Map head"); }

                virtual MapView forward(const nvtype::half *fusion_feature, void *stream)
                {
                    std::vector<const void *> bindings(engine_->num_bindings(), nullptr);

                    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

                    engine_->forward(std::unordered_map<std::string, const void *>{
                                         {"middle", fusion_feature},
                                         {"map_logits", output_},
                                     },
                                     _stream);

                    unsigned int total = view_.width * view_.height * view_.classes;
                    cuda_linear_launch(apply_threshold_kernel, _stream, total,
                                       reinterpret_cast<const half *>(output_),
                                       param_.threshold,
                                       mask_device_);

                    checkRuntime(cudaMemcpyAsync(mask_host_, mask_device_, total, cudaMemcpyDeviceToHost, _stream));
                    checkRuntime(cudaStreamSynchronize(_stream));

                    // Copy the mask back to the CPU
                    view_.masks.assign(mask_host_, mask_host_ + total);
                    return view_;
                }

            private:
                std::shared_ptr<TensorRT::Engine> engine_;
                half *output_ = nullptr;
                unsigned char *mask_device_ = nullptr;
                unsigned char *mask_host_ = nullptr;
                SegHeadParameters param_;
                MapView view_;
            };

            std::shared_ptr<SegMap> create_maphead(const SegHeadParameters &param)
            {
                std::shared_ptr<SegMapImplement> instance(new SegMapImplement());
                if (!instance->init(param))
                    instance.reset();
                return instance;
            }

        }
    }
}
