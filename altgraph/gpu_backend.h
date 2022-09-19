// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_GPU_BACKEND_H_
#define ALTGRAPH_GPU_BACKEND_H_

#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"

namespace altgraph {

struct HloEnvGpuBackend {
  static HloEnvGpuBackend& Instance() {
    static HloEnvGpuBackend s;
    return s;
  }  // instance

  HloEnvGpuBackend(const HloEnvGpuBackend&) = delete;
  HloEnvGpuBackend& operator=(const HloEnvGpuBackend&) = delete;

  static xla::PjRtClient* PjRtClient() { return Instance().client.get(); }
  static xla::gpu::GpuCompiler* GpuCompiler() { return Instance().compiler; }
  static xla::se::StreamExecutor* StreamExecutor() {
    return Instance().stream_exec;
  }
  static xla::se::DeviceMemoryAllocator* DeviceMemoryAllocator() {
    return Instance().device_allocator;
  }

  // TODO(ohcy): Consider pybinding StreamExec, StreamExec::Platform
  // and StreamExec::CudaComputeCapability if more of these hooks are needed.
  static const std::string& GetStreamExecPlatform() {
    return StreamExecutor()->platform()->Name();
  }
  static bool CudaComputeIsAtLeast(int other_major, int other_minor = 0) {
    return StreamExecutor()
        ->GetDeviceDescription()
        .cuda_compute_capability()
        .IsAtLeast(other_major, other_minor);
  }

 private:
  std::unique_ptr<xla::PjRtClient> client;
  xla::gpu::GpuCompiler* compiler;
  xla::se::StreamExecutor* stream_exec;
  xla::se::DeviceMemoryAllocator* device_allocator;

  HloEnvGpuBackend() {
    std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
        [](xla::HloModuleConfig* config) { config->set_seed(42); };

    // Dummy empty Hlo Module
    const std::string dummy_hlo_txt =
        "HloModule dummy\nENTRY main.1 {ROOT Arg_0.1 = s32[] parameter(0)}";
    std::unique_ptr<xla::HloModule> dummy_hlo_module =
        LoadModuleFromData(dummy_hlo_txt, "txt",
                           xla::hlo_module_loader_details::Config(),
                           config_modifier_hook)
            .ValueOrDie();

    xla::GpuAllocatorConfig gpu_allocator_config = xla::GpuAllocatorConfig();
    // TODO(ohcy): Allow user to specify memory_fraction? I feel since
    // we only have a single allocator, just leave to defaults for now.
    // gpu_allocator_config.memory_fraction = memory_fraction;
    // gpu_allocator_config.preallocate = preallocate;

    client = xla::GetGpuClient(/*asynchronous=*/true, gpu_allocator_config,
                               nullptr, 0)
                 .ValueOrDie();

    // Compile XlaComputation to PjRtExecutable.
    const xla::HloModuleProto hlo_module_proto = dummy_hlo_module->ToProto();
    xla::XlaComputation xla_computation(hlo_module_proto);
    xla::CompileOptions compile_options;
    try {
      std::unique_ptr<xla::PjRtExecutable> executable =
          client->Compile(xla_computation, compile_options).ValueOrDie();
    } catch (xla::Intercept<xla::gpu::GpuCompiler>& gpu_intercept) {
      compiler = gpu_intercept.compiler;
      stream_exec = gpu_intercept.stream_exec;
      device_allocator = gpu_intercept.options.device_allocator;
    }
  }
  ~HloEnvGpuBackend() {}
};  // struct HloEnvGpuBackend

}  // namespace altgraph

#endif  // ALTGRAPH_GPU_BACKEND_H_
