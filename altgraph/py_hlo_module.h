// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_PY_HLO_MODULE_H_
#define ALTGRAPH_PY_HLO_MODULE_H_

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <utility>

#include "absl/hash/hash.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/tools/hlo_module_loader.h"

namespace py = pybind11;

class PyHloModule {
 public:
  explicit PyHloModule(std::unique_ptr<xla::HloModule> hlo_module) {
    hlo_module_ = std::move(hlo_module);
  }
  explicit PyHloModule(const std::string& hlo_filepath) {
    std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
        [](xla::HloModuleConfig* config) { config->set_seed(42); };

    hlo_module_ =
        std::move(LoadModuleFromFile(hlo_filepath,
                                     xla::hlo_module_loader_details::Config(),
                                     "txt", config_modifier_hook)
                      .ValueOrDie());
  }
  explicit PyHloModule(PyHloModule&& other) {
    hlo_module_ = std::move(other.hlo_module_);
  }
  PyHloModule& operator=(PyHloModule&& other) {
    if (this != &other) {
      hlo_module_ = std::move(other.hlo_module_);
    }
    return *this;
  }

  virtual ~PyHloModule() {}

  xla::HloModule* hlo_module_ptr() { return hlo_module_.get(); }

  std::shared_ptr<PyHloModule> Clone() {
    std::unique_ptr<xla::HloModule> saved_hlo_module = std::move(hlo_module_);
    hlo_module_ = std::move(saved_hlo_module->Clone());
    return std::make_shared<PyHloModule>(std::move(saved_hlo_module));
  }

  std::string ToString() { return hlo_module_->ToString(); }

  uint64_t Hash() { return absl::HashOf(*hlo_module_); }

  xla::HloModuleProto ToProto() { return hlo_module_->ToProto(); }

 private:
  std::unique_ptr<xla::HloModule> hlo_module_;
};

#endif  // ALTGRAPH_PY_HLO_MODULE_H_
