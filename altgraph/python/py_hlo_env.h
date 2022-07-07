// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_PYTHON_PY_HLO_ENV_H_
#define ALTGRAPH_PYTHON_PY_HLO_ENV_H_

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "altgraph/hlo_env.h"
#include "altgraph/python/py_hlo_graph.h"
#include "altgraph/python/py_hlo_passes.h"

namespace py = pybind11;

namespace altgraph {

class PyHloEnv : public HloEnv {
 public:
  PyHloEnv(std::shared_ptr<AltHloModule> alt_hlo_module,
           const std::string& platform)
      : HloEnv(alt_hlo_module, platform) {}

  PyHloEnv(const std::string& hlo_input, const std::string& format,
           const std::string& platform)
      : HloEnv(hlo_input, format, platform) {}

  PyHloEnv(const std::string& hlo_filepath, const std::string& platform)
      : HloEnv(hlo_filepath, "path", platform) {}

  PyHloGraph GetHloGraph(bool inline_fused_comp, bool do_hash_verification) {
    return PyHloGraph(GetHloModule()->hlo_module_ptr(), inline_fused_comp,
                      do_hash_verification);
  }
};

}  // namespace altgraph

#endif  // ALTGRAPH_PYTHON_PY_HLO_ENV_H_
