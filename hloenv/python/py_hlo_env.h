// Copyright 2022 Garena Online Private Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HLOENV_PYTHON_PY_HLO_ENV_H_
#define HLOENV_PYTHON_PY_HLO_ENV_H_

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "hloenv/hlo_env.h"
#include "hloenv/python/py_hlo_graph.h"
#include "hloenv/python/py_hlo_passes.h"

namespace py = pybind11;

namespace hloenv {

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

  PyHloGraph GetHloGraph(bool debug, bool inline_fused_comp,
                         bool do_hash_verification) {
    return PyHloGraph(GetHloModule()->hlo_module_ptr(), debug,
                      inline_fused_comp, do_hash_verification);
  }
};

}  // namespace hloenv

#endif  // HLOENV_PYTHON_PY_HLO_ENV_H_
