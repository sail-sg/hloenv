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

#ifndef HLOENV_PYTHON_PY_HLO_PASSES_H_
#define HLOENV_PYTHON_PY_HLO_PASSES_H_

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "hloenv/gpu_backend.h"
#include "hloenv/hlo_module.h"
#include "hloenv/hlo_pass_defs.h"

namespace py = pybind11;

namespace hloenv {

void py_init_hlo_passes(const py::module& m);

}  // namespace hloenv

#endif  // HLOENV_PYTHON_PY_HLO_PASSES_H_
