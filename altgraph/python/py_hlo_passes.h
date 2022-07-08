// Copyright 2021 Garena Online Private Limited

#ifndef ALTGRAPH_PYTHON_PY_HLO_PASSES_H_
#define ALTGRAPH_PYTHON_PY_HLO_PASSES_H_

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "altgraph/gpu_backend.h"
#include "altgraph/hlo_module.h"
#include "altgraph/hlo_pass_defs.h"

namespace py = pybind11;

void py_init_hlo_passes(const py::module& m);

#endif  // ALTGRAPH_PYTHON_PY_HLO_PASSES_H_
