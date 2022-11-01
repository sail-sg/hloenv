# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

pybind_library(
    name = "py_hlo_graph",
    srcs = ["py_hlo_graph.h"],
    deps = [
        "//hloenv:hlo_graph",
    ],
)

pybind_library(
    name = "py_hlo_pass",
    srcs = ["py_hlo_passes.cc"],
    hdrs = ["py_hlo_passes.h"],
    deps = [
        "//hloenv:gpu_backend",
        "//hloenv:hlo_module",
        "//hloenv:hlo_pass_defs",
    ],
)

pybind_extension(
    name = "py_hlo_env",
    srcs = ["py_hlo_env.cc"],
    hdrs = ["py_hlo_env.h"],
    copts = [
        "-fexceptions",
    ],
    link_in_framework = False,
    linkopts = [
    ],
    module_name = "py_hlo_env",
    visibility = ["//visibility:public"],
    deps = [
        ":py_hlo_graph",
        ":py_hlo_pass",
        "//hloenv:hlo_env",
    ],
)
