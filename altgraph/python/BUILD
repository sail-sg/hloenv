load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

pybind_library(
    name = "py_hlo_graph",
    srcs = ["py_hlo_graph.h"],
    deps = [
        "//altgraph:hlo_graph",
    ],
)

pybind_library(
    name = "py_hlo_pass",
    srcs = ["py_hlo_passes.cc"],
    hdrs = ["py_hlo_passes.h"],
    deps = [
        "//altgraph:gpu_backend",
        "//altgraph:hlo_module",
        "//altgraph:hlo_pass_defs",
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
        "//altgraph:hlo_env",
    ],
)
