load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension", "tf_cc_binary")
load("@pybind11_bazel//:build_defs.bzl", "pybind_library")

package(default_visibility = ["//visibility:public"])

tf_cc_binary(
    name = "xla_compile",
    srcs = ["xla_compile.cc"],
    copts = [
        "-fexceptions",
    ],
    deps = [
        ":hlo_graph",
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:status",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:cpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:gpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
        "@com_github_gflags_gflags//:gflags",
    ],
)

cc_library(
    name = "hlo_graph",
    srcs = ["hlo_graph.cc"],
    hdrs = ["hlo_graph.h"],
    deps = [
        "//altgraph/utils:hlo_utils",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla:util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:lib_internal",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

pybind_library(
    name = "py_hlo_graph",
    srcs = ["py_hlo_graph.h"],
    deps = [
        ":hlo_graph",
    ],
)

pybind_library(
    name = "py_hlo_module",
    srcs = ["py_hlo_module.h"],
    deps = [
        "//altgraph/utils:hlo_utils",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
    ],
)

pybind_extension(
    name = "hlo_env",
    srcs = ["py_hlo_env.cc"],
    hdrs = ["py_hlo_env.h"],
    copts = [
        "-fexceptions",
    ],
    link_in_framework = False,
    linkopts = [
    ],
    module_name = "hlo_env",
    deps = [
        ":py_hlo_graph",
        ":py_hlo_module",
        "//altgraph/evaluation:evaluator",
        "@com_google_absl//absl/hash",
        "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
        "@org_tensorflow//tensorflow/compiler/jit:xla_gpu_jit",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:literal_comparison",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:status",
        "@org_tensorflow//tensorflow/compiler/xla:statusor",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:cpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:gpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@org_tensorflow//tensorflow/compiler/xla/python:types",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
        "@org_tensorflow//tensorflow/python:pybind11_lib",
    ],
)

genrule(
    name = "gen_extract_subgraphs",
    srcs = ["//altgraph/bin:extract_subgraphs"],
    outs = ["extract_subgraphs"],
    cmd = "cp $< $@ && chmod 755 $@",
)

py_library(
    name = "altgraph",
    srcs = ["__init__.py"],
    data = [
        "hlo_env.so",
        ":gen_extract_subgraphs",
    ],
)

