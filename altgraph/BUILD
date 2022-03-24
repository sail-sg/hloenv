load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension", "tf_cc_binary", "tf_cc_test")
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
        "@com_github_gflags_gflags//:gflags",
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
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla:util",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:lib_internal",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "hlo_graph_test",
    srcs = ["hlo_graph_test.cc"],
    deps = [
        ":hlo_graph",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_matchers",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo_memory_scheduler",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla:test",
        "@org_tensorflow//tensorflow/compiler/xla:util",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla/tests:hlo_test_base",
        "@org_tensorflow//tensorflow/compiler/xla/tests:xla_internal_test_main",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)


pybind_library(
    name = "py_hlo_graph",
    srcs = ["py_hlo_graph.cc"],
    hdrs = ["py_hlo_graph.h"],
    deps = [
        ":hlo_graph",
    ],
)

pybind_extension(
    name = "hlo_ir",
    srcs = ["py_hlo_ir.cc"],
    hdrs = ["py_hlo_ir.h"],
    copts = [
        "-fexceptions",
        "-s",
    ],
    link_in_framework = False,
    linkopts = [
        "-s",
    ],
    module_name = "hlo_ir",
    deps = [
        ":py_hlo_graph",
        "@org_tensorflow//tensorflow/compiler/jit:xla_cpu_jit",
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
        "@org_tensorflow//tensorflow/python:pybind11_lib",
    ],
)

py_library(
    name = "altgraph",
    srcs = ["__init__.py"],
    data = ["hlo_ir.so"],
)
