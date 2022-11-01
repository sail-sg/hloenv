package(default_visibility = ["//visibility:public"])

cc_library(
    name = "hlo_utils",
    srcs = ["hlo_utils.cc"],
    hdrs = ["hlo_utils.h"],
    deps = [
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@org_tensorflow//tensorflow/compiler/xla:types",
        "@org_tensorflow//tensorflow/compiler/xla:util",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_extractor",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:lib_internal",
    ],
)
