package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "extract_subgraphs",
    srcs = ["extract_subgraphs.cc"],
    deps = [
        "//altgraph/utils:hlo_utils",
        "@com_github_gflags_gflags//:gflags",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
    ],
)
