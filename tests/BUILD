load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@locked_deps//:requirements.bzl", "requirement")

py_test(
    name = "hlo_ir_test",
    srcs = ["hlo_ir_test.py"],
    data = [":test_data"],
    deps = [
        "//altgraph",
        requirement("absl-py"),
    ],
)

tf_cc_test(
    name = "hlo_graph_test",
    srcs = ["hlo_graph_test.cc"],
    deps = [
        "//altgraph:hlo_graph",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@com_google_googletest//:gtest_main",
    ],
    linkstatic = 1,
)

tf_cc_test(
    name = "evaluator_test",
    srcs = ["evaluator_test.cc"],
    data = [
        ":test_data",
    ],
    deps = [
        "//altgraph/evaluation:evaluator",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:cpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
        "@com_google_googletest//:gtest_main",
    ],
    linkstatic = 1,
)

filegroup(
    name = "test_data",
    srcs = glob(["*.txt"]),
)
