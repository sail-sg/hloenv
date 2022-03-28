load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@locked_deps//:requirements.bzl", "requirement")

py_test(
    name = "hlo_ir_test",
    srcs = ["hlo_ir_test.py"],
    data = ["hlo_test.txt"],
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
