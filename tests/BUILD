load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load("@locked_deps//:requirements.bzl", "requirement")

py_test(
    name = "hlo_env_test",
    srcs = ["hlo_env_test.py"],
    data = [":test_data"],
    deps = [
        "//altgraph",
        requirement("absl-py"),
    ],
)

tf_cc_test(
    name = "hlo_graph_test",
    srcs = ["hlo_graph_test.cc"],
    linkstatic = 1,
    deps = [
        "//altgraph:hlo_graph",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/compiler/xla:literal",
        "@org_tensorflow//tensorflow/compiler/xla:literal_util",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
    ],
)

tf_cc_test(
    name = "evaluator_test",
    srcs = ["evaluator_test.cc"],
    data = [
        ":test_data",
    ],
    linkstatic = 1,
    deps = [
        "//altgraph/evaluation:evaluator",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt:cpu_device",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
        "@org_tensorflow//tensorflow/compiler/xla/tools:hlo_module_loader",
    ],
)

tf_cc_test(
    name = "cycle_detection_test",
    srcs = ["cycle_detection_test.cc"],
    linkstatic = 1,
    deps = [
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/compiler/xla/service:computation_placer",
        "@org_tensorflow//tensorflow/compiler/xla/service:hlo",
    ],
)


filegroup(
    name = "test_data",
    srcs = glob([
        "*.txt",
        "hlo_texts/*.txt",
        "hlo_texts/test_hlos/**/*.txt",
    ]),
)

sh_test(
    name = "extract_test",
    srcs = ["extract_test.sh"],
    args = [
        "$(rootpath //altgraph/bin:extract_subgraphs)",
    ],
    data = [
        ":test_data",
        "//altgraph/bin:extract_subgraphs",
    ],
)
