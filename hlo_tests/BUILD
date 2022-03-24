load("@locked_deps//:requirements.bzl", "requirement")

py_test(
    name = "hlo_ir_test",
    srcs = ["hlo_ir_test.py"],
    data = ["hlo_test.txt"],
    deps = [
        "//hlo",
        requirement("absl-py"),
    ],
)
