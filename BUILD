load("@third_party//setup:setup.bzl", "build_wheel")
load("@locked_deps//:requirements.bzl", "requirement")

py_binary(
    name = "setup",
    srcs = [
        "@third_party//setup:setup.py",
    ],
    args = [
        "--main-pkg",
        "hlo",
        "--name",
        "hlo",
        "--bdist",
    ],
    data = [
        "setup.cfg",
    ],
    main = "@third_party//setup:setup.py",
    python_version = "PY3",
    deps = [
        "//hlo",
        requirement("setuptools"),
    ],
)
