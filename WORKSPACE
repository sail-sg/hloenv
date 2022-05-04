workspace(name = "org_altgraph")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    sha256 = "954aa89b491be4a083304a2cb838019c8b8c3720a7abb9c4cb81ac7a24230cea",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_python/releases/download/0.4.0/rules_python-0.4.0.tar.gz",
        "https://github.com/bazelbuild/rules_python/releases/download/0.4.0/rules_python-0.4.0.tar.gz",
    ],
)

load("@rules_python//python:pip.bzl", "pip_install")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "third_party",
    branch = "main",
    remote = "https://linmin:HdD2m3njejen6axwztYz@git.insea.io/sail/gameai/third_party.git",
)

load("@third_party//py:workspace0.bzl", workspace0 = "workspace")

workspace0()

load("@third_party//py:workspace1.bzl", workspace1 = "workspace")

workspace1()

git_repository(
    name = "org_tensorflow",
    branch = "main",
    remote = "https://AltGraph:p2WoxJyV93twzkTEyXvL@git.insea.io/sail/aisys/tensorflow.git",
)

# local_repository(
#     name = "org_tensorflow",
#     path = "/home/aiops/ohcy/tf_graph/tensorflow",
# )

git_repository(
    name = "jax",
    remote = "https://github.com/google/jax.git",
    tag = "jaxlib-v0.3.0",
)


http_archive(
  name = "com_google_googletest",
  urls = ["https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip"],
  strip_prefix = "googletest-609281088cfefc76f9d0ce82e1ff6c30cc3591e5",
)

http_archive(
    name = "pocketfft",
    build_file = "@jax//third_party/pocketfft:BUILD.bazel",
    sha256 = "66eda977b195965d27aeb9d74f46e0029a6a02e75fbbc47bb554aad68615a260",
    strip_prefix = "pocketfft-f800d91ba695b6e19ae2687dd60366900b928002",
    urls = [
        "https://github.com/mreineck/pocketfft/archive/f800d91ba695b6e19ae2687dd60366900b928002.tar.gz",
        "https://storage.googleapis.com/jax-releases/mirror/pocketfft/pocketfft-f800d91ba695b6e19ae2687dd60366900b928002.tar.gz",
    ],
)

http_archive(
    name = "pybind11_bazel",
    sha256 = "a5666d950c3344a8b0d3892a88dc6b55c8e0c78764f9294e806d69213c03f19d",
    strip_prefix = "pybind11_bazel-26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip"],
)

# ***************************************************************
# TensorFlow Initialization
# ***************************************************************

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")

workspace()

# Initialize bazel package rules' external dependencies.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

rules_pkg_dependencies()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()
