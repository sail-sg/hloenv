workspace(name = "org_altgraph")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "third_party",
    branch = "main",
    remote = "https://linmin:HdD2m3njejen6axwztYz@git.insea.io/sail/gameai/third_party.git",
)

# git_repository(
#     name = "org_tensorflow",
#     branch = "14-pybind-interface-to-compilation-loop-2",
#     remote = "https://AltGraph:p2WoxJyV93twzkTEyXvL@git.insea.io/sail/aisys/tensorflow.git",
# )

local_repository(
    name = "org_tensorflow",
    path = "/home/aiops/ohcy/tf_graph/tensorflow",
)

http_archive(
  name = "pybind11_bazel",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip"],
  sha256 = "a5666d950c3344a8b0d3892a88dc6b55c8e0c78764f9294e806d69213c03f19d",
  strip_prefix = "pybind11_bazel-26973c0ff320cb4b39e45bc3e4297b82bc3a6c09",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

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
