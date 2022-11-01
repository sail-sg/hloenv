# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@third_party//setup:setup.bzl", "build_wheel")
load("@locked_deps//:requirements.bzl", "requirement")

py_binary(
    name = "setup",
    srcs = [
        "@third_party//setup:setup.py",
    ],
    args = [
        "--main-pkg",
        "altgraph",
        "--name",
        "altgraph",
        "--bdist",
        "--release",
    ],
    data = [
        "setup.cfg",
        "//altgraph",
    ],
    main = "@third_party//setup:setup.py",
    python_version = "PY3",
    deps = [
        "//altgraph",
        requirement("setuptools"),
    ],
)
