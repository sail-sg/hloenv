package(default_visibility = ["//visibility:public"])

genrule(
    name = "hlo_ir",
    srcs = ["//altgraph:hlo_ir.so"],
    outs = ["hlo_ir.so"],
    cmd = "cp $< $@",
)

py_library(
    name = "hlo",
    srcs = ["__init__.py"],
    data = [":hlo_ir"],
)
