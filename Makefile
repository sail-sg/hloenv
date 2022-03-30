SHELL=/bin/bash
LINT_PATHS=${PROJECT_PATH}
CPP_FILES = $(shell find altgraph/ -type f -name "*.h" -o -name "*.cc")

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

build:
	bazel --output_user_root=/tmp/altgraph run --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com //:setup

install: build
	pip install --force-reinstall "bazel-bin/setup.runfiles/org_altgraph/dist/altgraph-0.0.1-cp38-cp38-linux_x86_64.whl"

test:
	bazel --output_user_root=/tmp/altgraph test --test_output=all --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com //tests/...

yapf:
	$(call check_install, yapf)
	yapf -r -i altgraph/ tests/

clang-format:
	clang-format-11 --style=Google -i ${CPP_FILES}
