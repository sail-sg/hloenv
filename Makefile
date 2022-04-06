SHELL=/bin/bash
CPP_FILES = $(shell find altgraph/ -type f -name "*.h" -o -name "*.cc")
ROOT_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

clean:
	rm -rf /localfolder/${USER}
	rm -rf ${HOME}/.cache//bazel
	bazel --output_user_root=/localfolder/${USER} clean

build:
	bazel --output_user_root=/localfolder/${USER} run --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com //:setup
	cp -r /localfolder/${USER}/*/execroot/org_altgraph/bazel-out/k8-opt/bin/setup.runfiles/org_altgraph/dist/ ${ROOT_DIR} \

install:
	pip install --force-reinstall "${ROOT_DIR}/dist/altgraph-0.0.1-cp38-cp38-linux_x86_64.whl"

test:
	bazel --output_user_root=/tmp/${USER} test --test_output=all --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com //tests/...

format: yapf clang-format

lint:
	flake8 altgraph/ --count --show-source --statistics
	cpplint --root=. --recursive altgraph/
	clang-format-11 --style=Google -i ${CPP_FILES} -n -Werror

yapf:
	$(call check_install, yapf)
	yapf -r -i altgraph/ tests/

clang-format:
	clang-format-11 --style=Google -i ${CPP_FILES}
