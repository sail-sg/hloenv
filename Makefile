SHELL=/bin/bash
CPP_FILES = $(shell find hloenv/ -type f -name "*.h" -o -name "*.cc")
ROOT_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

build:
	bazel --output_user_root=/tmp/${USER} run --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com //:setup --verbose_failures
	cp -r /tmp/${USER}/*/execroot/org_hloenv/bazel-out/k8-opt/bin/setup.runfiles/org_hloenv/dist/ ${ROOT_DIR} \

build-debug:
	bazel --output_user_root=/tmp/${USER} run --strip=never --copt="-DNDEBUG" --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com --compilation_mode=dbg //:setup

build-no-remote:
	bazel --output_user_root=/tmp/${USER} run //:setup
	cp -r /tmp/${USER}/*/execroot/org_hloenv/bazel-out/k8-opt/bin/setup.runfiles/org_hloenv/dist/ ${ROOT_DIR} \

build-refresh-remote:
	bazel --output_user_root=/tmp/${USER} run --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com --remote_accept_cached=false //:setup
	cp -r /tmp/${USER}/*/execroot/org_hloenv/bazel-out/k8-opt/bin/setup.runfiles/org_hloenv/dist/ ${ROOT_DIR} \

install:
	pip install --force-reinstall "${ROOT_DIR}/dist/hloenv-0.0.1-cp38-cp38-linux_x86_64.whl"

test:
	bazel --output_user_root=/tmp/${USER} test --test_output=all --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com //tests/...

format: yapf clang-format

lint:
	flake8 hloenv/ --count --show-source --statistics
	cpplint --root=. --recursive hloenv/
	clang-format-11 --style=Google -i ${CPP_FILES} -n -Werror

clean:
	rm -rf /tmp/${USER}
	rm -rf ${HOME}/.cache//bazel
	bazel --output_user_root=/tmp/${USER} clean

yapf:
	$(call check_install, yapf)
	yapf -r -i hloenv/ tests/ examples/

clang-format:
	clang-format-11 --style=Google -i ${CPP_FILES}
