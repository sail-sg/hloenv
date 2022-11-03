SHELL=/bin/bash
PROJECT_NAME = hloenv
CPP_FILES = $(shell find hloenv/ -type f -name "*.h" -o -name "*.cc")
ROOT_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
VERSION = 0.0.1
check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

go-install:
	# requires go >= 1.16
	command -v go || (sudo apt-get install -y golang-1.16 && sudo ln -sf /usr/lib/go-1.16/bin/go /usr/bin/go)

bazel-install: go-install
	command -v bazel || (go install github.com/bazelbuild/bazelisk@latest && ln -sf $(HOME)/go/bin/bazelisk $(HOME)/go/bin/bazel)

build: bazel-install
	bazel run //:setup -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/org_$(PROJECT_NAME)/dist/*.whl ./dist

build-debug: bazel-install
	bazel run --strip=never --copt="-DNDEBUG" --compilation_mode=dbg //:setup -- bdist_wheel
	mkdir -p dist
	cp bazel-bin/setup.runfiles/org_$(PROJECT_NAME)/dist/*.whl ./dist

clean: bazel-install
	bazel clean --expunge

doc-install:
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)

doc: doc-install
	cd docs && make html && cd _build/html && python3 -m http.server

install:
	python3 -m pip install --force-reinstall "${ROOT_DIR}/dist/hloenv-${VERSION}-cp38-cp38-linux_x86_64.whl"

test:
	bazel test --test_output=all //tests/...

format: yapf clang-format

lint:
	flake8 hloenv/ --count --show-source --statistics
	cpplint --root=. --recursive hloenv/
	clang-format-11 --style=Google -i ${CPP_FILES} -n -Werror

yapf:
	$(call check_install, yapf)
	yapf -r -i hloenv/ tests/ examples/

clang-format:
	clang-format-11 --style=Google -i ${CPP_FILES}
