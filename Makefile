SHELL=/bin/bash
LINT_PATHS=${PROJECT_PATH}
CPP_FILES = $(shell find altgraph/ -type f -name "*.h" -o -name "*.cc")

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

yapf:
	$(call check_install, yapf)
	yapf -r -i hlo/ hlo_tests/ altgraph/

clang-format:
	clang-format-11 --style=Google -i ${CPP_FILES}
ca
