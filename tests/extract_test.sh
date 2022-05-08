#!/usr/bin/env bash

find tests/hlo_texts -name "*.txt" | xargs -I{} sh -c "$1 --hlo {} || exit 255"
