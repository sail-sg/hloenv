#!/usr/bin/env bash
bazel --output_user_root=/tmp/ohcy build --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com //altgraph:hlo_ir --config=monolithic
