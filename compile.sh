#!/usr/bin/env bash
bazel --output_user_root=/tmp/altgraph run --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com //:setup
