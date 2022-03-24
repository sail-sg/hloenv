rm ./bazel-bin/setup.runfiles/org_altgraph/dist/*.whl
rm ./dist/*.whl
bazel --output_user_root=/tmp/ohcy build --remote_cache=http://bazel-cache-http.ai.seacloud.garenanow.com //:setup --config=monolithic
bazel --output_user_root=/tmp/ohcy run //:setup bdist_wheel --config=monolithic
mkdir -p dist
cp ./bazel-bin/setup.runfiles/org_altgraph/dist/*.whl ./dist
