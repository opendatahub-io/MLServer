#!/bin/bash
set -euo pipefail

_copy() {
  local _src=$1
  local _newModelName=$2
  local _modelName=${_src##*/}
  local _dst="$(dirname "$_src")/$_newModelName"

  mkdir -p "$_dst"
  cp -R "$_src/." "$_dst/"
  sed -i "s/$_modelName/$_newModelName/" "$_dst/model-settings.json"
}

_main() {
  local _copies=10

  local _i
  for ((_i=1; _i<=_copies; _i++)); do
    _copy "$PWD/testserver/models/iris" "iris-$_i"
    _copy "$PWD/testserver/models/sum-model" "sum-model-$_i"
  done
}

_main
