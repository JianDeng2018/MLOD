#!/usr/bin/env bash

set -e
set -x

autopep8 --aggressive --aggressive --in-place $1
