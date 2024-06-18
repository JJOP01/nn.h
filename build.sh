#!/bin/sh

set -xe

clang -Wall -Wextra -o xor xor.c -lm
