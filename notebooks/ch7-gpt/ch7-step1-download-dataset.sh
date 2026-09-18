#!/bin/sh

uv run modelscope download --dataset BAAI/CCI3-HQ --local_dir ./cci3-hq-data --include "data/part_000[0-2][0-9][0-9].jsonl"
