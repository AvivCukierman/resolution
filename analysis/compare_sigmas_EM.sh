#/bin/bash

python compare_resolution.py \
--collections EM_collections \
--minnpv 5 \
--maxnpv 30 \
--npvbin 5 \
--minpt 20 \
--maxpt 80 \
--ptbin 2
