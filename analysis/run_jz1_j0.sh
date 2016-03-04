#/bin/bash

python resolution.py -i j0_jz1_LC \
-r \
-n \
--minnpv 5 \
--maxnpv 30 \
--npvbin 5 \
--ptbin 5 \
--minpt 20 \
--maxpt 60 \
--inputDir ../../Voronoi_xAOD/LSF_LC_JZ1_all/fetch/data-outputTree/ \
--jetpt j0pt \
--tjetpt tj0pt \
--npv NPV \
--tjeteta tj0eta \
--tjetmindr tj0mindr \
--maxeta 1.0 \
--mindr 0.6 \
--numEvents 1000000

