#/bin/bash

python resolution.py -i j0_jz2_LC \
-n \
--minnpv 5 \
--maxnpv 30 \
--npvbin 5 \
--ptbin 2 \
--inputDir ../../Voronoi_xAOD/LSF_LC_JZ2_all/fetch/data-outputTree/ \
--jetpt j0pt \
--tjetpt tj0pt \
--npv NPV \
--tjeteta tj0eta \
--tjetmindr tj0mindr \
--maxeta 1.0 \
--mindr 0.6 \
--numEvents 1000000

