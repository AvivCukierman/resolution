#/bin/bash

python resolution.py -i j0_jz1_EM \
-m mode \
-e \
--minnpv 5 \
--maxnpv 25 \
--npvbin 5 \
--ptbin 5 \
--minpt 20 \
--maxpt 60 \
--inputDir ../../Voronoi_xAOD/LSF_EM_JZ1_all_jvoro5/fetch/data-outputTree/ \
--submitDir ../output_absolute/ \
--jetpt j0pt \
--tjetpt tj0pt \
--npv NPV \
--tjeteta tj0eta \
--tjetmindr tj0mindr \
--all_tjetpt tjpt \
--all_tjeteta tjeta \
--all_tjetmindr tjmindr \
--maxeta 0.8 \
--mindr 0.6 \
--numEvents 1000000
