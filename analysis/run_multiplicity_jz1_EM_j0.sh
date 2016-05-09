#/bin/bash

python multiplicity.py -i j0_jz1_EM \
-c 20 \
--inputDir ../../Voronoi_xAOD/LSF_EM_JZ1_all/fetch/data-outputTree/ \
--submitDir ../output_absolute/ \
--jetpt j0pt \
--npv NPV \
--jeteta j0eta \
--maxeta 0.8 \
--numEvents 100000

