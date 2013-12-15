#!/bin/bash
# submit all jobs step1 --> step5
echo "Submit all steps' pbs scripts"
qsub step1_trainEncoder.pbs
qsub step2_encoding.pbs
qsub step3_aggrFeat.pbs
qsub step4_train.pbs
qsub step5_test.pbs
echo "Done"
exit 0