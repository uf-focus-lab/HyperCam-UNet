SCRIPT=$(realpath $(pwd)/check.py)
WORK_DIR=$(realpath $(pwd)/var/run/6a10ec8c/test_results)
python $SCRIPT $WORK_DIR/Dec13_00.C2.npy -B 150 -P "119,115:215,77:69,204"
python $SCRIPT $WORK_DIR/Dec13_00.C5.npy -B 150 -P "103,84:171,86:86,212"
python $SCRIPT $WORK_DIR/Dec13_00.D3.npy -B 150 -P "89,211:125,65:210,83"
python $SCRIPT $WORK_DIR/Dec21_77.B7.npy -B 150 -P "73,85:183,67:108,221"
python $SCRIPT $WORK_DIR/Dec21_77.C8.npy -B 150 -P "26,104:117,44:217,93"
python $SCRIPT $WORK_DIR/Dec21_79.B7.npy -B 150 -P "89,211:125,65:210,83"
