# parity-glm Wed Sep  2 10:51:09 PM UTC 2026
unit=parity-glm lane=pair ttl=2h
HEAD=20f2a98d22003f1f3c74e20cfafd7216a0fe881f branch=agent/kit
kit/doctor.sh . | tee $EV/doctor-preboot.txt
VALIDATE_ONLY=1 ./run.sh | tee $EV/validate.txt
./run.sh 2>&1 | tee $EV/run.log   # defaults, no env overrides
kit/doctor.sh . | tee $EV/doctor.txt
kit/probes/run-all.sh . $EV
python3 kit/bench_decode.py --recipe . --phase both --out $EV
python3 kit/bench_compare.py evidence/rebench-20260902T204243Z/bench.json $EV/bench.json | tee $EV/parity.txt
./stop.sh 2>&1 | tee $EV/stop.txt
