#!/bin/bash
# Check wildreceipt_conformal_eval (639899) state + output on Trillium.
echo "=== job state ==="
ssh -n trig-login01 'sacct -j 639899 -o "JobID,State,Elapsed,ExitCode"' 2>/dev/null
echo "=== err tail ==="
tail -8 /scratch/hamdan/CaliDoc/logs/wildreceipt_conformal_eval_639899.err 2>&1
echo "=== result file on Trillium ==="
ls -l /scratch/hamdan/CaliDoc/results/wildreceipt_conformal_eval.json 2>&1
echo "=== done ==="
