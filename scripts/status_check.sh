#!/bin/bash
# Job status check, executed on either cluster via `ssh ... 'bash -s'`.
# Reports queue, accounting, and log tails for the Phase 2 primary jobs.
echo "=== host: $(hostname) ==="
# Trillium's GPU cluster has its own Slurm: query it from trig-login01,
# otherwise GPU jobs are invisible to squeue/sacct on CPU login nodes.
case "$(hostname)" in
  tri-*) RUN="ssh -n trig-login01" ;;
  *)     RUN="" ;;
esac
echo "=== queue ==="
$RUN squeue -u hamdan
echo "=== accounting (yesterday onward) ==="
$RUN sacct -S 2026-07-04 -u hamdan -o "JobID,JobName%22,State,Elapsed,ExitCode" 2>&1 | head -15
echo "=== log tails ==="
for f in /scratch/hamdan/CaliDoc/logs/*retrain_s42*.err; do
  [ -e "$f" ] || continue
  echo "--- $f (last 15 lines):"; tail -15 "$f"
done
for f in /scratch/hamdan/CaliDoc/logs/*retrain_s42*.out; do
  [ -e "$f" ] || continue
  echo "--- $f (last 10 lines):"; tail -10 "$f"
done
echo "=== STATUS_DONE ==="
