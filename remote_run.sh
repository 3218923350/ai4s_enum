#!/bin/bash
set -e
cd /root/ai4s_enum || exit 1
source /root/ai4s_enum/.env || exit 1

nohup /opt/mamba/bin/python run.py --all > log 2>&1 &

disown $!

exit 0