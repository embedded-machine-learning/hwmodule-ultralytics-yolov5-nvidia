#!/bin/bash

echo "Setup task spooler socket for GPU."

export TS_SOCKET="/home/cdleml/Task-Spooler-Socket/Xavier.socket"
chmod 777 /home/cdleml/Task-Spooler-Socket/Xavier.socket

export TS_TMPDIR=~/logs

echo task spooler output directory: ~/logs
echo "Task spooler initialized $TS_SOCKET"
