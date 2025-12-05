#!/bin/bash
echo "Waiting for Prometheus..."
until curl -s http://localhost:9090 >/dev/null; do
  sleep 2
done
echo "Prometheus is UP!"

