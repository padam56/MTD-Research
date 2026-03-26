#!/bin/bash
# Lateral movement — attacker pivots through the network
# Usage (in Mininet CLI): attacker bash attack_scripts/lateral.sh

echo "=== LATERAL MOVEMENT ==="

echo "[Stage 1] Probing DMZ Web Server..."
ping -c 3 -W 1 10.0.0.30
echo ""

echo "[Stage 2] Attempting to reach Internal App Server via Web..."
ping -c 3 -W 1 10.0.0.100
# Try HTTP request if curl available
curl -s -o /dev/null -w "HTTP %{http_code}" --connect-timeout 2 http://10.0.0.100:8080 2>/dev/null || echo "(no HTTP)"
echo ""

echo "[Stage 3] Attempting to reach DB Server..."
ping -c 3 -W 1 10.0.0.40
echo ""

echo "[Stage 4] Traceroute to DB (shows path)..."
traceroute -n -m 5 10.0.0.40 2>/dev/null || echo "traceroute not available"
echo ""

echo "=== LATERAL MOVEMENT COMPLETE ==="
echo "If MTD is active, traceroute paths should change between runs!"
