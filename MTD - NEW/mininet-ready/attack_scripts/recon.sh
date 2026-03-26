#!/bin/bash
# Reconnaissance attack — run from Mininet attacker host
# Usage (in Mininet CLI): attacker bash attack_scripts/recon.sh

echo "=== RECONNAISSANCE SCAN ==="
echo "Scanning DMZ Web Server..."
nmap -sS -T4 10.0.0.30 2>/dev/null || ping -c 5 10.0.0.30

echo ""
echo "Scanning Internal App Server..."
nmap -sS -T4 10.0.0.100 2>/dev/null || ping -c 5 10.0.0.100

echo ""
echo "Scanning DB Server..."
nmap -sS -T4 10.0.0.40 2>/dev/null || ping -c 5 10.0.0.40

echo ""
echo "Full subnet scan..."
nmap -sn 10.0.0.0/24 2>/dev/null || echo "nmap not available, using ping sweep"
for i in 10 20 30 40 100; do
    ping -c 1 -W 1 10.0.0.$i > /dev/null 2>&1 && echo "  10.0.0.$i is UP" || echo "  10.0.0.$i is DOWN"
done

echo ""
echo "=== RECON COMPLETE ==="
