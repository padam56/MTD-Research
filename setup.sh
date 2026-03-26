#!/bin/bash
# ============================================================
# MTD-Playground Full Setup Script
# ============================================================
# Run this with: sudo bash setup.sh
#
# This installs:
#   1. Mininet (network emulator)
#   2. Open vSwitch (software switch)
#   3. ONOS SDN Controller (via Docker)
#   4. Python dependencies
#
# Requirements: Ubuntu 22.04/24.04, Docker installed
# ============================================================

set -e
echo "============================================"
echo "  MTD-Playground Setup"
echo "============================================"

# --- Step 1: Install Mininet + OVS ---
echo ""
echo "[1/4] Installing Mininet and Open vSwitch..."
apt-get update -qq
apt-get install -y mininet openvswitch-switch openvswitch-common net-tools iputils-ping curl wget
echo "  ✓ Mininet installed: $(mn --version 2>&1)"
echo "  ✓ OVS installed: $(ovs-vswitchd --version 2>&1 | head -1)"

# --- Step 2: Start OVS ---
echo ""
echo "[2/4] Starting Open vSwitch service..."
systemctl enable openvswitch-switch
systemctl start openvswitch-switch
echo "  ✓ OVS running"

# --- Step 3: Pull and run ONOS ---
echo ""
echo "[3/4] Pulling ONOS Docker image (this may take a few minutes)..."
docker pull onosproject/onos:2.7.0
echo "  ✓ ONOS image pulled"

echo ""
echo "Starting ONOS container..."
# Stop existing ONOS container if running
docker rm -f onos-mtd 2>/dev/null || true

docker run -d \
  --name onos-mtd \
  --restart unless-stopped \
  -p 8181:8181 \
  -p 8101:8101 \
  -p 6653:6653 \
  -p 6640:6640 \
  onosproject/onos:2.7.0

echo "  ✓ ONOS container started"
echo "  Waiting for ONOS to boot (this takes 30-60 seconds)..."

# Wait for ONOS to be ready
for i in $(seq 1 60); do
  if curl -s -o /dev/null -w "%{http_code}" -u onos:rocks http://localhost:8181/onos/v1/cluster 2>/dev/null | grep -q "200"; then
    echo "  ✓ ONOS is ready!"
    break
  fi
  echo "    Waiting... ($i/60)"
  sleep 2
done

# --- Step 4: Activate required ONOS apps ---
echo ""
echo "[4/4] Activating ONOS applications..."

# OpenFlow provider (connects to Mininet switches)
curl -s -X POST -u onos:rocks http://localhost:8181/onos/v1/applications/org.onosproject.openflow/active
echo "  ✓ OpenFlow provider activated"

# Reactive forwarding (basic L2 forwarding)
curl -s -X POST -u onos:rocks http://localhost:8181/onos/v1/applications/org.onosproject.fwd/active
echo "  ✓ Reactive forwarding activated"

# Host provider (discovers hosts)
curl -s -X POST -u onos:rocks http://localhost:8181/onos/v1/applications/org.onosproject.hostprovider/active
echo "  ✓ Host provider activated"

# --- Step 5: Python dependencies ---
echo ""
echo "[5/5] Installing Python dependencies..."
pip3 install --break-system-packages requests gymnasium stable-baselines3 numpy matplotlib rich 2>/dev/null || \
pip3 install requests gymnasium stable-baselines3 numpy matplotlib rich
echo "  ✓ Python dependencies installed"

# --- Done ---
echo ""
echo "============================================"
echo "  SETUP COMPLETE!"
echo "============================================"
echo ""
echo "  ONOS Web GUI:   http://localhost:8181/onos/ui"
echo "  Username:        onos"
echo "  Password:        rocks"
echo ""
echo "  ONOS REST API:   http://localhost:8181/onos/v1/"
echo "  OpenFlow port:   6653"
echo ""
echo "  Next steps:"
echo "    1. Open http://localhost:8181/onos/ui in browser"
echo "    2. Run the topology: sudo python3 mininet-ready/topology.py"
echo "    3. Run the MTD agent: python3 mininet-ready/run_live.py"
echo ""
echo "  To check ONOS status:"
echo "    docker logs onos-mtd --tail 20"
echo "    curl -u onos:rocks http://localhost:8181/onos/v1/cluster"
echo "============================================"
