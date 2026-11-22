#!/usr/bin/env bash
# ==============================================================================
# StudyTrade - 1-Click GCP Always-Free VM Automated Provisioning Script
# Target OS: Ubuntu 22.04 / 24.04 LTS (Compute Engine e2-micro)
# ==============================================================================
set -euo pipefail

echo "========================================================================="
echo "   StudyTrade - Automated 1-Click Setup for GCP Compute Engine (Free)"
echo "========================================================================="

# 1. Ensure Running as Root
if [ "$EUID" -ne 0 ]; then
  echo "[-] Please run as root or with sudo: sudo bash deploy_gcp.sh"
  exit 1
fi

APP_DIR="/opt/studytrade"
APP_USER="${SUDO_USER:-ubuntu}"
PORT=8000

echo "[1/6] Updating system packages & installing prerequisites..."
apt-get update -y
apt-get install -y python3-venv python3-pip python3-dev build-essential git nginx curl sqlite3 libatlas-base-dev

echo "[2/6] Configuring 2GB Swap Space (Essential for e2-micro 1GB RAM)..."
if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    if ! grep -q "/swapfile" /etc/fstab; then
        echo '/swapfile none swap sw 0 0' >> /etc/fstab
    fi
    echo "[+] 2GB Swap space successfully created and enabled."
else
    echo "[*] Swap file already exists. Skipping."
fi

echo "[3/6] Setting up project directory at ${APP_DIR}..."
if [ ! -d "${APP_DIR}" ]; then
    mkdir -p "${APP_DIR}"
    # Copy from current directory if script is run inside cloned repo
    if [ -f "$(dirname "$0")/../pyproject.toml" ]; then
        cp -r "$(dirname "$0")/.."/* "${APP_DIR}/"
    else
        echo "[*] Please clone or copy your StudyTrade repository into ${APP_DIR}"
    fi
fi

chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"

echo "[4/6] Creating Python virtual environment & installing dependencies..."
sudo -u "${APP_USER}" bash << EOF
cd "${APP_DIR}"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install gunicorn

# Run Django migrations and initialize data
cd web
python manage.py migrate --noinput
EOF

echo "[5/6] Creating Systemd Service for Gunicorn..."
cat > /etc/systemd/system/studytrade.service << EOF
[Unit]
Description=StudyTrade Quant Trading & Paper Cockpit Engine
After=network.target

[Service]
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${APP_DIR}/web
Environment="PATH=${APP_DIR}/.venv/bin"
ExecStart=${APP_DIR}/.venv/bin/gunicorn \
    --workers 2 \
    --bind 127.0.0.1:${PORT} \
    --timeout 120 \
    --access-logfile /var/log/studytrade_access.log \
    --error-logfile /var/log/studytrade_error.log \
    studytrade_web.wsgi:application
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable studytrade
systemctl restart studytrade

echo "[6/6] Configuring Nginx Reverse Proxy on Port 80..."
cat > /etc/nginx/sites-available/studytrade << EOF
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    client_max_body_size 50M;

    location /static/ {
        alias ${APP_DIR}/web/static/;
        expires 30d;
    }

    location / {
        proxy_pass http://127.0.0.1:${PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 120s;
    }
}
EOF

# Enable Nginx site
rm -f /etc/nginx/sites-enabled/default || true
ln -sf /etc/nginx/sites-available/studytrade /etc/nginx/sites-enabled/studytrade
nginx -t
systemctl restart nginx

PUBLIC_IP=\$(curl -s ifconfig.me || hostname -I | awk '{print \$1}')

echo "========================================================================="
echo "   [SUCCESS] StudyTrade is successfully deployed & running 24/7!"
echo "   Access Dashboard: http://\${PUBLIC_IP}/"
echo "   Gamma Squeeze:    http://\${PUBLIC_IP}/gamma-squeeze/"
echo "   Paper Cockpit:    http://\${PUBLIC_IP}/paper/"
echo "========================================================================="
