#!/bin/bash
# Открывает порт веб-разметчика (tools/label_server.py) на сервере и включает HTTPS.
# Запускать на сервере из папки с label_server.py, start.sh и token:
#
#     sudo ./open_labeling_port.sh public   # порт открыт всем (для проброса с роутера из интернета)
#     sudo ./open_labeling_port.sh lan      # только 192.168.50.0/24
#     sudo ./open_labeling_port.sh close    # убрать правила ufw
#
# Сертификат самоподписанный: браузер один раз предупредит, что ему не доверяет — это ожидаемо.
set -euo pipefail

MODE="${1:-public}"
PORT="${PORT:-8765}"
LAN="${LAN:-192.168.50.0/24}"
OWNER="${SUDO_USER:-pin}"
DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="smoke-labeling"

[ "$(id -u)" -eq 0 ] || { echo "запустите через sudo"; exit 1; }
command -v ufw >/dev/null || { echo "ufw не найден"; exit 1; }

delete_rules() {
    # правила с нашим комментарием удаляем по номерам с конца, чтобы номера не съезжали
    while read -r n; do ufw --force delete "$n" >/dev/null; done < <(
        ufw status numbered | grep "$TAG" | sed -E 's/^\[ *([0-9]+)\].*/\1/' | sort -rn || true)
}

case "$MODE" in
close)
    delete_rules
    echo "правила ufw для порта $PORT удалены"
    exit 0
    ;;
lan | public) ;;
*)
    echo "режим: public | lan | close"
    exit 1
    ;;
esac

# TLS: без него токен и кадры по интернету уходят открытым текстом
if [ ! -f "$DIR/tls/cert.pem" ]; then
    sudo -u "$OWNER" mkdir -p "$DIR/tls"
    sudo -u "$OWNER" openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=smoke-labeling" \
        -keyout "$DIR/tls/key.pem" -out "$DIR/tls/cert.pem" 2>/dev/null
    chmod 600 "$DIR/tls/key.pem"
    echo "создан сертификат $DIR/tls/cert.pem (365 дней)"
fi

delete_rules
if [ "$MODE" = "public" ]; then
    ufw allow "$PORT"/tcp comment "$TAG" >/dev/null
    echo "порт $PORT открыт для всех"
else
    ufw allow from "$LAN" to any port "$PORT" proto tcp comment "$TAG" >/dev/null
    echo "порт $PORT открыт для $LAN"
fi

# перезапуск разметчика от имени владельца; start.sh сам подхватывает tls/, если он есть
sudo -u "$OWNER" sh -c "cd '$DIR' && (setsid ./start.sh </dev/null >/dev/null 2>&1 &)"
sleep 2
if curl -ksf -o /dev/null -m 5 "https://127.0.0.1:$PORT/?token=$(cat "$DIR/token")"; then
    echo "разметчик отвечает по HTTPS: https://<адрес>:$PORT/?token=$(cat "$DIR/token")"
else
    echo "разметчик не отвечает, смотрите $DIR/server.log"
    exit 1
fi
