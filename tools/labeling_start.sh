#!/bin/sh
# Запуск веб-разметчика на сервере (кладётся рядом с label_server.py как start.sh).
#   data/ref        — набор (результат tools/export_cycles.py)
#   users.txt       — если есть, включается вход по логину и паролю (python3 label_server.py --users users.txt --add-user ИМЯ)
#   token           — резервный доступ по токену (в окружении, чтобы не светился в ps)
#   tls/cert.pem, tls/key.pem — если есть, включается HTTPS (создаёт open_labeling_port.sh)
cd "$(dirname "$0")" || exit 1
pkill -f "label_server.py --data" 2>/dev/null
ARGS=""
[ -f users.txt ] && ARGS="$ARGS --users users.txt"
[ -f tls/cert.pem ] && [ -f tls/key.pem ] && ARGS="$ARGS --cert tls/cert.pem --key tls/key.pem"
LABEL_TOKEN="$([ -f token ] && cat token)" exec nohup python3 label_server.py --data data/ref --host 0.0.0.0 --port 8765 $ARGS >> server.log 2>&1
