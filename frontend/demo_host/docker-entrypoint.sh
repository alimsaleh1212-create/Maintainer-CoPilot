#!/bin/sh
set -e

DOC_ROOT="/srv/${HOST_TYPE:-allowed}"

cat > /etc/nginx/conf.d/default.conf <<EOF
server {
    listen 80;
    root ${DOC_ROOT};
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
    }
}
EOF

exec "$@"
