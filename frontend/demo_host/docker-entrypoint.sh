#!/bin/sh
set -e

DOC_ROOT="/srv/${HOST_TYPE:-allowed}"

# Cache-Control: no-store ensures Firefox/Chrome never serve a stale BFCache
# entry that could mistakenly mark the page as "embedded by another site."
# This is critical for the disallowed-host demo, where Firefox can otherwise
# refuse to render the page after a prior iframe-block event for the same URL.
cat > /etc/nginx/conf.d/default.conf <<EOF
server {
    listen 80;
    root ${DOC_ROOT};
    index index.html;

    add_header Cache-Control "no-store, no-cache, must-revalidate" always;
    add_header Pragma "no-cache" always;

    location / {
        try_files \$uri \$uri/ /index.html;
    }
}
EOF

exec "$@"
