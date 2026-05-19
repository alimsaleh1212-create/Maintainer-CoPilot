FROM nginx:alpine

# Copy both host variants; the entrypoint script selects one based on HOST_TYPE env var.
COPY allowed/index.html /srv/allowed/index.html
COPY disallowed/index.html /srv/disallowed/index.html

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 80

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
