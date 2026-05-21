# ---- build stage ----
FROM node:20-alpine AS build

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts

COPY . .
RUN npm run build

# ---- runtime stage ----
FROM nginx:alpine AS runtime

# Copy built bundle to nginx web root (also mounted as widget_dist volume for API)
COPY --from=build /app/dist /usr/share/nginx/html

# Loader script is served at /widget.js
COPY --from=build /app/public/loader.js /usr/share/nginx/html/loader.js

# Health check endpoint via nginx
RUN echo "OK" > /usr/share/nginx/html/health

EXPOSE 80
