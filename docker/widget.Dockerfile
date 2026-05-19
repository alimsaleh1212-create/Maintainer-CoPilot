# ---- build stage ----
FROM node:20-alpine AS build

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts

COPY . .
RUN npm run build

# ---- runtime stage ----
FROM nginx:alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html

# Health check endpoint via nginx
RUN echo "OK" > /usr/share/nginx/html/health

EXPOSE 80
