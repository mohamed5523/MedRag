# syntax=docker/dockerfile:1

# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# Build with production mode
RUN npm run build

# Runtime stage with Nginx
FROM nginx:1.27-alpine AS runtime
WORKDIR /usr/share/nginx/html

# Copy built assets
COPY --from=build /app/dist .

# Nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]


