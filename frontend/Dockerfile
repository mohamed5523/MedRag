

# Build stage
FROM node:20-alpine AS build
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
# Build with production mode and inject Vite envs
ENV VITE_SUPABASE_URL=${VITE_SUPABASE_URL}
ENV VITE_SUPABASE_ANON_KEY=${VITE_SUPABASE_ANON_KEY}
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