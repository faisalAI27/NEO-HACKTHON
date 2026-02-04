# MOSAIC Frontend Dockerfile
# Multi-stage build: Node.js build + Nginx serve

# ==============================================================================
# Stage 1: Build - Compile React TypeScript application
# ==============================================================================
FROM node:20-alpine as builder

WORKDIR /app

# Copy package files first for layer caching
COPY mosaic-dashboard/package.json mosaic-dashboard/package-lock.json* ./

# Install dependencies
RUN npm ci --silent

# Copy source code
COPY mosaic-dashboard/ .

# Build production bundle
RUN npm run build

# ==============================================================================
# Stage 2: Production - Nginx to serve static files
# ==============================================================================
FROM nginx:alpine as production

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Create a non-root user for security
RUN addgroup -g 1001 -S nginx-user && \
    adduser -S -D -H -u 1001 -h /var/cache/nginx -s /sbin/nologin -G nginx-user nginx-user && \
    chown -R nginx-user:nginx-user /var/cache/nginx && \
    chown -R nginx-user:nginx-user /usr/share/nginx/html && \
    touch /var/run/nginx.pid && \
    chown -R nginx-user:nginx-user /var/run/nginx.pid

# Expose HTTP port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
