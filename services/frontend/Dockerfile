# Frontend image: build the SPA with Node, serve the static bundle with nginx.
# Build context is the REPO ROOT (matches the other services).
#
#   docker build -f services/frontend/Dockerfile -t recap-frontend .

# ---- build ----------------------------------------------------------------
FROM node:22-alpine AS build
WORKDIR /app

# Install from the committed lockfile first (cached until deps change).
COPY services/frontend/package.json services/frontend/package-lock.json ./
RUN npm ci

# Build the production bundle.
COPY services/frontend/ ./
RUN npm run build

# ---- runtime --------------------------------------------------------------
FROM nginx:1.30.3-alpine AS runtime
COPY services/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
# nginx runs in the foreground via its default entrypoint/CMD.
