FROM node:23 AS common-builder

WORKDIR /app

RUN npm install -g pnpm@10

COPY pnpm-workspace.yaml pnpm-lock.yaml package.json ./

WORKDIR /app/common
COPY common/package.json common/tsconfig.json ./
COPY common/src ./src

RUN pnpm install
