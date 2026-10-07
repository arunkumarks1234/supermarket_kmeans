FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Build for production
RUN npm run build

# Install serve to serve the built assets
RUN npm install -g serve

EXPOSE 3000

CMD ["serve", "-s", "dist", "-p", "3000"]
