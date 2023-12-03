FROM node:20

WORKDIR /workdir
COPY frontend/package*.json /workdir/
RUN npm install
COPY frontend/ /workdir/
EXPOSE 5173
CMD ["npm", "run", "dev"]
