const express = require('express');
const http = require('http');
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: ["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    methods: ["GET", "POST"]
  }
});

const PORT = 5003;

io.on('connection', (socket) => {
  console.log('Socket.io client connected on port 5003');
  
  socket.on('disconnect', () => {
    console.log('Socket.io client disconnected from port 5003');
  });
  
  // Echo any messages back
  socket.on('message', (data) => {
    console.log('Message received on port 5003:', data);
    socket.emit('message', data);
  });
});

server.listen(PORT, () => {
  console.log(`🔌 Socket.io server running on port ${PORT}`);
});
