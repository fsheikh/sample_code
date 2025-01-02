import express from 'express'; // part of node.js or we can say - that it's the framework of node.js (https://expressjs.com/)
import cors from 'cors'; // package to enable communication between different port numbers of client & server
import sqlite3 from 'sqlite3'; // Import sqlite3
import blogRoutes from './Controllers/blogControllers.js'; //blogRoutes is the instance of blogController file - means that we are importing the functions from blogController file to here
import { DB_NAME } from './Config.js' // importing the database name from the config.js file

const app = express();
const PORT = 3030;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize SQLite3 database
const db = new sqlite3.Database(DB_NAME, (err) => {
  if (err) {
    console.error('Error connecting to the database:', err);
  } else {
    console.log('Connected to the SQLite database');
  }
});

// Make the database connection available to routes
app.use((req, res, next) => {
  req.db = db; // Attach the database connection to the request object
  next(); // https://expressjs.com/en/guide/writing-middleware.html - purple of using next() middleware
});

// Routes
app.use('/blogs', blogRoutes);

app.get('/', (req, res) => {
  res.send('Hello World');
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});