// config.js - Here we will initialize the terms which are being used repeatedly in the server and then we will call their instance/nick_name,
// instead of calling their actual_name

// This is for the 1-time initialization for the Database name
export const DB_NAME = 'deurdu.db';

// This is for the 1-time initialization for the Image upload path
export const IMAGE_UPLOAD_PATH = './server/Controllers/pictures';


// The database is being initialized 1-time here
import sqlite3 from 'sqlite3';

const db = new sqlite3.Database(DB_NAME, (err) => {
  if (err) {
    console.error('Error connecting to the database:', err);
  } else {
    console.log('Connected to the SQLite database');
  }
});

export default db;