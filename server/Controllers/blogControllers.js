import express from 'express';
import sqlite3 from 'sqlite3';
import multer from 'multer';
import { IMAGE_UPLOAD_PATH } from '../Config.js';

const router = express.Router();

// SQLite3 database connection
const db = new sqlite3.Database('deurdu.db', (err) => {
  if (err) {
    console.error('Database connection failed:', err);
  } else {
    console.log('Connected to the SQLite database');
  }
});

// Configure multer for file uploads
const storage = multer.diskStorage({ //multer.diskStorage is used to configure where and how files are stored.
  destination: (req, file, cb) => {
    cb(null, IMAGE_UPLOAD_PATH); // Folder to save uploaded images - defined in Config.js file
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  },
});

// TBD: Image upload functionality is not complete yet. Needs further implementation.
const upload = multer({ storage });

// Create (POST) a new blog
router.post('/postblog', upload.single('image'), (req, res) => {
  const { author_name, blog_category, blog_language, blog_title, blog_content } = req.body;
  const image = req.file ? req.file.filename : null; // req.file.file/filename contains the path where the file is saved.


  const sql = `INSERT INTO postblog 
    (author_name, image, blog_category, blog_language, blog_title, blog_content)
    VALUES (?, ?, ?, ?, ?, ?)`; // The ? syntax used in the SQL query is a placeholder for dynamic values that will be provided at runtime. https://www.sqlite.org/lang_expr.html (Move to 4.Parameters).

  db.run(sql, [author_name, image, blog_category, blog_language, blog_title, blog_content], function (err) {
    if (err) {
      console.error('Error inserting blog:', err);
      return res.status(500).json({ message: 'Error inserting blog', error: err });
    }
    res.status(200).json({ message: 'Blog added successfully', blogId: this.lastID });
  });
});


// Read (GET) all blogs
router.get('/', (req, res) => {
  const limit = req.query.limit || 20; // Default to 20 if not provided
  const sql = 'SELECT * FROM postblog LIMIT ?';
  db.all(sql, [limit], (err, rows) => {
    if (err) {
      return res.status(500).json({ message: 'Error fetching blogs', error: err });
    }
    res.status(200).json({ blogs: rows });
  });
});


// Read (GET) blogs with filters
router.get('/search', (req, res) => {
  const { category = '', author_name = '', language = '' } = req.query;

  let sql = 'SELECT * FROM postblog WHERE 1=1';
  const params = [];

  if (category) {
    sql += ' AND LOWER(blog_category) = LOWER(?)';
    params.push(category);
  }
  if (author_name) {
    sql += ' AND LOWER(author_name) = LOWER(?)';
    params.push(author_name);
  }
  if (language) {
    sql += ' AND LOWER(blog_language) = LOWER(?)';
    params.push(language);
  }

  db.all(sql, params, (err, rows) => {
    if (err) {
      return res.status(500).json({ message: 'Error fetching blogs', error: err });
    }
    res.status(200).json({ blogs: rows });
  });
});


// Read (GET) blogs by category and language
router.get('/selected-category', (req, res) => {
  const { category, language } = req.query;

  const sql = `SELECT * FROM postblog 
    WHERE LOWER(blog_category) = LOWER(?) AND LOWER(blog_language) = LOWER(?)`;

  db.all(sql, [category, language], (err, rows) => {
    if (err) {
      return res.status(500).json({ message: 'Error fetching blogs', error: err });
    }
    res.status(200).json({ blogs: rows });
  });
});

// Update (PUT) a blog
router.put('/:id', (req, res) => {
  const { id } = req.params;
  const { author_name, blog_category, blog_language, blog_title, blog_content } = req.body;

  const sql = `UPDATE postblog 
    SET author_name = ?, blog_category = ?, blog_language = ?, blog_title = ?, blog_content = ? 
    WHERE id = ?`;

  db.run(sql, [author_name, blog_category, blog_language, blog_title, blog_content, id], function (err) {
    if (err) {
      return res.status(500).json({ message: 'Error updating blog', error: err });
    }
    if (this.changes === 0) {
      return res.status(404).json({ message: 'Blog not found' });
    }
    res.status(200).json({ message: 'Blog updated successfully' });
  });
});

// Delete (DELETE) a blog
router.delete('/:id', (req, res) => {
  const { id } = req.params;

  const sql = 'DELETE FROM postblog WHERE id = ?';

  db.run(sql, [id], function (err) {
    if (err) {
      return res.status(500).json({ message: 'Error deleting blog', error: err });
    }
    if (this.changes === 0) {
      return res.status(404).json({ message: 'Blog not found' });
    }
    res.status(200).json({ message: 'Blog deleted successfully' });
  });
});

export default router;