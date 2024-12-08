import express from 'express';
import mysql from 'mysql';
import multer from 'multer';

const router = express.Router();

// MySQL database connection
const db = mysql.createConnection({
  host: 'localhost',
  user: 'root',
  password: '',
  database: 'deurdu',
});

db.connect((err) => {
  if (err) {
    console.error('Database connection failed:', err);
    return;
  }
  console.log('Database connected');
});

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './pictures'); // Folder to save uploaded images
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  },
});

const upload = multer({ storage });

router.post('/postblog', upload.single('image'), (req, res) => {
    console.log('Request Body:', req.body);
    console.log('Uploaded File:', req.file);

    const { author_name, blog_category, blog_language, blog_title, blog_content } = req.body;
    const image = req.file ? req.file.filename : null;

    const sql = 'INSERT INTO postblog (author_name, image, blog_category, blog_language, blog_title, blog_content) VALUES (?, ?, ?, ?, ?, ?)';
    db.query(sql, [author_name, image, blog_category, blog_language, blog_title, blog_content], (err, result) => {
        if (err) {
        console.error('Error inserting blog:', err);
        return res.status(500).send('Error inserting blog');
        }
        res.status(200).send('Blog added successfully');
    });
});


// This will display all the blogs in database
// Read (GET) blogs
router.get('/', (req, res) => {
  const sql = 'SELECT * FROM postblog';
  db.query(sql, (err, result) => {
    if (err) {
      res.status(500).json({ message: 'Error fetching blogs', error: err });
    } else {
      res.status(200).json({ result });
    }
  });
});



router.get('/search', (req, res) => {
  const category = req.query.category || '';
  const author_name = req.query.author_name || '';
  const language = req.query.language || '';

  // Build dynamic SQL query based on available filters
  let sql = 'SELECT * FROM postblog WHERE 1=1'; // Start with a valid base query
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

  console.log('SQL Query:', sql);
  db.query(sql, params, (err, result) => {
    if (err) {
      res.status(500).json({ message: 'Error fetching blogs', error: err });
    } else {
      res.status(200).json({ result });
    }
  });
});







// This will display blogs based on the user-selected category and language
router.get('/selected-category', (req, res) => {
  const category = req.query.category;
  const language = req.query.language; // Get the language from the query parameters

  console.log('Received category:', category);
  console.log('Received language:', language);

  const sql = 'SELECT * FROM postblog WHERE LOWER(blog_category) = LOWER(?) AND LOWER(blog_language) = LOWER(?)';
  
  db.query(sql, [category, language], (err, result) => {
    if (err) {
      res.status(500).json({ message: 'Error fetching blogs', error: err });
    } else {
      console.log('Blogs retrieved from DB:', result); // Log the retrieved blogs
      res.status(200).json({ result }); // Ensure that "result" is being sent in the correct format
    }
  });
});






// GET blogs by category and language
router.get('/blogs', (req, res) => {
  const category = req.query.category;
  const language = req.query.language;

  const sql = 'SELECT * FROM postblog WHERE LOWER(blog_category) = LOWER(?) AND LOWER(blog_language) = LOWER(?)';
  const params = [category, language];

  console.log('Executing SQL:', sql);
  console.log('With parameters:', params);

  db.query(sql, params, (err, results) => {
    if (err) {
      console.error('Error fetching blogs:', err);
      return res.status(500).json({ message: 'Error fetching blogs', error: err });
    }
    console.log('Blogs retrieved from DB:', results);
    res.status(200).json(results);
  });
});







// Update (PUT) a blog
router.put('/:id', (req, res) => {
  const { id } = req.params;
  const { author_name, blog_category, blog_language, blog_title, blog_content } = req.body;
  const sql = 'UPDATE postblog SET author_name = ?, blog_category = ?, blog_language = ?, blog_title = ?, blog_content = ? WHERE id = ?';

  db.query(sql, [author_name, blog_category, blog_language, blog_title, blog_content, id], (err, result) => {
    if (err) {
      res.status(500).json({ message: 'Error updating blog', error: err });
    } else {
      res.status(200).json({ message: 'Blog updated successfully' });
    }
  });
});

// Delete (DELETE) a blog
router.delete('/:id', (req, res) => {
  const { id } = req.params;
  const sql = 'DELETE FROM postblog WHERE id = ?';
  
  db.query(sql, [id], (err, result) => {
    if (err) {
      res.status(500).json({ message: 'Error deleting blog', error: err });
    } else {
      res.status(200).json({ message: 'Blog deleted successfully' });
    }
  });
});

export default router;