import express from 'express';
import cors from 'cors';
import blogControllerRoutes from './Controllers/blogControllers.js';
// import authControllerRoutes from './Controllers/authController.js';

const app = express();

app.use(express.json());

app.use(cors());

app.use('/blogs', blogControllerRoutes);
// app.use('/auth', authControllerRoutes);

app.get('/', (req, res) => {
  res.send('Hello World');
});

app.listen(3030, () => {
  console.log('Server is running on port 3030');
});