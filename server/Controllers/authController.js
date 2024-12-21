// import mysql from 'mysql'
// import jwt from 'jsonwebtoken'
// import express from 'express'
// import cors from 'cors'
// import cookieParser from 'cookie-parser'
// import bcrypt, { hash } from 'bcrypt'
// const salt = 10;
// const router = express.Router();

// const app = express();
// app.use(express.json());

// // Configure CORS to allow requests from your frontend
// app.use(cors({
//     origin: 'http://localhost:5173', // Allow this origin to access your backend
//     credentials: true // Allow cookies to be sent with requests
// }));

// app.use(cookieParser());

// const db = mysql.createConnection({
//     host: 'localhost',
//     user: 'root',
//     database: 'deurdu',
//     password: ''
// })


// This is Registeration Route
// router.post('/register', (req, res) => {
//     const sql = "INSERT INTO user_register (username, gmail, password) VALUES (?, ?, ?)";
//     bcrypt.hash(req.body.password.toString(), salt, (err, hash) => {
//         if (err) return res.json({ Error: 'Error hashing password' });
        
//         const values = [
//             req.body.username,
//             req.body.gmail,
//             hash
//         ];
        
//         db.query(sql, values, (err, result) => {
//             if (err) {
//                 console.error('SQL Error:', err); // Log the detailed error
//                 return res.json({ Error: 'Inserting data into db error', Details: err.message });
//             }
//             return res.json({ Status: 'Success' });
//         });
//     });
// });


// // This is Login Route
// router.post('/login', (req, res) => {
//     const sql = "SELECT * FROM user_register WHERE gmail = ?";        
//     db.query(sql, [req.body.gmail], (err, data) => {
//         if (err) return res.json({ Error: 'Inserting data into db error', Details: err.message });

//         if (data.length > 0) {
//             bcrypt.compare(req.body.password.toString(), data[0].password, (err, response) => {
//                 if (err) return res.json(`Password Compare error`);
                
//                 if (response) {
//                     const name = data[0].name;
//                     const token = jwt.sign({ name }, 'jwt-secret-key');

//                     // Insert login record into the user_login table
//                     const logSql = "INSERT INTO user_login (gmail, password) VALUES (?, ?)";
//                     const logValues = [req.body.gmail, data[0].password];  // Store the original hashed password

//                     db.query(logSql, logValues, (logErr, logResult) => {
//                         if (logErr) {
//                             console.error('Error logging login activity:', logErr);
//                         } else {
//                             console.log('Login activity logged for user:', req.body.gmail);
//                         }
//                     });

//                     res.cookie('token', token, { httpOnly: true, secure: false });
//                     return res.json({ Status: 'Success', Token: token });
//                 } else {
//                     return res.json({ Error: 'Password not matched' });
//                 }
//             });
//         } else {
//             return res.json({ Error: 'No email existed' });
//         }
//     });
// });




// router.get('/status', (req, res) => {
//     const cookies = req.cookies.token; // Get cookies
//     if (!cookies || !cookies.token) {
//         return res.json({ loggedIn: false });
//     }
//     jwt.verify(cookies.token, 'jwt-secret-key', (err) => {
//         if (err) {
//             return res.json({ loggedIn: false });
//         }
//         return res.json({ loggedIn: true });
//     });
// });




export default router;