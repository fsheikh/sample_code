This directory contains all data and source code for deurdu blogging
website.

## Dependencies
Tested with node version `20.18.0` and npm version `10.8.2`. On Ubuntu/Linux, please check this [link](https://askubuntu.com/questions/426750/how-can-i-update-my-nodejs-to-the-latest-version) for update instructions

## Database setup
1. Make sure to have XAMPP CONTROL PANEL, Linux/Ubuntu package may be installed from [here](https://www.apachefriends.org/download.html)

2. Create a sample database, launch `mysql` server from lampp/xampp installation e.g. `/opt/lampp/bin`

```bash
/mysql -h localhost -u root -p
Enter password: 
Welcome to the MariaDB monitor.  Commands end with ; or \g.
Your MariaDB connection id is 25
Server version: 10.4.32-MariaDB Source distribution

Copyright (c) 2000, 2018, Oracle, MariaDB Corporation Ab and others.

Type 'help;' or '\h' for help. Type '\c' to clear the current input statement.

MariaDB [(none)]> CREATE DATABASE deurdu;
Query OK, 1 row affected (0,010 sec)

MariaDB [(none)]> 
```

## Frontend/Client side
1. First install node.js in system
Goto --> 'https://nodejs.org/en/download/source-code'

2. After opeing the 'deurdu' folder
..* Move to client using 'cd client'
..* Type `npm i` -- this will install all the packages with a single command
..* then type `npm run dev` -- this will start project on localhost:5173


## Backend/Server side
1. Open another terminal - make sure that client keeps running in previous terminal
2. To install all the dependencies with a single command of server-side type `npm i` in the server side terminal
3. After that type `nodemon index.js`.