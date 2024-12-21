# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react/README.md) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

# 🛠️ **Setup Instructions for New Contributors**

Welcome to the project! 🎉 Follow these simple steps to set up your local development environment. This will ensure you avoid any errors related to missing configurations or dependencies.

---

### 1️⃣ **Clone the Repository**

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

---

### 2️⃣ **Install Project Dependencies**

Ensure that you have Node.js installed (version 14 or above is recommended). Then, install the required dependencies using npm or Yarn:

- Using npm
```bash
npm install
```
- Or if you want to use Yarn
```bash
yarn install
```

### 3️⃣ **Generate Missing Configuration Files**

As some configuration files were deleted, you'll need to regenerate or create them manually. Here are the necessary steps for each file:
## **For TailwindCSS:**
Regenerate the ```tailwind.config.js``` file:
```bash
npx tailwindcss init
```
This will generate the ```tailwind.config.js``` file.

## **For Vite:**
Regenerate the ```vite.config.js``` file by creating a new one in the root of your project and adding the following content:
```JavaScript
// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
})
```

## **For PostCSS**
If the ```postcss.config.js``` is missing, you can either create it manually or reinstall PostCSS:
```bash
npm install postcss
```
Then, manually create the ```postcss.config.js``` file with the necessary PostCSS configuration.

## **For ESLint:**
Regenerate your **ESLint** configuration by running:
```bash
npm install eslint --save-dev
npx eslint --init
```
Follow the prompts to generate the ```**eslint.config.js**``` file.

### 4️⃣ **Run the development server**
Once everything is set up, you can run the development server with one of the following commands:
- Using npm:
```bash
npm run dev
```
- Or using Yarn:
```bash
yarn dev
```

**🎉 Your development environment should now be up and running! Happy coding! 🎉**
