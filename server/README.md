## 🔄 **Regenerating `package.json` and `package-lock.json`**

If you've accidentally deleted the `package.json` or `package-lock.json` files in the **server** directory, you can follow these steps to regenerate them.

### 1️⃣ **Regenerate `package.json`**

To regenerate the `package.json` file with default values, run the following command in the **server** directory:

```bash
npm init -y
```

### **2️⃣ Regenerate `package-lock.json`**
The `package-lock.json` file will be automatically generated when you install your project's dependencies. To regenerate it, simply run:

```bash
npm install
```
This command installs the dependencies listed in the new `package.json` and regenerates the `package-lock.json` file.

### **3️⃣ To Start The Server**
The `index.js || server.js` file will be automatically runned when you will run this script on your terminal, simply run:

If Installed:
```bash
nodemon index.js
```

If not Installed (Install it first...):
```bash
npm install -g nodemon
```