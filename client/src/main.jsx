import React from 'react';
import ReactDOM from 'react-dom/client';
import { ClerkProvider } from '@clerk/clerk-react'; // Remove RedirectToSignIn import since it's not needed here
import App from './App.jsx';
import './index.css';

const PUBLISHABLE_KEY = 'pk_test_c2VjdXJlLXR1cnRsZS02Mi5jbGVyay5hY2NvdW50cy5kZXYk';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ClerkProvider publishableKey={PUBLISHABLE_KEY}>
      <App />
    </ClerkProvider>
  </React.StrictMode>
);
