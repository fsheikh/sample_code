import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import './i18.js'
import { Auth0Provider } from '@auth0/auth0-react';

const root = createRoot(document.getElementById('root'));

root.render(
<Auth0Provider
    domain="dev-2mlojserqc0ytr8b.us.auth0.com"
    clientId="NyjTsxGX6tAxOxKL6JkjzR6hY7MKhHn4"
    authorizationParams={{
      redirect_uri: "http://localhost:5173/en"
    }}
  >
    <App />
  </Auth0Provider>,
);