import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import './i18.js'
import { Auth0Provider } from '@auth0/auth0-react';

const root = createRoot(document.getElementById('root'));

root.render(
<Auth0Provider
    // https://auth0.com/docs/get-started/applications/configure-private-key-jwt
    domain={import.meta.env.VITE_AUTH0_DOMAIN_ID}
    clientId={import.meta.env.VITE_AUTH0_CLIENT_ID}
    authorizationParams={{
      redirect_uri: "http://localhost:5173/en"
    }}
  >
    <App />
  </Auth0Provider>,
);