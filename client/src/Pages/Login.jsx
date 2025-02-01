import React from 'react';
import Navbar from '../Components/Navbar';
import { SignIn } from '@clerk/clerk-react';

const Login = () => {
  return (
    <div className="h-screen flex flex-col">
      <Navbar />
      <div className="flex flex-grow items-center justify-center">
        <SignIn />
      </div>
    </div>
  );
};

export default Login;