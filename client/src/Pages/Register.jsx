import React, { useEffect, useState } from 'react'
import Navbar from '../Components/Navbar'
import axios from 'axios'
import { useNavigate } from 'react-router-dom'
import i18 from 'i18next'

function Register() {

    const navigate = useNavigate();

    const [values,setValues] = useState({
        username: '',
        gmail: '',
        password: ''
    })

    const getLanguagePath = () => `/${i18.language}`

    const handleSumbit = (e) =>
    {
        e.preventDefault();
        if(values.username==='' || values.gmail==='' || values.password==='')
            {
                alert('Make sure to fill all fields.')
                return;
            }
        axios.post('http://localhost:3030/auth/register', values)
        .then(res => 
            {
                console.log('Server response:', res.data); // Log the response data
                if(res.data.Status === 'Success') 
                {
                    navigate(`${getLanguagePath()}/login`);
                } 
                else 
                {
                    alert('Error while registering the user.');
                }
            })
            .catch(err => {
                console.error('Error submitting form:', err.response ? err.response.data : err.message);
                alert('An error occurred during registration.');
      });
    }

      

  return (
    <div>
        <Navbar/>
        <div className="flex flex-col justify-center items-center">
            <h1 className="text-[50px] font-semibold mt-1">Register</h1>
                <div className="flex flex-col bg-slate-300 w-[40%] rounded-xl h-auto mx-10 p-10 ">
                    <form action="" onSubmit={handleSumbit}>
                        
                        <div className="flex flex-col mb-5">
                            <label htmlFor="" className="text-white">Name:</label>
                            <input 
                                type="text" 
                                name='username'
                                onChange={(e) => setValues({...values,username: e.target.value})}
                                className="rounded-lg p-2 border-black border-[1px] " 
                                placeholder="Write Your Name..."
                            />
                        </div>

                        <div className="flex flex-col mb-5">
                            <label htmlFor="" className="text-white">E-mail/G-mail:</label>
                            <input 
                                type="text" 
                                name='gmail'
                                onChange={(e) => setValues({...values,gmail: e.target.value})}
                                className="rounded-lg p-2 border-black border-[1px]"
                                placeholder="Write Your gmail..."
                                />
                        </div>
                        
                        <div className="flex flex-col mb-5">
                            <label htmlFor="" className="text-white">Password:</label>
                            <input 
                                type="text" 
                                name='password' 
                                onChange={(e) => setValues({...values,password: e.target.value})}
                                className="rounded-lg p-2 border-black border-[1px]" 
                                placeholder="Enter the password..."
                            />
                        </div>

                        <div className="">
                            <button className="bg-slate-500 rounded-lg w-full text-white text-[20px] py-2 hover:bg-slate-400 hover:transition-all">Register Now</button> 
                        </div>

                    </form>
                </div>
            </div>
    </div>
  )
}

export default Register