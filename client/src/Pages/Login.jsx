import React, { useEffect, useState } from "react";
import Navbar from "../Components/Navbar";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import i18 from 'i18next'
import { Link } from "react-router-dom";

const Login = () =>{

    const [values,setValues] = useState({
        gmail: '',
        password: ''
    })

    const [isloggedIn, setisloggedIn] = useState(false);

    useEffect(() => {
        axios.get('http://localhost:3030/auth/status', { withCredentials: true })
            .then(response => {
                console.log('Auth status:', response.data);
                setisloggedIn(response.data.loggedIn);
            })
            .catch(err => {
                console.error('Error checking login status:', err);
            });
    }, [isloggedIn]);
    
    

    const navigate = useNavigate();

    const getLanguagePath = () => `/${i18.language}`

    axios.defaults.withCredentials = true;

    const handleSumbit = (e) =>
        {
            e.preventDefault();
            if(isloggedIn)
            {
                alert('You are already logged in');
                return;
            }
            if(values.gmail==='' || values.password==='')
            {
                alert('Make sure to fill both fields.')
                return;
            }
            axios.post('http://localhost:3030/auth/login', values)
            .then(res => 
                {
                    console.log('Server response:', res.data); // Log the response data
                    if(res.data.Status === 'Success') 
                    {
                        navigate(`${getLanguagePath()}/`);
                    }
                    else if(res.data.Error==='No email existed')
                    {
                        alert('Incorrect Email :/')
                    }
                    else if(res.data.Error==='Password not matched')
                    {
                        alert('Incorrect Password :/')
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


    return(
        <div className="">
            <Navbar/>
            <div className="flex flex-col justify-center items-center">
                <h1 className="text-[50px] font-semibold mt-6">Login</h1>
                <div className="flex flex-col bg-slate-300 w-[40%] rounded-xl h-auto mx-10 p-10 ">

                    <form action="" onSubmit={handleSumbit}>

                        <div className="flex flex-col mb-5">
                            <label htmlFor="" className="text-white">E-mail/G-mail:</label>
                            <input 
                                type="text" 
                                name="gmail"
                                onChange={(e)=> setValues({...values,gmail: e.target.value})}
                                className="rounded-lg p-2 border-black border-[1px]"
                                placeholder="Write Your gmail..."
                            />
                        </div>
                        
                        <div className="flex flex-col mb-5">
                            <label htmlFor="" className="text-white">Password:</label>
                            <input 
                                type="text"
                                name="password"
                                onChange={(e) => setValues({...values,password: e.target.value})}
                                className="rounded-lg p-2 border-black border-[1px]" 
                                placeholder="Enter the password..."
                                />
                        </div>

                        <div className="">
                            <button className="bg-slate-500 rounded-lg w-full text-white text-[20px] py-2 hover:bg-slate-400 hover:transition-all" disabled={isloggedIn}>Login Now</button>
                        </div>
                    </form>
                    <Link to={`${getLanguagePath()}/register`}>
                        <button className="bg-slate-400 rounded-lg w-full text-white text-[20px] my-5 py-2 hover:bg-slate-500 hover:transition-all">Create a new Account</button> 
                    </Link>

                </div>
            </div>
        </div>
    )
}

export default Login;