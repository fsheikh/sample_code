import React, { useState, useEffect } from 'react';
import Navbar from '../Components/Navbar';
import Footer from '../Components/Footer';
import ImagesSlidebar from '../Components/ImagesSlidebar';
import axios from 'axios';

const Home = () => {
  const [blogs, setBlogs] = useState([]);

  // Fetch top 20 recent blogs on component mount
  useEffect(() => {
    fetchBlogs();
  }, []);

  // Function to fetch recent blogs (limit 20)
  const fetchBlogs = () => {
    axios.get(`http://localhost:3030/blogs?limit=20`)
      .then(response => {
        console.log('Blogs received on frontend: ', response.data.result);
        setBlogs(response.data.result || []);
      })
      .catch(error => console.error('Error fetching blogs: ', error));
  };

  return (
    <>
      <Navbar />
      <ImagesSlidebar />

      {/* Displaying the Blogs */}
      <div className="flex flex-wrap">
        {blogs.length > 0
          ? blogs.map(blog => (
            <div className="flex flex-col w-[250px] m-6 border-2 border-black text-center p-2 rounded-lg" key={blog.id}>
              <img src="https://via.placeholder.com/150" className='p-2 bg-slate-700 rounded-lg' alt="Blog Image" />
              <hr className='m-1 bg-gray-500 h-0.5' />
              <h2 className='font-semibold text-center justify-center'>{blog.blog_title}</h2>
              <hr className='m-1 bg-gray-500 h-0.5' />
              <h5 className='whitespace-pre-wrap break-words'>{blog.blog_content}</h5>
              <hr className='m-1 bg-gray-500 h-0.5' />
              <h3 className='text-[15px]'>Blog language: <span className='font-bold m-0 p-0'>{blog.blog_language}</span></h3>
              <h3 className='text-[15px]'>Author Name: <span className='font-bold m-0 p-0'>{blog.author_name}</span></h3>
              <h3 className='text-[15px]'>Blog Category: <span className='font-bold m-0 p-0'>{blog.blog_category}</span></h3>
              <button className='bg-slate-500 text-white rounded-md px-1 py-2 mt-5 hover:bg-slate-400 transition-all'>Read More</button>
            </div>
          ))
          : <h2 className='flex justify-center items-center ml-[40%]'>
            No blogs available
          </h2>
        }
      </div>

      <Footer />
    </>
  );
};

export default Home;
