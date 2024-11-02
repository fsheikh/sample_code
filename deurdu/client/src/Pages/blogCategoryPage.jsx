import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import Navbar from '../Components/Navbar';
import Footer from '../Components/Footer'
import axios from 'axios';


const BlogCategoryPage = () => {
  const { category, lng } = useParams(); 
  const [blogs, setBlogs] = useState([]);
  // const selectedLanguage = localStorage.getItem('language') || 'en';
  console.log('Selected Language:', lng); // Check the retrieved language

  useEffect(() => {
    // Fetch blogs based on category and selected language
    let language = 'english'
    if(lng === 'de')
    {
      language = 'german'
    }
    else if(lng === 'ur')
    {
      language = 'urdu'
    }
    axios.get(`http://localhost:3030/blogs/selected-category?category=${category}&language=${language}`)
      .then(response => {
        console.log('Fetched blogs:', response.data.result); // Debugging line
        setBlogs(response.data.result || []);
      })
      .catch(error => console.error('Error fetching blogs:', error));
  }, [category, lng]); // Add `lng` to the dependency array
  


  return (
    <div>
      <Navbar />
      <h1 className='flex justify-center items-center p-5 text-[20px] font-semibold'>Blogs for <span className='ml-1 hover:underline'>{category}</span></h1>
      <div className="flex flex-wrap">
        {blogs.length > 0
          ? blogs.map(blog => (
              <div className="flex flex-col w-[250px] m-6 border-2 border-black text-center p-2 rounded-lg hover:shadow-2xl hover:duration-100" key={blog.id}>
                <img src={blog.image ? `http://localhost:3030/pictures/${blog.image}` : "https://via.placeholder.com/150"} className='p-2 bg-slate-700 rounded-lg' alt="Blog Image" />
                <hr className='m-1 bg-gray-500 h-0.5' />
                <h2 className='font-semibold'>{blog.blog_title}</h2>
                <hr className='m-1 bg-gray-500 h-0.5' />
                <h5 className='whitespace-pre-wrap break-words'>{blog.blog_content}</h5>
                <hr className='m-1 bg-gray-500 h-0.5' />
                <h3 className='text-[15px]'>Blog language: <span className='font-bold'>{blog.blog_language}</span></h3>
                {blog.author_name && (
                  <h3 className='text-[15px]'>Author Name: <span className='font-bold'>{blog.author_name}</span></h3>
                )}
                <h3 className='text-[15px]'>Blog Category: <span className='font-bold'>{blog.blog_category}</span></h3>
                <button className='bg-slate-500 text-white rounded-md px-1 py-2 mt-5 hover:bg-slate-400 transition-all'>Read More</button>
              </div>
          ))
          : 
          <h2 className='flex justify-center items-center ml-[40%] mb-[20%]'>No blogs available for this category</h2>
        }
      </div>
      <Footer/>
    </div>
  );
};

export default BlogCategoryPage;
