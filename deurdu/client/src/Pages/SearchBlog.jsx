import React, { useEffect, useState } from 'react'
import Navbar from '../Components/Navbar'
import Footer from '../Components/Footer'
import axios from 'axios'

const SearchBlog = () => {
  const [blogs, setBlogs] = useState([]);
  const [authorName, setAuthorName] = useState('');
  const [category, setCategory] = useState('');
  const [language, setLanguage] = useState('');

  // Fetch top 20 blogs initially
  useEffect(() => {
    fetchBlogs();
  }, []);

  // Function to fetch blogs based on filters
  const fetchBlogs = (filterCategory = category, filterAuthor = authorName, filterLanguage = language) => {
    axios.get(`http://localhost:3030/blogs/search?category=${filterCategory}&author_name=${filterAuthor}&language=${filterLanguage}&limit=20`)
      .then(response => {
        console.log('Blogs received on frontend: ', response.data.result);
        setBlogs(response.data.result || []);
      })
      .catch(error => console.error('Error fetching blogs: ', error));
  };
  

  // Handle Category change
  const handleCategoryChange = (event) => {
    const selectedCategory = event.target.value;
    setCategory(selectedCategory);
    fetchBlogs(selectedCategory, authorName, language); // Fetch with updated category
  };

  // Handle Author name change
  const handleAuthorChange = (event) => {
    const name = event.target.value;
    setAuthorName(name);
    fetchBlogs(category, name, language); // Fetch with updated author
  };

  const handleLanguageChange = (event) => {
    const selectedLanguage = event.target.value;
    setLanguage(selectedLanguage);
    console.log('Selected language:', selectedLanguage);
    fetchBlogs(category, authorName, selectedLanguage); // Fetch with updated language
  };
  

  return (
    <>
      <Navbar />

      {/* Input fields for filtering */}
      <div className="">
        <div className='mt-14 mb-10 flex gap-52 text-center justify-center'>
          {/* Author Name Input */}
          <div className="flex flex-col">
            <label htmlFor="">Write Author Name:</label>
            <input
              type="text"
              placeholder='Write Author Name...'
              className='border-[2px] border-black rounded-lg pl-1'
              value={authorName}
              onChange={handleAuthorChange}
            />
          </div>

          {/* Blog Category Dropdown */}
          <div className="flex flex-col justify-center text-center">
            <label htmlFor="">Select Blog-Category</label>
            <select value={category} onChange={handleCategoryChange} className='border-[1px] border-black bg-gray-100'>
              <option value="" disabled>Select</option>
              <option value="Technology">Technology</option>
              <option value="Politics">Politics</option>
              <option value="Literature">Literature</option>
              <option value="Music">Music</option>
              <option value="Culture">Culture</option>
            </select>
          </div>

          {/* Blog Language Dropdown */}
          <div className="flex flex-col">
            <label htmlFor="">Select Blog-Language:</label>
            <select value={language} onChange={handleLanguageChange} className='border-[1px] border-black bg-gray-100'>
              <option value="" disabled>Select</option>
              <option value="English">English</option>
              <option value="German">German</option>
              <option value="Urdu">Urdu</option>
            </select>
          </div>
        </div>
      </div>

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
            No blogs available for this category
          </h2>
        }
      </div>

      <Footer />
    </>
  )
}

export default SearchBlog;
