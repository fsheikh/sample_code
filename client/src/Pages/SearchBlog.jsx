import React, { useEffect, useState } from 'react';
import Navbar from '../Components/Navbar';
import Footer from '../Components/Footer';
import axios from 'axios';

const SearchBlog = () => {
  const [blogs, setBlogs] = useState([]);
  const [authorName, setAuthorName] = useState('');
  const [category, setCategory] = useState('');
  const [language, setLanguage] = useState('');
  const [topAuthors, setTopAuthors] = useState([]);
  const [selectedAuthor, setSelectedAuthor] = useState("");

  // Fetch all blogs initially
  useEffect(() => {
    fetchBlogs();
  }, []);

  // Fetch top authors when the component loads
  useEffect(() => {
    axios.get("http://localhost:3030/blogs/top-authors")
      .then(response => setTopAuthors(response.data))
      .catch(error => console.error("Error fetching authors:", error));
  }, []);

  // Function to fetch blogs based on filters
  const fetchBlogs = (filterCategory = '', filterAuthor = '', filterLanguage = '') => {
    axios
      .get(`http://localhost:3030/blogs/search`, {
        params: {
          category: filterCategory,
          author_name: filterAuthor,
          language: filterLanguage,
          limit: 20,
        },
      })
      .then((response) => {
        console.log('Blogs received on frontend: ', response.data.blogs);
        setBlogs(response.data.blogs || []);
      })
      .catch((error) => console.error('Error fetching blogs: ', error));
  };

  // Handle Category change
  const handleCategoryChange = (event) => {
    const selectedCategory = event.target.value;
    setCategory(selectedCategory);
    fetchBlogs(selectedCategory, authorName, language);
  };

  // Handle Author name change from text input
  const handleAuthorChange = (event) => {
    const name = event.target.value;
    setAuthorName(name);
    fetchBlogs(category, name, language);
  };

  // Handle Language change
  const handleLanguageChange = (event) => {
    const selectedLanguage = event.target.value;
    setLanguage(selectedLanguage);
    fetchBlogs(category, authorName, selectedLanguage);
  };

  // Handle Top Author selection from dropdown
  const handleSelectedAuthorChange = (event) => {
    const author = event.target.value;
    setSelectedAuthor(author);
    fetchBlogs(category, author, language); // Fetch blogs of the selected top author
  };

  return (
    <>
      <Navbar />

      {/* Input fields for filtering */}
      <div className="mt-14 mb-10 flex gap-52 text-center items-center justify-center">
        {/* Author Name Input */}
        <div className="flex flex-col">
          <label>Write Author Name:</label>
          <input
            type="text"
            placeholder="Write Author Name..."
            className="border-[2px] border-black rounded-lg pl-1"
            value={authorName}
            onChange={handleAuthorChange}
          />
        </div>

        <div className="flex flex-col gap-2">
          <label>Select Top Author:</label>
          <select
            className="border-[2px] border-black rounded-lg pl-1"
            value={selectedAuthor}
            onChange={handleSelectedAuthorChange}
          >
            <option value="">-- Select an Author --</option>
            {topAuthors.map((author, index) => (
              <option key={index} value={author.author_name}>
                {author.author_name} ({author.blog_count} posts)
              </option>
            ))}
          </select>
        </div>

        {/* Blog Category Dropdown */}
        <div className="flex flex-col justify-center text-center">
          <label>Select Blog-Category</label>
          <select value={category} onChange={handleCategoryChange} className="border-[1px] border-black bg-gray-100">
            <option value="">All</option>
            <option value="Technology">Technology</option>
            <option value="Politics">Politics</option>
            <option value="Literature">Literature</option>
            <option value="Music">Music</option>
            <option value="Culture">Culture</option>
          </select>
        </div>

        {/* Blog Language Dropdown */}
        <div className="flex flex-col">
          <label>Select Blog-Language:</label>
          <select value={language} onChange={handleLanguageChange} className="border-[1px] border-black bg-gray-100">
            <option value="">All</option>
            <option value="English">English</option>
            <option value="German">German</option>
            <option value="Urdu">Urdu</option>
          </select>
        </div>
      </div>

      {/* Displaying the Blogs */}
      <div className="flex flex-wrap justify-center">
        {blogs.length > 0 ? (
          blogs.map((blog) => (
            <div className="flex flex-col w-[250px] m-6 border-2 border-black text-center p-2 rounded-lg" key={blog.id}>
              <img
                src={`http://localhost:3030/pictures/${blog.image}` || 'https://via.placeholder.com/150'}
                className="p-2 bg-slate-700 rounded-lg"
                alt="Blog Image"
              />
              <hr className="m-1 bg-gray-500 h-0.5" />
              <h2 className="font-semibold">{blog.blog_title}</h2>
              <hr className="m-1 bg-gray-500 h-0.5" />
              <h5 className="whitespace-pre-wrap break-words">{blog.blog_content}</h5>
              <hr className="m-1 bg-gray-500 h-0.5" />
              <h3 className="text-[15px]">
                Blog language: <span className="font-bold">{blog.blog_language}</span>
              </h3>
              <h3 className="text-[15px]">
                Author Name: <span className="font-bold">{blog.author_name}</span>
              </h3>
              <h3 className="text-[15px]">
                Blog Category: <span className="font-bold">{blog.blog_category}</span>
              </h3>
              <button className="bg-slate-500 text-white rounded-md px-1 py-2 mt-5 hover:bg-slate-400 transition-all">
                Read More
              </button>
            </div>
          ))
        ) : (
          <h2 className="flex justify-center items-center">No blogs available</h2>
        )}
      </div>

      <Footer />
    </>
  );
};

export default SearchBlog;