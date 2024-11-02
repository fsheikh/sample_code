import React, { useState } from 'react';
import Navbar from '../Components/Navbar';
import Footer from '../Components/Footer';
// import InputBar from '../Components/InputBar';
import { useTranslation } from 'react-i18next';
import axios from 'axios';
import { useAuth0 } from '@auth0/auth0-react';


const PostBlog = () => {
  const { t } = useTranslation();
  const { AuthorName, PreviewImage, BlogCategory, BlogLang, BlogTitle, BlogContent } = t('postBlog');
  const { Literature, Music, Politics, Culture, Technology } = t('blogTopics');
  const { English, German, Urdu } = t('blogLang');
  const { isAuthenticated } = useAuth0();

  const [formData, setFormData] = useState({
    author_name: '',
    image: null,
    blog_category: '',
    blog_language: '',
    blog_title: '',
    blog_content: '',
  });

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: files ? files[0] : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const formDataObj = new FormData();
    Object.entries(formData).forEach(([key, value]) => {
      formDataObj.append(key, value);
    });
  
    try {
      const response = await axios.post('http://localhost:3030/blogs/postblog', formDataObj, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      console.log(response.data);
      alert('Blog Added Successfully');
      setFormData({
        author_name: '',
        image: null,
        blog_category: '',
        blog_language: '',
        blog_title: '',
        blog_content: '',
      });
    } catch (error) {
      console.log('Error submitting the blog: ', error);
      alert('Failed to add the blog');
    }
  };
  

  return (
    <>
      <Navbar />
      {isAuthenticated
        ?
        <form onSubmit={handleSubmit}>
          <div className="mt-20 mb-10">
            <div className="flex gap-20 ml-10 mb-5">
              <div className="flex flex-col">
                <label htmlFor="">{AuthorName}</label>
                <input type="text"
                  className='border-[2px] border-black rounded-lg pl-1'
                  required 
                  placeholder='Author Name'
                  name='author_name'
                  value={formData.author_name}
                  onChange={handleChange}
                  />
              </div>
              <div className="flex flex-col">
                <label htmlFor="">{PreviewImage}</label>
                <input type="file"
                  className='border-[2px] border-black rounded-lg pl-1'
                  name='image'
                  value={formData.image}
                  onChange={handleChange}  
                />
              </div>
            </div>

            <div className="flex gap-28 ml-10 mb-5">
              <div className="flex flex-col">
                <label htmlFor="blog_category" className="p-1">
                  {BlogCategory}
                </label>
                <select
                  required
                  name="blog_category"
                  value={formData.blog_category}
                  onChange={handleChange}
                  className="border-[1px] border-black bg-gray-100 mt-2"
                >
                  <option value="" disabled selected>
                    select
                  </option>
                  <option value="literature">{Literature}</option>
                  <option value="politics">{Politics}</option>
                  <option value="technology">{Technology}</option>
                  <option value="music">{Music}</option>
                  <option value="culture">{Culture}</option>
                </select>
              </div>
              <div className="flex flex-col">
                <label htmlFor="blog_language" className="p-1">
                  {BlogLang}
                </label>
                <select
                  required
                  name="blog_language"
                  value={formData.blog_language}
                  onChange={handleChange}
                  className="border-[1px] border-black bg-gray-100 mt-2"
                >
                  <option value="" disabled selected>
                    select
                  </option>
                  <option value="english">{English}</option>
                  <option value="german">{German}</option>
                  <option value="urdu">{Urdu}</option>
                </select>
              </div>
            </div>

            <div className="flex flex-col border-[1px] border-black rounded-xl w-[650px] ml-10 mt-10 p-1">
              <input
                required
                type="text"
                placeholder={BlogTitle}
                className="text-3xl border-none mb-5 m-1 focus:outline-none focus:border-transparent"
                name="blog_title"
                value={formData.blog_title}
                onChange={handleChange}
              />

              <textarea
                required
                placeholder={BlogContent}
                className="h-[300px] m-1 text-xl focus:outline-none focus:border-transparent"
                name="blog_content"
                value={formData.blog_content}
                onChange={handleChange}
              ></textarea>
            </div>
            <button
              type="submit"
              className="bg-green-500 rounded-2xl w-[650px] p-1 py-3 m-4 ml-10 hover:bg-green-400 hover:transition-all duration-200"
            >
              Post
            </button>
          </div>
        </form>
        :
        <>
          <h1 className='absolute top-[30%] left-[30%] p-4 font-semibold text-[50px] mx-[30%]'>Make sure to Login for Positng the blog</h1>
            <form onSubmit={handleSubmit}>
            <div className="mt-20 mb-10">
              <div className="flex gap-20 ml-10 mb-5">
                <div className="flex flex-col">
                  <label htmlFor="">{AuthorName}</label>
                  <input type="text"
                    className='border-[2px] border-black rounded-lg pl-1 cursor-not-allowed'
                    required 
                    placeholder='Author Name'
                    name='author_name'
                    value={formData.author_name}
                    onChange={handleChange}
                    disabled={true}
                  />
                </div>
                <div className="flex flex-col">
                  <label htmlFor="">{PreviewImage}</label>
                  <input type="file"
                    className='border-[2px] border-black rounded-lg pl-1 cursor-not-allowed'
                    name='image'
                    value={formData.image}
                    onChange={handleChange}
                    disabled={true}
                  />
                </div>
              </div>

              <div className="flex gap-28 ml-10 mb-5">
                <div className="flex flex-col">
                  <label htmlFor="blog_category" className="p-1">
                    {BlogCategory}
                  </label>
                  <select
                    required
                    name="blog_category"
                    value={formData.blog_category}
                    onChange={handleChange}
                    className="border-[1px] border-black bg-gray-100 mt-2 cursor-not-allowed"
                    disabled={true}
                  >
                    <option value="" disabled selected>
                      select
                    </option>
                    <option value="literature">{Literature}</option>
                    <option value="politics">{Politics}</option>
                    <option value="technology">{Technology}</option>
                    <option value="music">{Music}</option>
                    <option value="culture">{Culture}</option>
                  </select>
                </div>
                <div className="flex flex-col">
                  <label htmlFor="blog_language" className="p-1">
                    {BlogLang}
                  </label>
                  <select
                    required
                    name="blog_language"
                    value={formData.blog_language}
                    onChange={handleChange}
                    className="border-[1px] border-black bg-gray-100 mt-2 cursor-not-allowed"
                    disabled={true}
                  >
                    <option value="" disabled selected>
                      select
                    </option>
                    <option value="english">{English}</option>
                    <option value="german">{German}</option>
                    <option value="urdu">{Urdu}</option>
                  </select>
                </div>
              </div>

              <div className="flex flex-col border-[1px] border-black rounded-xl w-[650px] ml-10 mt-10 p-1">
                <input
                  required
                  type="text"
                  placeholder={BlogTitle}
                  className="text-3xl border-none mb-5 m-1 focus:outline-none cursor-not-allowed focus:border-transparent"
                  name="blog_title"
                  value={formData.blog_title}
                  onChange={handleChange}
                  disabled={true}
                />

                <textarea
                  required
                  placeholder={BlogContent}
                  className="h-[300px] m-1 text-xl focus:outline-none cursor-not-allowed focus:border-transparent"
                  name="blog_content"
                  value={formData.blog_content}
                  onChange={handleChange}
                  disabled={true}
                ></textarea>
              </div>
              <button
                type="submit"
                className="bg-green-500 rounded-2xl w-[650px] p-1 py-3 m-4 ml-10 cursor-not-allowed"
                disabled={true}
              >
                Post
              </button>
            </div>
          </form>
        </>
      }
      <Footer />
    </>
  );
};

export default PostBlog;
