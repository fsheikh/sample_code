import React, { useState,useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faGlobe } from '@fortawesome/free-solid-svg-icons';
import i18n from 'i18next';
import i18 from '../i18.js';
import { useTranslation } from 'react-i18next';

const Navbar = () => {
  const [dropdownVisible, setDropdownVisible] = useState(false);
  const { t } = useTranslation();
  const navigate = useNavigate();

  const { ul1, ul2, ul3, ul4, ul5 } = t('ulTranslation');
  const { login, register, logOut } = t('logRegTranslation');
  const { Technology, Music, Literature, Politics, Culture } = t('blogTopics');

  useEffect(() => {
    const savedLang = localStorage.getItem('i18nextLng') || 'en';
    i18.changeLanguage(savedLang);
  }, []);

  
  const handleLanguageChange = (lang) => {
    const currentPath = window.location.pathname;
    const newPath = currentPath.replace(`/${i18n.language}`, `/${lang}`);
    
    i18n.changeLanguage(lang); // Change the language
    localStorage.setItem('i18nextLng', lang); // Store the language
    navigate(newPath); // Navigate to the new path
    setDropdownVisible(false); // Hide the dropdown after changing language
  };

  return (
    <div className="gap-[1150px]">
      {/* First Navbar */}
      <div className="bg-green-700">
        <div
          className={`flex bg-green-700 gap-[1150px] ${i18n.language === 'ur' ? 'gap-[1130px]' : 'gap-[1150px]'}
                                                          ${i18n.language === 'de' ? 'gap-[1090px]' : 'gap-[1140px]'}`}
        >
          <div className="flex">
            <Link to={`/${i18n.language}`}>
              <h3 className="text-white text-[20px] mr-3 ml-4">DeUrdu</h3>
            </Link>
            <Link to={`/${i18n.language}`}>
              <h3 className={`${i18n.language === 'ur' ? 'urdu' : ''} text-[20px] text-white mx-2`}>ڈردو</h3>
            </Link>
          </div>
          <div className="flex gap-10 items-center">
            {/* Globe button to toggle language dropdown */}
            <button
              className="px-2 py-0.5 m-1 rounded-lg text-white relative"
              onClick={() => setDropdownVisible(!dropdownVisible)}
            >
              <FontAwesomeIcon icon={faGlobe} className="mr-1" />
              {i18n.language}
            </button>

            {/* Language Dropdown */}
            {dropdownVisible && (
                <div className="absolute mt-20 bg-gray-200 rounded-md shadow-lg">
                  {i18n.language !== 'en' && (
                    <button
                      className="block px-4 py-2 text-black hover:bg-gray-300"
                      onClick={() => handleLanguageChange('en')}
                    >
                      English (EN)
                    </button>
                  )}
                  {i18n.language !== 'de' && (
                    <button
                      className="block px-4 py-2 text-black hover:bg-gray-300"
                      onClick={() => handleLanguageChange('de')}
                    >
                      German (De)
                    </button>
                  )}
                  {i18n.language !== 'ur' && (
                    <button
                      className="block px-[25px] py-2 text-black hover:bg-gray-300"
                      onClick={() => handleLanguageChange('ur')}
                    >
                      Urdu (UR)
                    </button>
                  )}
                </div>
              )}
              <div className="">
                <Link to={`/${i18n.language}/login`}>
                  <button className="text-white text-[15px]  bg-gray-500 px-2 my-2 rounded-lg hover:bg-gray-500 transition py-2 hover:text-white">
                    {login}
                  </button>
                </Link>
              </div>
          </div>
        </div>
      </div>

      {/* Second Navbar */}
      <div className={`flex bg-gray-700 ${i18n.language === 'ur' ? 'urdu gap-[770px]' : 'gap-[880px]'}
                 ${i18n.language === 'de' ? 'gap-[700px]' : 'gap-[880px]'}`}>
        <div className="flex gap-10 py-2">
          <Link to={`/${i18n.language}`}>
            <button className="text-white text-[15px] font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white">
              {ul1}
            </button>
          </Link>
          <Link to={`/${i18n.language}/about-us`}>
            <button className="text-white text-[15px] font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white">
              {ul2}
            </button>
          </Link>
          <Link to={`/${i18n.language}/contact-us`}>
            <button className="text-white text-[15px] font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white">
              {ul3}
            </button>
          </Link>
        </div>
        <div className="flex gap-6">
          <Link to={`/${i18n.language}/post-blog`}>
            <button className="text-white text-[15px]  bg-gray-500 px-2 my-2 rounded-lg hover:bg-gray-500 transition py-2 hover:text-white">
              {ul4}
            </button>
          </Link>
          <Link to={`/${i18n.language}/search-blog`}>
            <button className="text-white text-[15px]  bg-gray-500 px-2 my-2 rounded-lg hover:bg-gray-500 transition py-2 hover:text-white">
              {ul5}
            </button>
          </Link>
        </div>
      </div>

      {/* Third Navbar */}
      <div className={`flex justify-center gap-[100px] py-2 ${i18n.language === 'ur' ? 'urdu' : ''}`}>
        <Link to={`/${i18n.language}/blog/literature`}>
          <button className="text-black text-[15px] font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white">
            {Literature}
          </button>
        </Link>
        <Link to={`/${i18n.language}/blog/music`}>
          <button className="text-black text-[15px] font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white">
            {Music}
          </button>
        </Link>
        <Link to={`/${i18n.language}/blog/politics`}>
          <button className="text-black text-[15px] font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white">
            {Politics}
          </button>
        </Link>
        <Link to={`/${i18n.language}/blog/culture`}>
          <button className="text-black text-[15px] font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white">
            {Culture}
          </button>
        </Link>
        <Link to={`/${i18n.language}/blog/technology`}>
          <button className="text-black text-[15px] font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white">
            {Technology}
          </button>
        </Link>
      </div>
      <hr className="bg-gray-200 h-1" />
    </div>
  );
};

export default Navbar;
