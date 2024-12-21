import React, { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {faGlobe} from '@fortawesome/free-solid-svg-icons'
import i18 from 'i18next'
import { useAuth0 } from "@auth0/auth0-react";


const Navbar = () => {
  const [dropdownVisible, setDropdownVisible] = useState(false);
  const {t} =  useTranslation();
  const {lng} = useParams(); //capture the language from url
  const navigate = useNavigate();
  const { user, isLoading, isAuthenticated, loginWithRedirect, logout } = useAuth0();

  const {ul1,ul2,ul3,ul4,ul5} = t('ulTranslation')
  const {login,register,logOut} = t('logRegTranslation')
  const {Technology,Music,Literature,Politics,Culture} = t('blogTopics')

  useEffect(()=>{
    const savedLang = localStorage.getItem('i18nextLng') || 'en';
    i18.changeLanguage(savedLang);
  },[])
  
  const handleLanguageChange = (lang) => {
    const currentPath = window.location.pathname;
    const newPath = currentPath.replace(`/${i18.language}`, `/${lang}`);
    i18.changeLanguage(lang);
    localStorage.setItem('i18nextLng', lang);
    navigate(newPath);
    setDropdownVisible(false);
  };
  
  // const getLanguagePath = () => `/${i18.language}`

  if(isLoading)
  {
    return <div>Loading...</div>
  }

  return (
    <div className='gap-[1150px]'>

      {/* This is the 1st navbar */}
      <div className= "bg-green-700}">
        <div className={`flex bg-green-700 gap-[1150px] ${i18.language === 'ur' ? 'gap-[1130px]' : 'gap-[1150px'}
                                                          ${i18.language === 'de' ? 'gap-[1090px]' : 'gap-[1140px'}`}>
          <div className="flex flex-row items-center">
            <Link to={`/${i18.language}`}>
              <h3 className='text-white mr-3 ml-4'>DeUrdu</h3>
            </Link>
            <Link to={`/${i18.language}`}>
              <h3 className='text-white'> ڈی اردو </h3>
            </Link>
          </div>
            <div className="flex gap-1">
               {/* Globe button with onClick to toggle dropdown */}
              <button 
                className='px-2 py-0.5 m-1 rounded-lg text-white relative'
                onClick={() => setDropdownVisible(!dropdownVisible)}
              >
                <FontAwesomeIcon icon={faGlobe} className='mr-1' />
                {i18.language.toUpperCase()}
              </button>

              {/* Dropdown menu */}
              {dropdownVisible && (
                <div className='absolute mt-2 bg-gray-200 rounded-md shadow-lg'>
                  <button 
                    className='block px-4 py-2 text-black hover:bg-gray-300'
                    onClick={() => handleLanguageChange('en')}
                  >
                    English (EN)
                  </button>
                  <button 
                    className='block px-4 py-2 text-black hover:bg-gray-300'
                    onClick={() => handleLanguageChange('de')}
                  >
                    German (De)
                  </button>
                  <button 
                    className='block px-[25px] py-2 text-black hover:bg-gray-300'
                    onClick={() => handleLanguageChange('ur')}
                  >
                    Urdu (UR)
                  </button>
                </div>
              )}
              {/* <Link to={`${getLanguagePath()}/login`}>
                <button className='px-2 py-0.5 m-1 rounded-lg text-white'>{login}</button>
              </Link>
              <Link to={`${getLanguagePath()}/register`}>
                <button className='px-2 py-0.5 m-1 rounded-lg text-white'>{register}</button>
              </Link> */}
              {isAuthenticated
                ?
                  <>
                    <img src={user.picture} alt={user.name} className='rounded-full w-10 p-1'/>
                    <button className='px-2 py-0.5 m-1 rounded-lg text-white' 
                      onClick={() => logout({ logoutParams: { returnTo: window.location.origin },clientId: "NyjTsxGX6tAxOxKL6JkjzR6hY7MKhHn4" })}>
                        {logOut}
                    </button>
                  </>
                :
                  <>
                    <button className='px-2 py-0.5 m-1 rounded-lg text-white' 
                      onClick={() => loginWithRedirect()}>
                        {register}
                    </button>
                  </>
              }
            </div>
        </div>
      </div>

        {/* This is 2nd Navbar */}
        <div className={`flex bg-gray-700 ${i18.language === 'ur' ? 'gap-[775px]' : 'gap-[880px]'}
                                            ${i18.language === 'de' ? 'gap-[700px]' : 'gap-[880px]'}`}>

          <div className='flex gap-10 py-2'>
            <Link to={`/${i18.language}`}>
              <button className='text-white font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white'>{ul1}</button>
            </Link>
              <button className='text-white font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white'>{ul2}</button>
              <button className='text-white font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white'>{ul3}</button>
          </div>
          <div className="flex gap-6">
            <Link to={`/${i18.language}/post-blog`}>
              <button className='text-white bg-gray-500 px-2 my-2 rounded-lg hover:bg-gray-500 transition py-1 hover:text-white'>{ul4}</button>
            </Link>
            <Link to={`/${i18.language}/search-blog`}>
              <button className='text-white bg-gray-500 px-2 my-2 rounded-lg hover:bg-gray-500 transition py-1 hover:text-white'>{ul5}</button>
            </Link>
          </div>
        </div>
        
        {/* This is 3rd Navbar */}
          <div className='flex justify-center gap-[100px] py-2'>
            <Link to={`/${i18.language}/blog/literature`}>
              <button className='text-black font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white'>{Literature}</button>
            </Link>
            <Link to={`/${i18.language}/blog/music`}>
              <button className='text-black font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white'>{Music}</button>
            </Link>
            <Link to={`/${i18.language}/blog/politics`}>
              <button className='text-black font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white'>{Politics}</button>
            </Link>
            <Link to={`/${i18.language}/blog/culture`}>
              <button className='text-black font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white'>{Culture}</button>
            </Link>
            <Link to={`/${i18.language}/blog/technology`}>
              <button className='text-black font-medium ml-4 hover:bg-gray-500 transition px-3 py-1 rounded-lg hover:text-white'>{Technology}</button>
            </Link>
          </div>
          <hr className='bg-gray-200 h-1'/>

    </div>
  )
}

export default Navbar