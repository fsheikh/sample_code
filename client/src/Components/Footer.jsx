import React from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";

const Footer = () =>
{
    let currentYear = new Date().getFullYear();
    const {t} = useTranslation();
    const {ul1,ul2,ul3,ul4} = t('ulTranslation')

    return(
        <div className="bg-green-200 py-5 flex space-x-[1050px]">
            <ul className="ml-10 space-y-4">
                <Link to={'/'}>
                    <li className="hover:underline transition cursor-pointer">{ul1}</li>
                </Link>
                <li className="hover:underline transition cursor-pointer">{ul2}</li>
                <li className="hover:underline transition cursor-pointer">{ul3}</li>
                <li className="hover:underline transition cursor-pointer">{ul4}</li>
        
            </ul>
            <div className="flex flex-col">
                <h3>DeUrdu</h3>
                <span>All rights are reserved by @DeUrdu - {currentYear}</span>
                <hr className="h-0.5 bg-gray-500 mb-3 mt-3"/>
                <img src="https://starfinderfoundation.org/wp-content/uploads/2015/07/Facebook-logo-blue-circle-large-transparent-png.png" 
                alt="fbLogo"
                className="w-10 h-10 flex justify-center text-center"/>
            </div>
        </div>
    );
}

export default Footer;