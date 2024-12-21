import React from 'react'

const Cards = ({img,title,description}) => {
    return (
        <div className='flex flex-col w-[250px] m-6 border-2 border-black text-center p-2 rounded-lg'>
            <img src={img} alt="Placeholder Image" className='p-2 bg-slate-700 rounded-lg'/>
            <hr className='m-1 bg-gray-500 h-0.5'/>
            <h2 className='font-semibold text-center justify-center'>{title}</h2>
            <hr className='m-1 bg-gray-500 h-0.5'/>
            <h5>{description}</h5>
            <button className='bg-slate-500 text-white rounded-md px-1 py-2 mt-5 hover:bg-slate-400 transition-all'>Read More</button>
        </div>
    )
}

export default Cards