import React from 'react'

const InputBar = ({title,inputType,placeholder}) => {
  return (
    <div className='flex flex-col'>
        <label htmlFor="">{title}</label>
        <input type={inputType} className='border-[2px] border-black rounded-lg pl-1' placeholder={placeholder}/>
    </div>
  )
}

export default InputBar