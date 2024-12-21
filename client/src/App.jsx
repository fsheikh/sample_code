import Home from "./Pages/Home"
import { BrowserRouter, Route, Routes,Navigate} from 'react-router-dom'
import PostBlog from "./Pages/PostBlog"
import SearchBlog from "./Pages/SearchBlog"
import BlogCategoryPage from "./Pages/blogCategoryPage"
import i18 from 'i18next'
import Login from "./Pages/Login"
import Register from "./Pages/Register"

function App() {

  // const {lng} = useParams();

  return (
    <>
    <BrowserRouter>
      <Routes>
        {/* Fallback route to redirect to default language 'en' */}
        <Route path="/" element={<Navigate to={`/${i18.language}`}/>} />

        <Route path={"/:lng"} element={<Home/>}/>
        <Route path={"/:lng/login"} element={<Login/>}/>
        <Route path={"/:lng/register"} element={<Register/>}/>
        <Route path="/:lng/post-blog" element={<PostBlog/>}/>
        <Route path="/:lng/search-blog" element={<SearchBlog/>}/>
        <Route path="/:lng/blog/:category" element={<BlogCategoryPage/>} />
      </Routes>
    </BrowserRouter>
    </>
  )
}

export default App