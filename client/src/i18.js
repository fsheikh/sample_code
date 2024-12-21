import i18 from 'i18next'
import LanguageDetector from 'i18next-browser-languagedetector'
import {initReactI18next} from 'react-i18next'
import Backend from 'i18next-http-backend'


i18.use(LanguageDetector).use(initReactI18next).use(Backend).init({
        debug:true,
        fallbackLng:'en',
        returnObjects: true,
        backend: {
            loadPath: '/locales/{{lng}}/translation.json', // This is the path to your translation files
        },
    })