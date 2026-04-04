import React from 'react'
import ReactDOM from 'react-dom/client'
import ReactGA from "react-ga4"
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import App from './App.jsx'
import LandingPage from './components/LandingPage.jsx'
import LiveSession from './components/LiveSession.jsx'
import PrivacyPage from './components/PrivacyPage.jsx'
import TermsPage from './components/TermsPage.jsx'
import './index.css'

ReactGA.initialize("G-TET6JN36Q4")

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <BrowserRouter>
            <Routes>
                <Route path="/" element={<LandingPage />} />
                <Route path="/analyze" element={<App />} />
                <Route path="/live" element={<LiveSession />} />
                <Route path="/privacy" element={<PrivacyPage />} />
                <Route path="/terms" element={<TermsPage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
        </BrowserRouter>
    </React.StrictMode>,
)
