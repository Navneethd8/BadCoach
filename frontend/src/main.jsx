import React, { useState } from 'react'
import ReactDOM from 'react-dom/client'
import ReactGA from "react-ga4";
import App from './App.jsx'
import LandingPage from './components/LandingPage.jsx'
import './index.css'

ReactGA.initialize("G-TET6JN36Q4");

function Root() {
    const [showApp, setShowApp] = useState(false)

    if (showApp) {
        return <App onBack={() => setShowApp(false)} />
    }

    return <LandingPage onGetStarted={() => setShowApp(true)} />
}

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <Root />
    </React.StrictMode>,
)
