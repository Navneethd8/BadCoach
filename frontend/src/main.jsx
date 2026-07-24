import React, { Suspense, lazy } from 'react'
import ReactDOM from 'react-dom/client'
import ReactGA from 'react-ga4'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './fonts.css'
import './index.css'
import { ThemeProvider } from './context/ThemeContext.jsx'
import JsonLd from './seo/JsonLd.jsx'

const LandingPage = lazy(() => import('./components/LandingPage.jsx'))
const App = lazy(() => import('./App.jsx'))
const LiveSession = lazy(() => import('./components/LiveSession.jsx'))
const PrivacyPage = lazy(() => import('./components/PrivacyPage.jsx'))
const TermsPage = lazy(() => import('./components/TermsPage.jsx'))
const FaqPage = lazy(() => import('./components/FaqPage.jsx'))
const GlossaryPage = lazy(() => import('./components/GlossaryPage.jsx'))
const CompareBadmintonPeakPage = lazy(() => import('./components/CompareBadmintonPeakPage.jsx'))
const CompareKreedaPage = lazy(() => import('./components/CompareKreedaPage.jsx'))
const NotFoundPage = lazy(() => import('./components/NotFoundPage.jsx'))

ReactGA.initialize('G-TET6JN36Q4')

function RouteFallback() {
    return (
        <div className="theme-page flex min-h-screen items-center justify-center text-sm text-[var(--text-muted)]">
            Loading…
        </div>
    )
}

ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
        <ThemeProvider>
            <BrowserRouter>
                <JsonLd />
                <Suspense fallback={<RouteFallback />}>
                    <Routes>
                        <Route path="/" element={<LandingPage />} />
                        <Route path="/analyze" element={<App />} />
                        <Route path="/live" element={<LiveSession />} />
                        <Route path="/faq" element={<FaqPage />} />
                        <Route path="/glossary" element={<GlossaryPage />} />
                        <Route path="/compare/badmintonpeak" element={<CompareBadmintonPeakPage />} />
                        <Route path="/compare/kreeda" element={<CompareKreedaPage />} />
                        <Route path="/privacy" element={<PrivacyPage />} />
                        <Route path="/terms" element={<TermsPage />} />
                        <Route path="*" element={<NotFoundPage />} />
                    </Routes>
                </Suspense>
            </BrowserRouter>
        </ThemeProvider>
    </React.StrictMode>,
)
