import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import axios from 'axios'
import ReactGA from 'react-ga4'
import Logo from './Logo'
import HeroFigmaBackdrop from './HeroFigmaBackdrop'

const BRAND = '#6c9c8d'
const PAGE_BG = '#fafafa'

const features = [
    {
        icon: 'directions_run',
        title: 'Pose tracing',
        description:
            'Split-step late? Contact behind you? See your body frame by frame so you fix the same miss twice, not twenty times.',
    },
    {
        icon: 'query_stats',
        title: 'Stroke reads',
        description:
            'What you hit, where it went, and how clean it looked — across ten real-world strokes, not textbook labels.',
    },
    {
        icon: 'tips_and_updates',
        title: 'Coaching notes',
        description:
            'Short, specific cues you can take to the hall: what to try on the next rep, not generic “keep practising.”',
    },
]

const flowSteps = [
    {
        label: 'upload a clip',
        image: '/phone-mockup.svg',
        imageSide: 'left',
    },
    {
        label: 'let birdzo do its thing',
        image: '/phone-mockup-alt.svg',
        imageSide: 'right',
    },
    {
        label: 'review the rally',
        image: '/phone-mockup.svg',
        imageSide: 'left',
    },
]

function Icon({ name, size = 20, className = '' }) {
    return (
        <span className={`material-symbols-outlined ${className}`} style={{ fontSize: size }}>
            {name}
        </span>
    )
}

function FigmaButton({ children, variant = 'pill', onClick, className = '' }) {
    const radius =
        variant === 'pill'
            ? 'rounded-bl-[25px] rounded-tr-[25px]'
            : 'rounded-bl-[8px] rounded-tr-[8px] shadow-[0_4px_4px_rgba(0,0,0,0.25)]'
    return (
        <button
            type="button"
            onClick={onClick}
            className={`figma-cta ${radius} ${className}`}
            style={{ backgroundColor: BRAND }}
        >
            {children}
        </button>
    )
}

function ShuttleDecor({ className = '', rotate = 0 }) {
    return (
        <img
            src="/shuttlecock.png"
            alt=""
            aria-hidden
            className={`shuttle-decor ${className}`}
            style={{ transform: `rotate(${rotate}deg)` }}
        />
    )
}

export default function LandingPage() {
    const navigate = useNavigate()
    const [fbName, setFbName] = useState('')
    const [fbEmail, setFbEmail] = useState('')
    const [fbMessage, setFbMessage] = useState('')
    const [fbStatus, setFbStatus] = useState('idle')
    const [fbError, setFbError] = useState('')

    const API = import.meta.env.VITE_API_URL || ''

    const goAnalyze = (label) => {
        ReactGA.event({ category: 'Navigation', action: 'analyze_click', label })
        navigate('/analyze')
    }

    const goLive = (label) => {
        ReactGA.event({ category: 'Navigation', action: 'live_coaching_click', label })
        navigate('/live')
    }

    const handleFeedbackSubmit = async (e) => {
        e.preventDefault()
        if (!fbName.trim() || !fbEmail.trim() || !fbMessage.trim()) return
        setFbStatus('sending')
        setFbError('')
        try {
            await axios.post(`${API}/feedback`, {
                name: fbName.trim(),
                email: fbEmail.trim(),
                message: fbMessage.trim(),
            })
            setFbStatus('sent')
            setFbName('')
            setFbEmail('')
            setFbMessage('')
            ReactGA.event({ category: 'Feedback', action: 'feedback_sent', label: 'landing_page' })
        } catch (err) {
            setFbStatus('error')
            setFbError(err?.response?.data?.detail || 'Something went wrong. Please try again.')
        }
    }

    return (
        <div className="figma-landing min-h-screen w-full overflow-x-hidden" style={{ backgroundColor: PAGE_BG, color: '#000' }}>
            {/* Top accent bar + nav */}
            <header className="figma-top-bar sticky top-0 z-50" style={{ backgroundColor: BRAND }}>
                <div className="mx-auto flex h-14 sm:h-16 max-w-6xl items-center justify-between gap-4 px-5 sm:px-8">
                    <button
                        type="button"
                        onClick={() => goAnalyze('landing_nav_logo')}
                        className="flex min-w-0 items-center gap-2.5 text-[#fafafa]"
                        aria-label="IsoCourt home"
                    >
                        <Logo size={24} className="shrink-0 text-[#fafafa]" />
                        <span className="font-display text-lg font-bold tracking-tight hidden sm:inline">
                            IsoCourt
                        </span>
                    </button>
                    <nav className="flex items-center gap-4 sm:gap-6" aria-label="Primary">
                        <button
                            type="button"
                            onClick={() => goAnalyze('landing_nav')}
                            className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#fafafa]/90 hover:text-white transition-colors"
                        >
                            Analyze
                        </button>
                        <button
                            type="button"
                            onClick={() => goLive('landing_nav')}
                            className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#fafafa]/90 hover:text-white transition-colors"
                        >
                            Live
                        </button>
                    </nav>
                </div>
            </header>

            {/* Hero */}
            <section className="figma-hero relative overflow-hidden">
                <HeroFigmaBackdrop />

                <div className="figma-hero-content mx-auto flex max-w-5xl flex-col items-center px-6 pt-12 pb-10 text-center sm:pt-16 sm:pb-14">
                    <h1 className="figma-display-title max-w-4xl">
                        <span>Meet </span>
                        <span style={{ color: BRAND }}>Birdzo</span>
                        <span>, your </span>
                        <span style={{ color: BRAND }}>second pair of eyes</span>
                        <br />
                        on court.
                    </h1>
                    <p className="mt-6 max-w-xl text-base sm:text-lg text-neutral-600 leading-relaxed font-sans">
                        Upload a rally or single stroke. IsoCourt reads footwork, contact, and shot type — fast, specific, zero fluff.
                    </p>

                    <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
                        <FigmaButton variant="pill" onClick={() => goAnalyze('landing_hero')}>
                            Drop a Clip!
                        </FigmaButton>
                        <FigmaButton variant="square" onClick={() => goLive('landing_hero')}>
                            Start a Live session!
                        </FigmaButton>
                    </div>
                </div>
            </section>

            {/* Video demo */}
            <section className="figma-section px-5 sm:px-8">
                <div className="mx-auto max-w-5xl">
                    <div className="figma-video-frame aspect-video w-full overflow-hidden bg-black">
                        <iframe
                            src="https://www.youtube-nocookie.com/embed/UA3KPoj0j70?rel=0&modestbranding=1"
                            title="IsoCourt demo"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                            className="h-full w-full"
                        />
                    </div>
                </div>
            </section>

            {/* Core features */}
            <section id="features" className="figma-section px-5 sm:px-8">
                <div className="mx-auto max-w-5xl">
                    <h2 className="figma-section-title text-center">CORE FEATURES</h2>
                    <div className="mt-12 grid gap-6 sm:grid-cols-3">
                        {features.map(({ icon, title, description }) => (
                            <article key={title} className="figma-feature-card">
                                <div
                                    className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-lg"
                                    style={{ backgroundColor: `${BRAND}22`, color: BRAND }}
                                >
                                    <Icon name={icon} size={24} />
                                </div>
                                <h3 className="font-display text-lg font-bold uppercase tracking-wide mb-2">{title}</h3>
                                <p className="text-sm text-neutral-600 leading-relaxed font-sans">{description}</p>
                            </article>
                        ))}
                    </div>
                </div>
            </section>

            {/* What to do */}
            <section id="how-it-works" className="figma-section px-5 sm:px-8">
                <div className="mx-auto max-w-5xl">
                    <h2 className="figma-section-title text-center">WHAT TO DO</h2>

                    <div className="mt-16 space-y-20 sm:space-y-28">
                        {flowSteps.map((step, i) => {
                            const isImageLeft = step.imageSide === 'left'
                            return (
                                <div
                                    key={step.label}
                                    className={`figma-flow-step grid items-center gap-8 sm:gap-12 md:grid-cols-2 ${
                                        isImageLeft ? '' : 'md:[&>*:first-child]:order-2'
                                    }`}
                                >
                                    <div className={`flex justify-center ${isImageLeft ? 'md:justify-start' : 'md:justify-end'}`}>
                                        <div className="figma-phone-frame aspect-[9/16] w-[min(100%,180px)] sm:w-[220px]">
                                            <img
                                                src={step.image}
                                                alt=""
                                                className="figma-phone-mockup h-full w-full object-contain"
                                            />
                                        </div>
                                    </div>
                                    <div className={`text-center ${isImageLeft ? 'md:text-left' : 'md:text-right'}`}>
                                        <p className="figma-flow-label">{step.label}</p>
                                        {i === 0 && (
                                            <p className="mt-3 text-sm text-neutral-500 font-sans max-w-sm mx-auto md:mx-0">
                                                One smash, a messy rally, or a drill — steady camera, shuttle in frame.
                                            </p>
                                        )}
                                        {i === 1 && (
                                            <p className="mt-3 text-sm text-neutral-500 font-sans max-w-sm mx-auto md:ml-auto md:mr-0">
                                                Poses, strokes, and scores stitch together while you grab water.
                                            </p>
                                        )}
                                        {i === 2 && (
                                            <p className="mt-3 text-sm text-neutral-500 font-sans max-w-sm mx-auto md:mx-0">
                                                A clear read on what broke down, plus cues for your next session.
                                            </p>
                                        )}
                                    </div>
                                    {i === 0 && (
                                        <div className="hidden md:block col-span-2 relative h-0" aria-hidden>
                                            <ShuttleDecor className="absolute right-[18%] top-[-2rem] w-14 opacity-90" rotate={165} />
                                            <ShuttleDecor className="absolute right-[8%] top-[-4rem] w-12 opacity-80" rotate={140} />
                                        </div>
                                    )}
                                </div>
                            )
                        })}
                    </div>
                </div>
            </section>

            {/* Final CTA */}
            <section className="figma-section figma-final-cta px-5 sm:px-8 pb-24">
                <div className="mx-auto max-w-3xl text-center">
                    <h2 className="figma-section-title">WHAT DO I DO NOW?</h2>
                    <p className="figma-final-sub mt-8 max-w-xl mx-auto">
                        give <span style={{ color: BRAND }}>IsoCourt</span> a go and see what{' '}
                        <span style={{ color: BRAND }}>birdzo</span> has in mind for you
                    </p>
                    <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
                        <FigmaButton variant="pill" onClick={() => goAnalyze('landing_footer')}>
                            Drop a Clip!
                        </FigmaButton>
                        <FigmaButton variant="square" onClick={() => goLive('landing_footer')}>
                            Start a Live session!
                        </FigmaButton>
                    </div>
                </div>
            </section>

            {/* Feedback — not in Figma frame, kept for product needs */}
            <section id="feedback" className="border-t border-neutral-200 px-5 sm:px-8 py-16 bg-white/60">
                <div className="mx-auto max-w-xl">
                    <h2 className="font-display text-2xl sm:text-3xl font-bold text-center uppercase tracking-tight">
                        Court notes welcome
                    </h2>
                    <p className="mt-3 text-center text-sm text-neutral-600 font-sans">
                        Wrong call, wild idea, or “this saved my smash.” We read every message between training blocks.
                    </p>

                    {fbStatus === 'sent' ? (
                        <div className="mt-8 rounded-xl border border-[#6c9c8d]/30 bg-[#6c9c8d]/10 p-8 text-center">
                            <Icon name="check_circle" size={40} className="mx-auto mb-3" style={{ color: BRAND }} />
                            <h3 className="text-lg font-semibold mb-2" style={{ color: BRAND }}>
                                Thanks for your feedback!
                            </h3>
                            <button
                                type="button"
                                onClick={() => setFbStatus('idle')}
                                className="text-xs hover:underline"
                                style={{ color: BRAND }}
                            >
                                Send another message
                            </button>
                        </div>
                    ) : (
                        <form onSubmit={handleFeedbackSubmit} className="mt-8 space-y-4 rounded-xl border border-neutral-200 bg-[#fafafa] p-6">
                            <div className="grid gap-4 sm:grid-cols-2">
                                <div>
                                    <label htmlFor="fb-name" className="text-xs font-medium text-neutral-500 block mb-1.5">
                                        Name
                                    </label>
                                    <input
                                        id="fb-name"
                                        type="text"
                                        value={fbName}
                                        onChange={(e) => setFbName(e.target.value)}
                                        required
                                        className="figma-input w-full"
                                    />
                                </div>
                                <div>
                                    <label htmlFor="fb-email" className="text-xs font-medium text-neutral-500 block mb-1.5">
                                        Email
                                    </label>
                                    <input
                                        id="fb-email"
                                        type="email"
                                        value={fbEmail}
                                        onChange={(e) => setFbEmail(e.target.value)}
                                        required
                                        className="figma-input w-full"
                                    />
                                </div>
                            </div>
                            <div>
                                <label htmlFor="fb-message" className="text-xs font-medium text-neutral-500 block mb-1.5">
                                    Message
                                </label>
                                <textarea
                                    id="fb-message"
                                    value={fbMessage}
                                    onChange={(e) => setFbMessage(e.target.value)}
                                    required
                                    rows={4}
                                    className="figma-input w-full resize-none"
                                />
                            </div>
                            {fbStatus === 'error' && (
                                <p className="text-xs text-red-600">{fbError}</p>
                            )}
                            <button
                                type="submit"
                                disabled={fbStatus === 'sending'}
                                className="figma-cta w-full rounded-bl-[8px] rounded-tr-[8px] disabled:opacity-50"
                                style={{ backgroundColor: BRAND }}
                            >
                                {fbStatus === 'sending' ? 'Sending…' : 'Send feedback'}
                            </button>
                        </form>
                    )}
                </div>
            </section>

            <footer className="border-t border-neutral-200 px-5 py-8 text-center sm:text-left">
                <div className="mx-auto flex max-w-5xl flex-col sm:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <Logo size={20} className="text-brand" />
                        <span className="font-display text-sm font-bold">
                            Iso<span style={{ color: BRAND }}>Court</span>
                        </span>
                    </div>
                    <nav className="flex flex-wrap justify-center gap-x-6 gap-y-2 text-xs text-neutral-500 font-sans" aria-label="Legal">
                        <Link to="/privacy" className="hover:opacity-80" style={{ color: BRAND }}>
                            Privacy
                        </Link>
                        <Link to="/terms" className="hover:opacity-80" style={{ color: BRAND }}>
                            Terms
                        </Link>
                        <a href="#feedback" className="hover:opacity-80" style={{ color: BRAND }}>
                            Contact
                        </a>
                    </nav>
                </div>
            </footer>
        </div>
    )
}
