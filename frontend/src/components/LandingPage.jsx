import { useRef, useState } from 'react'
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
        video: '/demo-videos/01-upload.mp4',
        frame: '/phone-mockup.svg',
        imageSide: 'left',
    },
    {
        label: 'let birdzo do its thing',
        video: '/demo-videos/02-analyzing.mp4',
        frame: '/phone-mockup-alt.svg',
        imageSide: 'right',
    },
    {
        label: 'review the rally',
        video: '/demo-videos/03-results.mp4',
        frame: '/phone-mockup.svg',
        imageSide: 'left',
    },
]

function PhoneCourtDemo({ video, frame, label }) {
    return (
        <div className="figma-phone-frame aspect-[9/16] w-[min(100%,180px)] sm:w-[220px]">
            <div className="figma-phone-screen">
                <video src={video} autoPlay loop muted playsInline aria-label={label} />
            </div>
            <img src={frame} alt="" className="figma-phone-court-frame figma-phone-mockup" aria-hidden />
        </div>
    )
}

function Icon({ name, size = 20, className = '' }) {
    return (
        <span className={`material-symbols-outlined ${className}`} style={{ fontSize: size }}>
            {name}
        </span>
    )
}

function FigmaButton({
    children,
    variant = 'primary',
    href,
    onClick,
    className = '',
    disabled = false,
    loading = false,
    type = 'button',
}) {
    const classes = [
        'figma-cta',
        variant === 'primary' ? 'figma-cta--primary' : 'figma-cta--secondary',
        loading ? 'figma-cta--loading' : '',
        className,
    ]
        .filter(Boolean)
        .join(' ')

    const content = loading ? <span className="figma-cta-spinner" aria-hidden /> : children

    if (href && !disabled && !loading) {
        return (
            <Link to={href} className={classes} onClick={onClick}>
                {content}
            </Link>
        )
    }

    return (
        <button type={type} onClick={onClick} disabled={disabled || loading} className={classes}>
            {content}
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
    const heroRef = useRef(null)
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
        <>
            <header className="figma-top-bar" style={{ backgroundColor: BRAND }}>
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

            <div
                className="figma-landing figma-page-body min-h-screen w-full"
                style={{ backgroundColor: PAGE_BG, color: '#000' }}
            >
                <div className="figma-top-bar-spacer" aria-hidden />

            {/* Hero — single scaled Figma artboard (1512×870) */}
            <section ref={heroRef} className="figma-hero">
                <div className="figma-hero-artboard" aria-hidden>
                    <HeroFigmaBackdrop scrollTarget={heroRef} />
                </div>

                <div className="figma-hero-content">
                    <h1 className="figma-display-title">
                        <span>Meet </span>
                        <span style={{ color: BRAND }}>Birdzo</span>
                        <span>, your </span>
                        <span style={{ color: BRAND }}>second pair of eyes</span>
                        <br />
                        on court.
                    </h1>
                    <p className="figma-hero-sub">
                        Upload a rally or single stroke. IsoCourt reads footwork, contact, and shot type — fast, specific, zero fluff.
                    </p>

                    <div className="figma-hero-ctas">
                        <FigmaButton
                            variant="primary"
                            href="/analyze"
                            onClick={() => ReactGA.event({ category: 'Navigation', action: 'analyze_click', label: 'landing_hero' })}
                        >
                            Drop a clip
                        </FigmaButton>
                        <FigmaButton
                            variant="secondary"
                            href="/live"
                            onClick={() => ReactGA.event({ category: 'Navigation', action: 'live_coaching_click', label: 'landing_hero' })}
                        >
                            Go live
                        </FigmaButton>
                    </div>
                </div>
            </section>

            {/* Video demo */}
            <section className="figma-section figma-video-section px-5 sm:px-8 scroll-mt-20">
                <div className="mx-auto max-w-5xl">
                    <h2 className="figma-section-title text-center">see it in action</h2>
                    <p className="figma-section-lead mt-4 text-center">
                        A real rally breakdown — footwork, contact, and stroke reads in under a minute.
                    </p>
                    <div className="figma-video-frame mt-10 aspect-video w-full overflow-hidden bg-black">
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
            <section id="features" className="figma-section px-5 sm:px-8 scroll-mt-20">
                <div className="mx-auto max-w-5xl">
                    <h2 className="figma-section-title text-center">core features</h2>
                    <div className="mt-12 grid gap-6 sm:grid-cols-3">
                        {features.map(({ icon, title, description }) => (
                            <article key={title} className="figma-feature-card">
                                <div
                                    className="mb-4 inline-flex h-11 w-11 items-center justify-center rounded-lg"
                                    style={{ backgroundColor: `${BRAND}22`, color: BRAND }}
                                >
                                    <Icon name={icon} size={24} />
                                </div>
                                <h3 className="figma-feature-title">{title}</h3>
                                <p className="figma-feature-desc">{description}</p>
                            </article>
                        ))}
                    </div>
                </div>
            </section>

            {/* What to do */}
            <section id="how-it-works" className="figma-section px-5 sm:px-8 scroll-mt-20">
                <div className="mx-auto max-w-5xl">
                    <h2 className="figma-section-title text-center">what to do</h2>

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
                                    <div className="figma-flow-step__visual flex justify-center items-center">
                                        <PhoneCourtDemo video={step.video} frame={step.frame} label={step.label} />
                                    </div>
                                    <div className="figma-flow-step__copy flex flex-col justify-center items-center text-center">
                                        <p className="figma-flow-label">{step.label}</p>
                                        {i === 0 && (
                                            <p className="mt-3 text-sm text-neutral-500 font-sans max-w-sm">
                                                One smash, a messy rally, or a drill — steady camera, shuttle in frame.
                                            </p>
                                        )}
                                        {i === 1 && (
                                            <p className="mt-3 text-sm text-neutral-500 font-sans max-w-sm">
                                                Poses, strokes, and scores stitch together while you grab water.
                                            </p>
                                        )}
                                        {i === 2 && (
                                            <p className="mt-3 text-sm text-neutral-500 font-sans max-w-sm">
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
                    <h2 className="figma-section-title">what do i do now?</h2>
                    <p className="figma-final-sub mt-8 max-w-xl mx-auto">
                        give <span style={{ color: BRAND }}>IsoCourt</span> a go and see what{' '}
                        <span style={{ color: BRAND }}>birdzo</span> has in mind for you
                    </p>
                    <div className="figma-hero-ctas mt-10">
                        <FigmaButton
                            variant="primary"
                            href="/analyze"
                            onClick={() => ReactGA.event({ category: 'Navigation', action: 'analyze_click', label: 'landing_footer' })}
                        >
                            Drop a clip
                        </FigmaButton>
                        <FigmaButton
                            variant="secondary"
                            href="/live"
                            onClick={() => ReactGA.event({ category: 'Navigation', action: 'live_coaching_click', label: 'landing_footer' })}
                        >
                            Go live
                        </FigmaButton>
                    </div>
                </div>
            </section>

            {/* Feedback — not in Figma frame, kept for product needs */}
            <section id="feedback" className="figma-section figma-feedback px-5 sm:px-8 scroll-mt-20">
                <div className="mx-auto max-w-xl">
                    <h2 className="figma-section-title text-center">court notes welcome</h2>
                    <p className="figma-section-lead mt-4 text-center">
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
                        <form onSubmit={handleFeedbackSubmit} className="figma-feedback-form mt-8 space-y-4">
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
                            <FigmaButton
                                type="submit"
                                variant="primary"
                                className="figma-cta--block"
                                disabled={fbStatus === 'sending'}
                                loading={fbStatus === 'sending'}
                            >
                                Send feedback
                            </FigmaButton>
                        </form>
                    )}
                </div>
            </section>

            <footer className="figma-footer px-5 py-8 text-center sm:text-left">
                <div className="mx-auto flex max-w-5xl flex-col sm:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <Logo size={20} className="text-brand" />
                        <span className="font-display text-sm font-bold">
                            Iso<span style={{ color: BRAND }}>Court</span>
                        </span>
                    </div>
                    <nav className="figma-footer-nav flex flex-wrap justify-center gap-x-6 gap-y-2" aria-label="Legal">
                        <Link to="/privacy" className="figma-footer-link">
                            Privacy
                        </Link>
                        <Link to="/terms" className="figma-footer-link">
                            Terms
                        </Link>
                    </nav>
                </div>
            </footer>
            </div>
        </>
    )
}
