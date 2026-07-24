import { useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import ReactGA from 'react-ga4'
import Logo from './Logo'
import HeroFigmaBackdrop from './HeroFigmaBackdrop'
import InteractivePoseFigure from './InteractivePoseFigure'
import LandingResultsPreview from './LandingResultsPreview'
import StrokeTicker from './StrokeTicker'
import ThemeToggle from './ThemeToggle'
import { usePageSeo } from '../seo/usePageSeo'

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
            'What you hit, where it went, and how clean it looked: across ten real-world strokes, not textbook labels.',
    },
    {
        icon: 'tips_and_updates',
        title: 'Coaching notes',
        description:
            'Short, specific cues you can take to the hall: what to try on the next rep, not generic "keep practising."',
    },
]

const processSteps = [
    {
        num: '01',
        title: 'toss us a clip',
        body: 'One smash, a messy rally, a drill. Keep the shuttle in frame and the camera steady.',
    },
    {
        num: '02',
        title: 'we do the tedious bit',
        body: 'Poses, strokes, and scores stitched together while you grab water. No manual tagging.',
    },
    {
        num: '03',
        title: 'walk back on court smarter',
        body: 'Clear read on what broke down, plus a few cues to run before your next session.',
    },
]

const FULL_FLOW_VIDEO = '/demo-videos/full-flow.mp4'

function PhoneCourtDemo({ video, frame, label }) {
    return (
        <div className="figma-phone-frame figma-phone-frame--split">
            <div className="figma-phone-screen">
                <video src={video} autoPlay loop muted playsInline preload="none" aria-label={label} />
            </div>
            <img src={frame} alt="" className="figma-phone-court-frame figma-phone-mockup" aria-hidden />
        </div>
    )
}

function Icon({ name, size, className = '' }) {
    return (
        <span
            className={`material-symbols-outlined ${className}`}
            style={size != null ? { fontSize: size } : undefined}
        >
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

export default function LandingPage() {
    usePageSeo('/')
    const heroRef = useRef(null)
    const [fbName, setFbName] = useState('')
    const [fbEmail, setFbEmail] = useState('')
    const [fbMessage, setFbMessage] = useState('')
    const [fbStatus, setFbStatus] = useState('idle')
    const [fbError, setFbError] = useState('')

    const API = import.meta.env.VITE_API_URL || ''

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
            <header className="figma-top-bar">
                <div className="mx-auto flex h-14 sm:h-16 max-w-6xl items-center justify-between gap-4 px-5 sm:px-8">
                    <Link
                        to="/"
                        className="flex min-w-0 items-center gap-2.5 text-[#fafafa]"
                        aria-label="IsoCourt home"
                    >
                        <Logo size={24} className="shrink-0 text-[#fafafa]" />
                        <span className="font-display text-lg font-bold tracking-tight hidden sm:inline">
                            IsoCourt
                        </span>
                    </Link>
                    <nav className="flex items-center gap-3 sm:gap-5" aria-label="Primary">
                        <Link
                            to="/analyze"
                            onClick={() => ReactGA.event({ category: 'Navigation', action: 'analyze_click', label: 'landing_nav' })}
                            className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#fafafa]/90 hover:text-white transition-colors"
                        >
                            Analyze
                        </Link>
                        <Link
                            to="/live"
                            onClick={() => ReactGA.event({ category: 'Navigation', action: 'live_coaching_click', label: 'landing_nav' })}
                            className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#fafafa]/90 hover:text-white transition-colors"
                        >
                            Live
                        </Link>
                        <ThemeToggle />
                    </nav>
                </div>
            </header>

            <div className="figma-landing figma-page-body theme-page min-h-screen w-full">
                <div className="figma-top-bar-spacer" aria-hidden />

            {/* Hero — single scaled Figma artboard (1512×870) */}
            <section ref={heroRef} className="figma-hero">
                <div className="figma-hero-artboard" aria-hidden>
                    <HeroFigmaBackdrop scrollTarget={heroRef} />
                </div>

                <div className="figma-hero-content">
                    <h1 className="figma-display-title">
                        <span>Meet </span>
                        <span className="figma-brand-accent">Birdzo</span>
                        <span>, your </span>
                        <span className="figma-brand-accent">second pair of eyes</span>
                        <br />
                        on court.
                    </h1>
                    <p className="figma-hero-sub">
                        Upload a rally or single stroke. IsoCourt&apos;s Birdzo coach reads footwork, contact, and shot type. Fast, specific, zero fluff.
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

            <StrokeTicker />

            {/* Results preview + pose */}
            <section id="results-preview" className="figma-split figma-split--results scroll-mt-20">
                <div className="figma-split__panel figma-split__panel--copy">
                    <div className="figma-split__inner">
                        <h2 className="figma-split-title">
                            read the rally. <span className="figma-brand-accent">see the gap.</span>
                        </h2>
                        <LandingResultsPreview />
                    </div>
                </div>
                <div className="figma-split__panel figma-split__panel--visual figma-split__panel--pose">
                    <InteractivePoseFigure />
                </div>
            </section>

            {/* Core features — before process video */}
            <section id="features" className="figma-section px-5 sm:px-8 scroll-mt-20">
                <div className="figma-section-inner">
                    <h2 className="figma-section-title text-center">core features</h2>
                    <div className="figma-feature-grid">
                        {features.map(({ icon, title, description }) => (
                            <article key={title} className="figma-feature-card">
                                <div className="figma-icon-badge rounded-lg">
                                    <Icon name={icon} />
                                </div>
                                <h3 className="figma-feature-title">{title}</h3>
                                <p className="figma-feature-desc">{description}</p>
                            </article>
                        ))}
                    </div>
                </div>
            </section>

            {/* Three steps + full process video */}
            <section id="how-it-works" className="figma-split figma-split--process scroll-mt-20">
                <div className="figma-split__panel figma-split__panel--copy">
                    <div className="figma-split__inner">
                        <h2 className="figma-split-title">
                            three steps. <span className="figma-brand-accent">smarter court.</span>
                        </h2>
                        <ol className="figma-split-steps">
                            {processSteps.map((step) => (
                                <li key={step.num} className="figma-split-step">
                                    <span className="figma-split-step__num">{step.num}</span>
                                    <div>
                                        <h3 className="figma-split-step__title">{step.title}</h3>
                                        <p className="figma-split-step__body">{step.body}</p>
                                    </div>
                                </li>
                            ))}
                        </ol>
                    </div>
                </div>
                <div className="figma-split__panel figma-split__panel--visual figma-split__panel--video">
                    <PhoneCourtDemo
                        video={FULL_FLOW_VIDEO}
                        frame="/phone-mockup.svg"
                        label="Full IsoCourt flow: upload, analyze, and review results"
                    />
                </div>
            </section>

            {/* Final CTA */}
            <section className="figma-section figma-final-cta px-5 sm:px-8 scroll-mt-20">
                <div className="figma-section-inner figma-section-inner--narrow text-center">
                    <h2 className="figma-section-title">what do i do now?</h2>
                    <p className="figma-final-sub mt-8">
                        give <span className="figma-brand-accent">IsoCourt</span> a go and see what{' '}
                        <span className="figma-brand-accent">birdzo</span> has in mind for you
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
                <div className="figma-section-inner figma-section-inner--tight">
                    <h2 className="figma-section-title text-center">court notes welcome</h2>
                    <p className="figma-section-lead mt-4 text-center">
                        Wrong call, wild idea, or “this saved my smash.” We read every message between training blocks.
                    </p>

                    {fbStatus === 'sent' ? (
                        <div className="mt-8 rounded-xl border border-brand/30 bg-brand/10 p-8 text-center">
                            <Icon name="check_circle" size={40} className="mx-auto mb-3 figma-brand-accent" />
                            <h3 className="text-lg font-semibold mb-2 figma-brand-accent">
                                Thanks for your feedback!
                            </h3>
                            <button
                                type="button"
                                onClick={() => setFbStatus('idle')}
                                className="text-xs hover:underline figma-brand-accent"
                            >
                                Send another message
                            </button>
                        </div>
                    ) : (
                        <form onSubmit={handleFeedbackSubmit} className="figma-feedback-form mt-8 space-y-4">
                            <div className="grid gap-4 sm:grid-cols-2">
                                <div>
                                    <label htmlFor="fb-name" className="text-xs font-medium text-[var(--text-subtle)] block mb-1.5">
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
                                    <label htmlFor="fb-email" className="text-xs font-medium text-[var(--text-subtle)] block mb-1.5">
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
                                <label htmlFor="fb-message" className="text-xs font-medium text-[var(--text-subtle)] block mb-1.5">
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
                <div className="figma-section-inner flex flex-col sm:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <Logo size={20} className="text-brand" />
                        <span className="font-display text-sm font-bold">
                            Iso<span className="figma-brand-accent">Court</span>
                        </span>
                    </div>
                    <nav className="figma-footer-nav flex flex-wrap justify-center gap-x-6 gap-y-2" aria-label="Learn and legal">
                        <Link to="/faq" className="figma-footer-link">
                            FAQ
                        </Link>
                        <Link to="/glossary" className="figma-footer-link">
                            Glossary
                        </Link>
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
