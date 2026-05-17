import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import axios from 'axios'
import ReactGA from 'react-ga4'
import Logo from './Logo'
import HeroRacketShaft from './HeroRacketShaft'

function Icon({ name, size = 20, className = '' }) {
    return (
        <span
            className={`material-symbols-outlined ${className}`}
            style={{ fontSize: size }}
        >
            {name}
        </span>
    )
}

const HERO_MICRO_LINES = [
    'Sideline or behind the baseline: if we can see the swing, we can read it.',
    'Ten stroke families, from net brushes to full rear-court power.',
    'Built between sessions by someone who burns through grips, not slide decks.',
]

function RotatingMicroLine({ lines }) {
    const [i, setI] = useState(0)
    useEffect(() => {
        const id = setInterval(() => setI((n) => (n + 1) % lines.length), 4200)
        return () => clearInterval(id)
    }, [lines.length])
    return (
        <p className="mt-3 min-h-[2.75rem] max-w-lg mx-auto text-sm text-emerald-500/90 text-center font-medium leading-snug px-2">
            {lines[i]}
        </p>
    )
}

function FadeUp({ children, className = '' }) {
    return <div className={className}>{children}</div>
}

const features = [
    {
        icon: 'directions_run',
        label: 'Pose tracing',
        description:
            'Split-step late? Contact behind you? See your body frame by frame so you can fix the same miss twice, not twenty times.',
    },
    {
        icon: 'query_stats',
        label: 'Stroke reads',
        description:
            'What you hit, roughly where it went, and how clean it looked, across ten real-world strokes, not textbook labels nobody says out loud.',
    },
    {
        icon: 'tips_and_updates',
        label: 'Coaching notes',
        description:
            'Short, specific cues you can take to the hall: what to try on the next rep, not a wall of generic “keep practising.”',
    },
]

const steps = [
    {
        n: '01',
        icon: 'upload',
        title: 'Toss us a clip',
        description: 'One hard smash, a messy rally, or a drill. Keep the camera steady and the shuttle in frame.',
    },
    {
        n: '02',
        icon: 'model_training',
        title: 'We do the tedious bit',
        description: 'Poses, strokes, and scores get stitched together while you grab water. No manual tagging.',
    },
    {
        n: '03',
        icon: 'emoji_events',
        title: 'Walk back on court smarter',
        description: 'A clear read on what broke down, plus a few cues to try before your next session.',
    },
]

const stats = [
    { value: '10', label: 'Strokes in the book', icon: 'sports' },
    { value: '4', label: 'Court reads', icon: 'explore' },
    { value: '1', label: 'Clip per analysis', icon: 'movie' },
    { value: 'Free', label: 'No signup to start', icon: 'lock_open' },
]

const navCtaClass =
    'text-sm font-medium px-3.5 sm:px-4 py-2 rounded-lg border border-emerald-600/45 bg-emerald-950/30 text-emerald-100 hover:bg-emerald-950/45 hover:border-emerald-500/65 transition-colors flex items-center justify-center gap-2'

export default function LandingPage() {
    const navigate = useNavigate()

    // Feedback form state
    const [fbName, setFbName] = useState('')
    const [fbEmail, setFbEmail] = useState('')
    const [fbMessage, setFbMessage] = useState('')
    const [fbStatus, setFbStatus] = useState('idle') // idle | sending | sent | error
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
        <div className="min-h-screen w-screen bg-neutral-950 text-neutral-100 overflow-x-hidden">

            {/* Navbar */}
            <nav className="sticky top-0 z-50 border-b border-neutral-800/60 bg-neutral-950/85 backdrop-blur-md">
                <div className="max-w-6xl mx-auto px-4 sm:px-6 min-h-14 py-2 flex items-center justify-between gap-3">
                    <button
                        type="button"
                        onClick={() => navigate('/analyze')}
                        className="flex items-center gap-2 shrink-0 min-w-0 rounded-md"
                        aria-label="IsoCourt home"
                    >
                        <Logo size={22} className="text-emerald-500 shrink-0" />
                        <span className="text-base font-semibold tracking-tight hidden sm:inline truncate">
                            Iso<span className="text-emerald-500">Court</span>
                        </span>
                    </button>
                    <div className="flex items-center justify-end gap-2 sm:gap-3 shrink-0">
                        <button
                            type="button"
                            onClick={() => {
                                ReactGA.event({ category: 'Navigation', action: 'analyze_click', label: 'landing_nav' })
                                navigate('/analyze')
                            }}
                            className={navCtaClass}
                        >
                            Drop a clip
                        </button>
                        <button
                            type="button"
                            onClick={() => {
                                ReactGA.event({ category: 'Navigation', action: 'live_coaching_click', label: 'landing_nav' })
                                navigate('/live')
                            }}
                            className={navCtaClass}
                        >
                            <Icon name="sensors" size={18} className="text-emerald-400" />
                            <span className="hidden sm:inline">Live coaching</span>
                            <span className="sm:hidden">Live</span>
                        </button>
                    </div>
                </div>
            </nav>

            <section className="hero-with-racket relative pt-16 md:pt-24 px-6 text-center bg-neutral-950">
                <div className="relative z-10 max-w-3xl mx-auto">
                    <FadeUp delay={0}>
                        <div className="inline-flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium px-3 py-1.5 rounded-full mb-6">
                            <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
                            Made for people who actually play
                        </div>
                    </FadeUp>

                    <FadeUp delay={0.08}>
                        <h1 className="text-5xl md:text-6xl font-bold tracking-tight leading-[1.1] mb-6">
                            Your{' '}
                            <span className="text-emerald-400">AI second pair of eyes</span>
                            <br />
                            on court.
                        </h1>
                    </FadeUp>

                    <FadeUp delay={0.16}>
                        <p className="text-lg text-neutral-400 max-w-xl mx-auto mb-2 leading-relaxed">
                            Upload a video clip (rally or single stroke). Our AI reads footwork, contact, and shot type like a club coach who had coffee first: fast,
                            specific, zero corporate fluff.
                        </p>
                        <RotatingMicroLine lines={HERO_MICRO_LINES} />
                    </FadeUp>

                    <FadeUp delay={0.22} className="mt-6">
                        <div className="hero-cta-row">
                            <div className="hero-shaft-layer" aria-hidden>
                                <HeroRacketShaft className="hero-shaft-svg text-white" />
                            </div>
                            <div className="relative z-10 flex flex-col sm:flex-row items-center justify-center gap-3 flex-wrap">
                                <button
                                type="button"
                                onClick={() => navigate('/analyze')}
                                className="w-full sm:w-auto px-8 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg text-sm transition-colors shadow-lg shadow-emerald-900/30"
                            >
                                Drop a clip, it’s free
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    ReactGA.event({ category: 'Navigation', action: 'live_coaching_click', label: 'landing_hero' })
                                    navigate('/live')
                                }}
                                className="w-full sm:w-auto px-8 py-3 border border-emerald-600/40 hover:border-emerald-500/70 bg-emerald-950/20 text-emerald-100 hover:text-white font-medium rounded-lg text-sm transition-colors flex items-center justify-center gap-2"
                            >
                                <Icon name="sensors" size={18} className="text-emerald-400" />
                                Live while you play
                            </button>
                            </div>
                        </div>
                    </FadeUp>
                </div>
            </section>

            {/* Stats Bar */}
            <section className="hero-stats-bar border-y border-neutral-800 bg-neutral-900/40">
                <div className="max-w-5xl mx-auto px-6 py-10 grid grid-cols-2 md:grid-cols-4 gap-8">
                    {stats.map(({ value, label, icon }, i) => (
                        <FadeUp key={label} delay={i * 0.07} className="text-center">
                            <Icon name={icon} size={22} className="text-emerald-500 mb-2" />
                            <div className="text-3xl font-bold text-white tracking-tight">{value}</div>
                            <div className="text-sm text-neutral-500 mt-0.5">{label}</div>
                        </FadeUp>
                    ))}
                </div>
            </section>

            {/* Features */}
            <section id="features" className="relative py-16 md:py-28 px-6 overflow-visible">
                <div className="max-w-5xl mx-auto relative z-10">
                    <FadeUp className="text-center mb-16">
                        <span className="text-xs font-semibold text-emerald-500 uppercase tracking-widest">Features</span>
                        <h2 className="text-3xl md:text-4xl font-bold mt-3 tracking-tight">
                            The stuff coaches nag you about,<br />
                            without the side-eye
                        </h2>
                        <p className="text-neutral-400 mt-4 max-w-lg mx-auto leading-relaxed">
                            Stroke calls and plain-language cues. The same details you’d get from a good training block, compressed into the minutes between rallies.
                        </p>
                    </FadeUp>

                    <div className="mt-10 md:mt-12 grid md:grid-cols-3 gap-5">
                        {features.map(({ icon, label, description }, i) => (
                            <FadeUp key={label} delay={i * 0.1}>
                                <div className="h-full bg-neutral-900 border border-neutral-800 hover:border-emerald-500/25 hover:-translate-y-0.5 rounded-xl p-6 transition-all duration-300">
                                    <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 mb-4">
                                        <Icon name={icon} size={22} className="text-emerald-400" />
                                    </div>
                                    <h3 className="text-base font-semibold text-white mb-2">{label}</h3>
                                    <p className="text-sm text-neutral-400 leading-relaxed">{description}</p>
                                </div>
                            </FadeUp>
                        ))}
                    </div>
                </div>
            </section>

            {/* Demo Video */}
            <section className="py-16 md:py-20 px-6">
                <div className="max-w-4xl mx-auto">
                    <FadeUp className="text-center mb-8">
                        <p className="text-xs text-neutral-500 uppercase tracking-widest font-semibold flex items-center justify-center gap-2">
                            <Icon name="play_circle" size={14} className="text-emerald-500" />
                            Real footage, real breakdown
                        </p>
                    </FadeUp>
                    <FadeUp delay={0.1}>
                        <div className="relative rounded-xl overflow-hidden border border-neutral-800 shadow-2xl shadow-black/60">
                            <div className="aspect-video">
                                <iframe
                                    src="https://www.youtube-nocookie.com/embed/UA3KPoj0j70?rel=0&modestbranding=1&color=white"
                                    title="IsoCourt Demo"
                                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                    allowFullScreen
                                    className="w-full h-full"
                                />
                            </div>
                        </div>
                    </FadeUp>
                </div>
            </section>

            {/* How It Works */}
            <section id="how-it-works" className="py-16 md:py-28 px-6 bg-neutral-900/30">
                <div className="max-w-4xl mx-auto">
                    <FadeUp className="text-center mb-16">
                        <span className="text-xs font-semibold text-emerald-500 uppercase tracking-widest">How It Works</span>
                        <h2 className="text-3xl md:text-4xl font-bold mt-3 tracking-tight">
                            Clip in. Notes out.<br />
                            Three steps.
                        </h2>
                    </FadeUp>

                    <div className="grid md:grid-cols-3 gap-8">
                        {steps.map(({ n, icon, title, description }, i) => (
                            <FadeUp key={n} delay={i * 0.12}>
                                <div className="relative flex flex-col items-start">
                                    <div className="flex items-center gap-3 mb-4">
                                        <span className="text-4xl font-black text-neutral-600 leading-none select-none">{n}</span>
                                        <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
                                            <Icon name={icon} size={20} className="text-emerald-400" />
                                        </div>
                                    </div>
                                    <h3 className="text-base font-semibold text-white mb-2">{title}</h3>
                                    <p className="text-sm text-neutral-400 leading-relaxed">{description}</p>
                                </div>
                            </FadeUp>
                        ))}
                    </div>
                </div>
            </section>

            {/* Final CTA */}
            <section className="relative py-16 md:py-28 px-6 overflow-hidden">
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                    <div className="w-[600px] h-[400px] bg-emerald-700/10 rounded-full blur-[100px]" />
                </div>
                <FadeUp className="relative max-w-2xl mx-auto text-center">
                    <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-4 leading-[1.1]">
                        Same session tomorrow.
                        <br />
                        Fewer blind spots.
                    </h2>
                    <p className="text-neutral-400 mb-10 text-lg">
                        No signup circus. One clip is enough to see how it reads your game.
                    </p>
                    <button
                        type="button"
                        onClick={() => navigate('/analyze')}
                        className="inline-flex items-center gap-2 px-8 py-3.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg text-sm transition-colors shadow-lg shadow-emerald-900/30"
                    >
                        <Icon name="upload" size={18} />
                        Drop a clip
                    </button>
                    <p className="mt-8 text-sm text-neutral-500">
                        Prefer feedback between points?{' '}
                        <button
                            type="button"
                            onClick={() => {
                                ReactGA.event({ category: 'Navigation', action: 'live_coaching_click', label: 'landing_footer_cta' })
                                navigate('/live')
                            }}
                            className="text-emerald-400 hover:text-emerald-300 font-medium underline-offset-2 hover:underline"
                        >
                            Open live coaching
                        </button>
                    </p>
                </FadeUp>
            </section>

            {/* Feedback / Contact */}
            <section id="feedback" className="py-16 md:py-24 px-6 bg-neutral-900/30">
                <div className="max-w-xl mx-auto">
                    <FadeUp className="text-center mb-10">
                        <span className="text-xs font-semibold text-emerald-500 uppercase tracking-widest">Feedback</span>
                        <h2 className="text-3xl md:text-4xl font-bold mt-3 tracking-tight">
                            Court notes welcome
                        </h2>
                        <p className="text-neutral-400 mt-4 max-w-md mx-auto leading-relaxed">
                            Wrong call, wild idea, or “this saved my smash.” We read every message between training blocks.
                        </p>
                    </FadeUp>

                    <FadeUp delay={0.1}>
                        {fbStatus === 'sent' ? (
                            <div
                                className="bg-emerald-950/40 border border-emerald-800/50 rounded-xl p-8 text-center"
                            >
                                <Icon name="check_circle" size={40} className="text-emerald-400 mx-auto mb-3" />
                                <h3 className="text-lg font-semibold text-emerald-300 mb-2">Thanks for your feedback!</h3>
                                <p className="text-sm text-neutral-400 mb-5">We've received your message and will get back to you soon.</p>
                                <button
                                    onClick={() => setFbStatus('idle')}
                                    className="text-xs text-emerald-500 hover:text-emerald-400 transition-colors"
                                >
                                    Send another message
                                </button>
                            </div>
                        ) : (
                            <form onSubmit={handleFeedbackSubmit} className="bg-neutral-900 border border-neutral-800 rounded-xl p-6 space-y-4">
                                <div className="grid sm:grid-cols-2 gap-4">
                                    <div>
                                        <label htmlFor="fb-name" className="text-xs text-neutral-500 font-medium block mb-1.5">Name</label>
                                        <input
                                            id="fb-name"
                                            type="text"
                                            value={fbName}
                                            onChange={(e) => setFbName(e.target.value)}
                                            placeholder="Your name"
                                            required
                                            className="w-full bg-neutral-950 border border-neutral-800 focus:border-emerald-600 text-neutral-100 placeholder-neutral-600 text-sm rounded-lg px-3.5 py-2.5 transition-colors"
                                        />
                                    </div>
                                    <div>
                                        <label htmlFor="fb-email" className="text-xs text-neutral-500 font-medium block mb-1.5">Email</label>
                                        <input
                                            id="fb-email"
                                            type="email"
                                            value={fbEmail}
                                            onChange={(e) => setFbEmail(e.target.value)}
                                            placeholder="you@example.com"
                                            required
                                            className="w-full bg-neutral-950 border border-neutral-800 focus:border-emerald-600 text-neutral-100 placeholder-neutral-600 text-sm rounded-lg px-3.5 py-2.5 transition-colors"
                                        />
                                    </div>
                                </div>
                                <div>
                                    <label htmlFor="fb-message" className="text-xs text-neutral-500 font-medium block mb-1.5">Message</label>
                                    <textarea
                                        id="fb-message"
                                        value={fbMessage}
                                        onChange={(e) => setFbMessage(e.target.value)}
                                        placeholder="e.g. love the clears read, wish it saw doubles rotation…"
                                        required
                                        rows={4}
                                        className="w-full bg-neutral-950 border border-neutral-800 focus:border-emerald-600 text-neutral-100 placeholder-neutral-600 text-sm rounded-lg px-3.5 py-2.5 transition-colors resize-none"
                                    />
                                </div>

                                {fbStatus === 'error' && (
                                    <div className="flex items-center gap-2 text-rose-400 text-xs bg-rose-500/10 border border-rose-500/20 rounded-lg px-3 py-2">
                                        <Icon name="error" size={14} />
                                        {fbError}
                                    </div>
                                )}

                                <button
                                    type="submit"
                                    disabled={fbStatus === 'sending'}
                                    className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-lg transition-colors flex items-center justify-center gap-2"
                                >
                                    {fbStatus === 'sending' ? (
                                        <>
                                            <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full" />
                                            Sending...
                                        </>
                                    ) : (
                                        <>
                                            <Icon name="send" size={16} />
                                            Send Feedback
                                        </>
                                    )}
                                </button>
                            </form>
                        )}
                    </FadeUp>
                </div>
            </section>

            {/* Footer */}
            <footer className="border-t border-neutral-800 py-8 px-6">
                <div className="max-w-5xl mx-auto flex flex-col gap-5">
                    <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
                        <div className="flex items-center gap-2">
                            <Logo size={18} className="text-emerald-500" />
                            <span className="text-sm font-semibold">
                                Iso<span className="text-emerald-500">Court</span>
                            </span>
                        </div>
                        <p className="text-xs text-neutral-600 text-center sm:text-right max-w-md sm:max-w-none leading-relaxed">
                            Stroke analysis for people who know what a feather costs · Shipped by someone who still chases shuttles off-court
                        </p>
                    </div>
                    <nav className="flex flex-wrap items-center justify-center sm:justify-end gap-x-6 gap-y-2 text-xs text-neutral-500" aria-label="Legal">
                        <Link to="/privacy" className="hover:text-emerald-400 transition-colors">
                            Privacy
                        </Link>
                        <Link to="/terms" className="hover:text-emerald-400 transition-colors">
                            Terms
                        </Link>
                        <a href="#feedback" className="hover:text-emerald-400 transition-colors">
                            Contact
                        </a>
                    </nav>
                </div>
            </footer>
        </div>
    )
}
