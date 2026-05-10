import { useState, useEffect, useCallback } from 'react'
import { motion, useReducedMotion, AnimatePresence } from 'framer-motion'
import { useNavigate, Link } from 'react-router-dom'
import axios from 'axios'
import ReactGA from 'react-ga4'
import Logo from './Logo'
import BadmintonNetScene from './BadmintonNetScene'
import { HeroCourtShuttleLayer, ShuttleInline } from './CourtShuttleMotif.jsx'
import HallCourtBand from './HallCourtBand.jsx'
import ThemeToggle from './ThemeToggle.jsx'
import { spineShort, cta, nav as brandNav, landing, flowSteps } from '../brand/isoCourtVoice.js'

function Icon({ name, size = 20, className = '' }) {
    return (
        <span className={`material-symbols-outlined ${className}`} style={{ fontSize: size }}>
            {name}
        </span>
    )
}

const HERO_MICRO_LINES = [
    'Tramlines to baseline: if the shuttle and your swing are on frame, IsoCourt can read it fairly.',
    'Ten stroke families — net brushes, slices, standing smashes — the taxonomy you argue about between ends.',
    'Built between club nights by someone who restrings before opening analytics tabs.',
]

function RotatingMicroLine({ lines }) {
    const shouldReduceMotion = useReducedMotion()
    const [i, setI] = useState(0)
    useEffect(() => {
        if (shouldReduceMotion) return
        const id = setInterval(() => setI((n) => (n + 1) % lines.length), 4200)
        return () => clearInterval(id)
    }, [shouldReduceMotion, lines.length])
    if (shouldReduceMotion) {
        return (
            <p className="mt-6 max-w-md text-sm leading-relaxed text-stone-400">{lines[0]}</p>
        )
    }
    return (
        <div className="mt-6 min-h-[3rem] max-w-md">
            <AnimatePresence mode="wait">
                <motion.p
                    key={i}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.35, ease: 'easeOut' }}
                    className="text-sm leading-relaxed text-stone-400"
                >
                    {lines[i]}
                </motion.p>
            </AnimatePresence>
        </div>
    )
}

function FadeUp({ children, delay = 0, className = '' }) {
    const shouldReduceMotion = useReducedMotion()
    return (
        <motion.div
            className={className}
            initial={{ opacity: 0, y: shouldReduceMotion ? 0 : 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ duration: shouldReduceMotion ? 0 : 0.5, delay, ease: 'easeOut' }}
        >
            {children}
        </motion.div>
    )
}

const features = [
    {
        icon: 'directions_run',
        label: 'Pose tracing',
        description:
            'Split-step late? Contact behind you? Your body frame, frame by frame—so you fix the same miss twice, not twenty times.',
    },
    {
        icon: 'query_stats',
        label: 'Stroke reads',
        description:
            'What you hit, roughly where it went, and how clean it looked—ten real-world strokes, not textbook labels nobody says out loud.',
    },
    {
        icon: 'tips_and_updates',
        label: 'Plain-language cues',
        description:
            'Short notes you can take to the hall: what to try on the next rep—never a wall of generic “keep practising.”',
    },
]

const steps = [
    {
        n: '01',
        icon: 'upload',
        title: 'Send a clip',
        description: 'One smash, a messy rally, or a drill. Steady camera, shuttle in frame.',
    },
    {
        n: '02',
        icon: 'model_training',
        title: 'We trace & score',
        description: 'Poses and strokes stitch together while you grab water. No tagging.',
    },
    {
        n: '03',
        icon: 'emoji_events',
        title: 'Walk back smarter',
        description: 'What broke down, plus a few cues before your next session.',
    },
]

const stats = [
    { value: '10', label: 'Stroke families', hint: 'From serves to rear-court attack' },
    { value: '4', label: 'Court reads', hint: 'Front, mid, rear, net' },
    { value: '1', label: 'Clip per run', hint: 'One rally or one stroke' },
    { value: '0', label: 'Accounts required', hint: 'Feather shuttle brain, not SaaS login' },
]

const FB_MESSAGE_MAX = 8000

export default function LandingPage() {
    const navigate = useNavigate()

    const [fbName, setFbName] = useState('')
    const [fbEmail, setFbEmail] = useState('')
    const [fbMessage, setFbMessage] = useState('')
    const [fbStatus, setFbStatus] = useState('idle')
    const [fbError, setFbError] = useState('')

    const API = import.meta.env.VITE_API_URL || ''

    const resetFeedbackError = useCallback(() => {
        setFbError('')
        setFbStatus((s) => (s === 'error' ? 'idle' : s))
    }, [])

    const handleFeedbackSubmit = async (e) => {
        e.preventDefault()
        const name = fbName.trim()
        const email = fbEmail.trim()
        const message = fbMessage.trim()
        if (!name || !email || !message) return
        if (message.length > FB_MESSAGE_MAX) {
            setFbStatus('error')
            setFbError(`Please shorten your message (max ${FB_MESSAGE_MAX.toLocaleString()} characters).`)
            return
        }
        setFbStatus('sending')
        setFbError('')
        try {
            await axios.post(`${API}/feedback`, { name, email, message })
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

    const navLink =
        'font-mono text-[11px] uppercase tracking-[0.22em] text-foreground-muted hover:text-foreground border-b border-transparent hover:border-brand pb-0.5 transition-colors'

    return (
        <div className="min-h-screen w-screen bg-page text-foreground overflow-x-hidden font-sans antialiased">

            {/* Navigation — flat, no pill CTAs */}
            <header className="sticky top-0 z-50 border-b-2 border-brand/20 bg-page/90 backdrop-blur-sm">
                <div className="mx-auto flex max-w-6xl items-center justify-between gap-6 px-5 py-4">
                    <button
                        type="button"
                        onClick={() => navigate('/analyze')}
                        className="flex min-w-0 items-baseline gap-3 text-left"
                        aria-label={brandNav.clipLabAria}
                    >
                        <Logo size={26} className="shrink-0 text-brand" />
                        <span className="font-display text-xl tracking-tight text-foreground">
                            Iso<span className="text-brand">Court</span>
                        </span>
                    </button>
                    <div className="flex flex-wrap items-center justify-end gap-x-6 gap-y-2 md:gap-x-8">
                        <nav className="flex flex-wrap items-center gap-x-8 gap-y-2" aria-label="Primary">
                            <button
                                type="button"
                                onClick={() => {
                                    ReactGA.event({ category: 'Navigation', action: 'analyze_click', label: 'landing_nav' })
                                    navigate('/analyze')
                                }}
                                className={navLink}
                            >
                                Analyze
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    ReactGA.event({ category: 'Navigation', action: 'live_coaching_click', label: 'landing_nav' })
                                    navigate('/live')
                                }}
                                className={navLink}
                            >
                                Live
                            </button>
                        </nav>
                        <ThemeToggle />
                    </div>
                </div>
            </header>

            {/* Hero — daylight court, editorial type, asymmetry */}
            <section className="relative min-h-[min(90vh,820px)] bg-stone-950 text-stone-100">
                <div className="pointer-events-none absolute inset-0">
                    <BadmintonNetScene className="h-full w-full object-cover opacity-[0.88]" />
                </div>
                <div
                    className="pointer-events-none absolute inset-0 z-[1] bg-[radial-gradient(ellipse_80%_70%_at_50%_38%,rgba(0,0,0,0.78)_0%,transparent_62%)]"
                    aria-hidden
                />
                <HeroCourtShuttleLayer className="z-[2]" />
                <div
                    className="pointer-events-none absolute inset-x-0 bottom-0 z-[3] h-[min(32vh,300px)] bg-gradient-to-t from-court-mat via-court-mat/55 to-transparent"
                    aria-hidden
                />

                <div className="relative z-10 mx-auto grid max-w-6xl gap-14 px-5 pb-24 pt-16 md:gap-20 md:pb-32 md:pt-24 lg:grid-cols-12 lg:items-end">
                    <div className="lg:col-span-7">
                        <p className="flex flex-wrap items-center gap-2 font-mono text-[11px] uppercase tracking-[0.35em] text-orange-200/85">
                            <ShuttleInline className="text-orange-200/90" />
                            <span>{landing.heroKicker}</span>
                        </p>
                        <h1 className="font-display mt-6 text-[2.65rem] font-normal leading-[1.02] tracking-tight sm:text-6xl lg:text-7xl">
                            Read the rally.
                            <br />
                            <span className="italic text-orange-200">Like you&apos;re beside the court tape.</span>
                        </h1>
                        <p className="mt-8 max-w-md text-lg leading-relaxed text-stone-400">{landing.heroLead}</p>
                        <RotatingMicroLine lines={HERO_MICRO_LINES} />

                        <div className="mt-10 flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
                            <button
                                type="button"
                                onClick={() => navigate('/analyze')}
                                className="inline-flex items-center justify-center border-2 border-orange-500 bg-orange-600 px-8 py-4 text-center text-xs font-semibold uppercase tracking-[0.18em] text-white transition-colors hover:bg-orange-500"
                            >
                                {cta.primary}
                            </button>
                            <button
                                type="button"
                                onClick={() => {
                                    ReactGA.event({ category: 'Navigation', action: 'live_coaching_click', label: 'landing_hero' })
                                    navigate('/live')
                                }}
                                className="inline-flex items-center justify-center gap-2 border border-orange-400/40 bg-black/30 px-8 py-4 text-xs font-medium uppercase tracking-[0.14em] text-orange-50 transition-colors hover:border-orange-300/70 hover:bg-black/45"
                            >
                                <Icon name="sensors" size={16} className="text-teal-300" />
                                {cta.secondary}
                            </button>
                        </div>
                        <p className="mt-8 max-w-lg font-mono text-[10px] uppercase tracking-[0.24em] text-stone-500/90">{spineShort}</p>
                    </div>

                    {/* Sample read — sideline ticket, not a SaaS card */}
                    <div className="lg:col-span-5">
                        <FadeUp delay={0.15}>
                            <div className="border-2 border-orange-400/25 bg-black/45 p-6 backdrop-blur-[2px]">
                                <div className="flex items-baseline justify-between gap-3 border-b border-white/10 pb-4">
                                    <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-stone-500">{landing.sampleReadLabel}</span>
                                    <span className="font-mono text-[10px] text-orange-300/90">8 / 10</span>
                                </div>
                                <p className="mt-4 flex flex-wrap gap-2 font-mono text-[10px] uppercase tracking-[0.18em] text-teal-300/85">
                                    <span className="border border-teal-400/35 px-2 py-0.5">FH clear</span>
                                    <span className="border border-teal-400/35 px-2 py-0.5">Rear tramline</span>
                                </p>
                                <p className="font-display mt-5 text-3xl text-orange-200">Advanced</p>
                                <div className="mt-4 h-1 w-full bg-stone-800">
                                    <div className="h-full w-4/5 bg-teal-500/90" />
                                </div>
                                <dl className="mt-6 space-y-3 border-t border-white/10 pt-5 font-mono text-[11px]">
                                    <div className="flex justify-between gap-4 text-stone-500">
                                        <dt>Technique</dt>
                                        <dd className="text-stone-300">Forehand clear</dd>
                                    </div>
                                    <div className="flex justify-between gap-4 text-stone-500">
                                        <dt>Placement</dt>
                                        <dd className="text-stone-300">Deep lift</dd>
                                    </div>
                                    <div className="flex justify-between gap-4 text-stone-500">
                                        <dt>Position</dt>
                                        <dd className="text-stone-300">Rear court</dd>
                                    </div>
                                </dl>
                                <p className="mt-6 border-t border-white/10 pt-5 text-sm leading-relaxed text-stone-500">
                                    <span className="text-orange-400">→</span> Open your non-racket shoulder earlier so contact stays in front.
                                </p>
                            </div>
                        </FadeUp>
                    </div>
                </div>
            </section>

            <HallCourtBand />

            {/* Stats — scoreboard strip (tramline tape) */}
            <section className="relative border-y-2 border-border bg-page-muted/80">
                <div
                    className="pointer-events-none absolute inset-x-0 top-0 h-[3px] border-b border-dashed border-brand/25"
                    aria-hidden
                />
                <p className="mx-auto max-w-2xl px-5 pt-12 text-center font-mono text-[11px] uppercase tracking-[0.22em] text-foreground-muted">
                    {landing.statsIntro}
                </p>
                <div className="mx-auto flex max-w-6xl flex-wrap justify-between gap-y-10 px-5 pb-14 pt-8 md:flex-nowrap md:divide-x md:divide-border">
                    {stats.map(({ value, label, hint }) => (
                        <div key={label} className="min-w-[45%] flex-1 text-center md:min-w-0 md:px-10 md:first:pl-0 md:last:pr-0">
                            <div className="font-display text-4xl tabular-nums text-foreground md:text-5xl">{value}</div>
                            <div className="font-mono mt-3 text-[10px] uppercase tracking-[0.24em] text-foreground-muted">{label}</div>
                            <p className="mx-auto mt-2 max-w-[12rem] font-sans text-[11px] leading-snug normal-case tracking-normal text-foreground-subtle">{hint}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Features — editorial stack, ruled sections */}
            <section id="features" className="relative mx-auto max-w-3xl px-5 py-20 md:py-28">
                <div
                    className="pointer-events-none absolute -left-4 top-24 hidden h-48 w-px bg-gradient-to-b from-brand/50 via-brand/15 to-transparent md:block"
                    aria-hidden
                />
                <div
                    className="pointer-events-none absolute -left-2 top-24 hidden h-48 w-px bg-gradient-to-b from-brand/25 via-transparent to-transparent md:block"
                    aria-hidden
                />
                <p className="sunrise-kicker flex flex-wrap items-center gap-2 text-foreground-muted">
                    <ShuttleInline className="text-brand/70" />
                    {landing.featuresKicker}
                </p>
                <h2 className="font-display mt-5 text-4xl leading-tight text-foreground md:text-5xl">{landing.featuresHeadline}</h2>
                <p className="mt-6 text-lg leading-relaxed text-foreground-muted">{landing.featuresLead}</p>

                <div className="mt-16 space-y-0">
                    {features.map(({ icon, label, description }) => (
                        <FadeUp key={label}>
                            <article className="border-t border-border py-14 first:border-t-0 first:pt-0">
                                <div className="mb-4 inline-flex h-11 w-11 items-center justify-center border border-border bg-page-muted text-brand">
                                    <Icon name={icon} size={22} />
                                </div>
                                <h3 className="font-display text-2xl text-foreground md:text-3xl">{label}</h3>
                                <p className="mt-4 text-[17px] leading-relaxed text-foreground-muted">{description}</p>
                            </article>
                        </FadeUp>
                    ))}
                </div>
            </section>

            {/* Film */}
            <section className="border-t border-border bg-page-muted/50 px-5 py-20 md:py-28">
                <div className="mx-auto max-w-4xl">
                    <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-foreground-muted">{landing.filmKicker}</p>
                    <h2 className="font-display mt-4 text-3xl text-foreground md:text-4xl">{landing.filmHeadline}</h2>
                    <p className="mt-4 max-w-2xl text-foreground-muted">{landing.filmLead}</p>
                    <div className="mt-10 border-2 border-court-mat bg-court-mat/[0.07] p-2 shadow-none ring-1 ring-court-line/25 md:p-3 dark:bg-court-mat/15 dark:ring-white/10">
                        <div className="aspect-video border border-court-mat/40 bg-black">
                            <iframe
                                src="https://www.youtube-nocookie.com/embed/UA3KPoj0j70?rel=0&modestbranding=1&color=white"
                                title="IsoCourt — rally read on tape"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowFullScreen
                                className="h-full w-full"
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* Flow — sideline timeline */}
            <section id="how-it-works" className="mx-auto max-w-2xl px-5 py-20 md:py-28">
                <p className="sunrise-kicker text-foreground-muted">{landing.flowKicker}</p>
                <h2 className="font-display mt-4 text-4xl text-foreground md:text-5xl">{landing.flowHeadline}</h2>

                <div className="relative mt-16 ml-2 border-l-2 border-brand pl-10 md:ml-4">
                    {flowSteps.map(({ n, icon, title, description }) => (
                        <FadeUp key={n}>
                            <div className="relative pb-16 last:pb-0">
                                <span className="absolute -left-[31px] top-1.5 flex h-[14px] w-[14px] border-2 border-brand bg-page" aria-hidden />
                                <p className="font-mono text-xs text-brand">{n}</p>
                                <div className="mt-3 inline-flex h-9 w-9 items-center justify-center border border-border bg-page-muted text-brand">
                                    <Icon name={icon} size={18} />
                                </div>
                                <h3 className="font-display mt-4 text-2xl text-foreground">{title}</h3>
                                <p className="mt-3 leading-relaxed text-foreground-muted">{description}</p>
                            </div>
                        </FadeUp>
                    ))}
                </div>
            </section>

            {/* Closing — flat band, no glow orb */}
            <section className="border-y-2 border-border bg-orange-500/[0.06] px-5 py-24 md:py-32">
                <div className="mx-auto max-w-2xl text-center">
                    <h2 className="font-display text-4xl leading-tight text-foreground md:text-5xl">
                        {landing.closingHeadline}
                        <br />
                        {landing.closingSub}
                    </h2>
                    <p className="mt-8 text-lg text-foreground-muted">{landing.closingLead}</p>
                    <button
                        type="button"
                        onClick={() => navigate('/analyze')}
                        className="mt-12 inline-flex items-center gap-2 border-2 border-brand bg-accent px-10 py-4 text-xs font-semibold uppercase tracking-[0.18em] text-onaccent transition-colors hover:bg-accent-hover"
                    >
                        <Icon name="upload" size={18} />
                        {cta.primary}
                    </button>
                    <p className="mt-10 font-mono text-[11px] uppercase tracking-[0.2em] text-foreground-muted">
                        Want cues between points instead?{' '}
                        <button
                            type="button"
                            onClick={() => {
                                ReactGA.event({ category: 'Navigation', action: 'live_coaching_click', label: 'landing_footer_cta' })
                                navigate('/live')
                            }}
                            className="border-b border-brand text-brand transition-colors hover:text-accent"
                        >
                            {cta.liveFooter}
                        </button>
                    </p>
                </div>
            </section>

            {/* Letters — optimized layout: intro column + sideline form */}
            <section
                id="feedback"
                className="border-t-2 border-border bg-page-muted/35 px-5 py-20 md:py-28"
                aria-labelledby="feedback-heading"
            >
                <div className="mx-auto grid max-w-5xl gap-14 lg:grid-cols-12 lg:gap-16 lg:items-start">
                    <div className="lg:col-span-5">
                        <p className="sunrise-kicker text-foreground-muted">{landing.feedbackKicker}</p>
                        <h2 id="feedback-heading" className="font-display mt-4 text-4xl text-foreground md:text-[2.5rem] md:leading-tight">
                            {landing.feedbackHeadline}
                        </h2>
                        <p className="mt-6 text-lg leading-relaxed text-foreground-muted">{landing.feedbackLead}</p>
                        <ul className="mt-10 space-y-3 border-l-2 border-brand/40 pl-5 text-sm leading-snug text-foreground-muted">
                            <li>
                                <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-brand">Reads</span>
                                <span className="mt-1 block">Stroke labels, live coaching, anything that felt off in a rally.</span>
                            </li>
                            <li className="pt-2">
                                <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-brand">Reply</span>
                                <span className="mt-1 block">No SLA wallpaper—we answer when we&apos;re back from the hall.</span>
                            </li>
                        </ul>
                    </div>

                    <div className="lg:col-span-7">
                        {fbStatus === 'sent' ? (
                            <div
                                className="border-2 border-brand-secondary/50 bg-surface px-8 py-12 text-center shadow-none"
                                role="status"
                                aria-live="polite"
                            >
                                <Icon name="check_circle" size={40} className="mx-auto text-brand-secondary" aria-hidden />
                                <p className="font-display mt-6 text-3xl text-foreground">On the list.</p>
                                <p className="mt-4 max-w-sm mx-auto text-sm leading-relaxed text-foreground-muted">
                                    Thanks—we&apos;ll get back when we can. If it&apos;s urgent, say so in the line above next time.
                                </p>
                                <button
                                    type="button"
                                    onClick={() => setFbStatus('idle')}
                                    className="font-mono mt-10 text-[11px] uppercase tracking-[0.22em] text-brand transition-colors hover:text-accent"
                                >
                                    Send another note
                                </button>
                            </div>
                        ) : (
                            <form
                                onSubmit={handleFeedbackSubmit}
                                className="border border-border border-l-[3px] border-l-brand bg-surface p-6 sm:p-8 shadow-none"
                                aria-busy={fbStatus === 'sending'}
                            >
                                <p id="feedback-form-desc" className="sr-only">
                                    Contact form: your name, email, and message. Required fields are marked.
                                </p>
                                <div className="grid gap-6 sm:grid-cols-2">
                                    <div>
                                        <label htmlFor="fb-name" className="font-mono text-[10px] uppercase tracking-[0.2em] text-foreground-muted">
                                            Name <span className="text-brand">*</span>
                                        </label>
                                        <input
                                            id="fb-name"
                                            name="name"
                                            type="text"
                                            autoComplete="name"
                                            value={fbName}
                                            onChange={(e) => {
                                                resetFeedbackError()
                                                setFbName(e.target.value)
                                            }}
                                            placeholder="How we should address you"
                                            required
                                            className="mt-2 w-full border border-border bg-inset px-4 py-3 text-sm text-foreground placeholder:text-foreground-subtle focus:border-accent focus:outline-none"
                                        />
                                    </div>
                                    <div>
                                        <label htmlFor="fb-email" className="font-mono text-[10px] uppercase tracking-[0.2em] text-foreground-muted">
                                            Email <span className="text-brand">*</span>
                                        </label>
                                        <input
                                            id="fb-email"
                                            name="email"
                                            type="email"
                                            autoComplete="email"
                                            inputMode="email"
                                            value={fbEmail}
                                            onChange={(e) => {
                                                resetFeedbackError()
                                                setFbEmail(e.target.value)
                                            }}
                                            placeholder="you@example.com"
                                            required
                                            className="mt-2 w-full border border-border bg-inset px-4 py-3 text-sm text-foreground placeholder:text-foreground-subtle focus:border-accent focus:outline-none"
                                        />
                                    </div>
                                </div>

                                <div className="mt-6">
                                    <label htmlFor="fb-message" className="font-mono text-[10px] uppercase tracking-[0.2em] text-foreground-muted">
                                        Message <span className="text-brand">*</span>
                                    </label>
                                    <textarea
                                        id="fb-message"
                                        name="message"
                                        value={fbMessage}
                                        onChange={(e) => {
                                            resetFeedbackError()
                                            setFbMessage(e.target.value.slice(0, FB_MESSAGE_MAX))
                                        }}
                                        placeholder="Clears read wrong on clip X, love the coach tips, idea for doubles footwork…"
                                        required
                                        rows={6}
                                        aria-describedby="fb-char-count"
                                        className="mt-2 min-h-[140px] w-full resize-y border border-border bg-inset px-4 py-3 text-sm leading-relaxed text-foreground placeholder:text-foreground-subtle focus:border-accent focus:outline-none"
                                    />
                                    <p id="fb-char-count" className="mt-2 text-right font-mono text-[10px] text-foreground-subtle tabular-nums">
                                        {fbMessage.length.toLocaleString()} / {FB_MESSAGE_MAX.toLocaleString()}
                                    </p>
                                </div>

                                {fbStatus === 'error' && fbError && (
                                    <div
                                        id="fb-error"
                                        role="alert"
                                        aria-live="assertive"
                                        className="mt-6 flex gap-3 border border-rose-500/35 bg-rose-500/[0.08] px-4 py-3 text-sm text-rose-800 dark:text-rose-200"
                                    >
                                        <Icon name="error" size={18} className="mt-0.5 shrink-0" aria-hidden />
                                        <span>{fbError}</span>
                                    </div>
                                )}

                                <button
                                    type="submit"
                                    disabled={fbStatus === 'sending'}
                                    className="mt-8 flex w-full items-center justify-center gap-2 border-2 border-brand bg-accent py-4 text-xs font-semibold uppercase tracking-[0.18em] text-onaccent transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    {fbStatus === 'sending' ? (
                                        <>
                                            <span
                                                className="h-4 w-4 animate-spin rounded-full border-2 border-onaccent/30 border-t-onaccent"
                                                aria-hidden
                                            />
                                            Sending…
                                        </>
                                    ) : (
                                        <>
                                            <Icon name="send" size={16} aria-hidden />
                                            Send note
                                        </>
                                    )}
                                </button>
                            </form>
                        )}
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="border-t-2 border-border bg-page-muted/40 px-5 py-12">
                <div className="mx-auto flex max-w-6xl flex-col gap-10 md:flex-row md:items-start md:justify-between">
                    <div>
                        <div className="flex items-baseline gap-3">
                            <Logo size={20} className="text-brand" />
                            <span className="font-display text-lg tracking-tight">
                                Iso<span className="text-brand">Court</span>
                            </span>
                        </div>
                        <p className="mt-6 max-w-sm text-sm leading-relaxed text-foreground-muted">{landing.footerLead}</p>
                    </div>
                    <nav className="font-mono flex flex-wrap gap-x-10 gap-y-3 text-[11px] uppercase tracking-[0.22em] text-foreground-muted" aria-label="Legal">
                        <Link to="/privacy" className="hover:text-brand">
                            Privacy
                        </Link>
                        <Link to="/terms" className="hover:text-brand">
                            Terms
                        </Link>
                        <a href="#feedback" className="hover:text-brand">
                            {landing.contactLink}
                        </a>
                    </nav>
                </div>
            </footer>
        </div>
    )
}
