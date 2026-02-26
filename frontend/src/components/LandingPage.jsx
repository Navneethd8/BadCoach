import { motion, useReducedMotion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import Logo from './Logo'

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

function FadeUp({ children, delay = 0, className = '' }) {
    const shouldReduceMotion = useReducedMotion()
    return (
        <motion.div
            className={className}
            initial={{ opacity: 0, y: shouldReduceMotion ? 0 : 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: shouldReduceMotion ? 0 : 0.55, delay, ease: 'easeOut' }}
        >
            {children}
        </motion.div>
    )
}

const features = [
    {
        icon: 'directions_run',
        label: 'Pose Tracing',
        description: 'See exactly how your body moves on every shot. We track your arms, legs, and footwork frame by frame so you can quickly spot and fix breakdowns.',
    },
    {
        icon: 'query_stats',
        label: 'Stroke Analysis',
        description: 'Detects the shot you played, where it landed, and how well you executed it across 10 common stroke types.',
    },
    {
        icon: 'tips_and_updates',
        label: 'AI Coaching',
        description: 'Get clear, practical tips based on what the AI sees, tailored to your level with specific actions to improve your next session.',
    },
]

const steps = [
    {
        n: '01',
        icon: 'upload',
        title: 'Upload Your Clip',
        description: 'Upload a badminton video showing a single stroke or a short rally.',
    },
    {
        n: '02',
        icon: 'model_training',
        title: 'AI Analyzes Your Play',
        description: 'IsoCourt tracks your movement, identifies each stroke, and scores your technique quickly.',
    },
    {
        n: '03',
        icon: 'emoji_events',
        title: 'Get Actionable Feedback',
        description: 'Receive a performance score, a shot by shot breakdown, and clear coaching tips you can apply in your next session.',
    },
]

const stats = [
    { value: '10', label: 'Stroke types', icon: 'sports' },
    { value: '4', label: 'Tactical metrics', icon: 'explore' },
    { value: 'Fast', label: 'Analysis', icon: 'timer' },
    { value: 'Free', label: 'To try', icon: 'card_giftcard' },
]

export default function LandingPage() {
    const navigate = useNavigate()
    return (
        <div className="min-h-screen w-screen bg-neutral-950 text-neutral-100 overflow-x-hidden">

            {/* Navbar */}
            <nav className="sticky top-0 z-50 border-b border-neutral-800/60 bg-neutral-950/80 backdrop-blur-md">
                <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
                    <button onClick={() => navigate('/analyze')} className="flex items-center gap-2 focus:outline-none" aria-label="IsoCourt home">
                        <Logo size={22} className="text-emerald-500" />
                        <span className="text-base font-semibold tracking-tight">
                            Iso<span className="text-emerald-500">Court</span>
                        </span>
                    </button>
                    <motion.button
                        onClick={() => navigate('/analyze')}
                        whileHover={{ scale: 1.04 }}
                        whileTap={{ scale: 0.97 }}
                        className="text-sm font-medium bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-1.5 rounded-md transition-colors"
                    >
                        Try it free
                    </motion.button>
                </div>
            </nav>

            {/* Hero */}
            <section className="relative pt-16 pb-20 md:pt-24 md:pb-32 px-6 text-center overflow-hidden">
                <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                    <div className="w-[700px] h-[500px] bg-emerald-600/8 rounded-full blur-[120px]" />
                </div>

                <div className="relative max-w-3xl mx-auto">
                    <FadeUp delay={0}>
                        <div className="inline-flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium px-3 py-1.5 rounded-full mb-6">
                            <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                            AI-powered badminton coaching
                        </div>
                    </FadeUp>

                    <FadeUp delay={0.08}>
                        <h1 className="text-5xl md:text-6xl font-bold tracking-tight leading-[1.1] mb-6">
                            Your{' '}
                            <span className="text-emerald-400">AI Coach.</span>
                            <br />
                            Real Badminton.
                        </h1>
                    </FadeUp>

                    <FadeUp delay={0.16}>
                        <p className="text-lg text-neutral-400 max-w-xl mx-auto mb-10 leading-relaxed">
                            Upload a clip and get instant feedback on your strokes, body positioning, and shot selection from an AI that knows badminton.
                        </p>
                    </FadeUp>

                    <FadeUp delay={0.22}>
                        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                            <motion.button
                                onClick={() => navigate('/analyze')}
                                whileHover={{ scale: 1.04 }}
                                whileTap={{ scale: 0.97 }}
                                className="w-full sm:w-auto px-8 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg text-sm transition-colors shadow-lg shadow-emerald-900/30"
                            >
                                Start Analyzing for Free
                            </motion.button>
                            <a
                                href="#how-it-works"
                                className="w-full sm:w-auto px-8 py-3 border border-neutral-700 hover:border-neutral-500 text-neutral-300 hover:text-white font-medium rounded-lg text-sm transition-colors text-center"
                            >
                                See how it works
                            </a>
                        </div>
                    </FadeUp>

                    {/* Mock analysis card */}
                    <FadeUp delay={0.32}>
                        <div className="mt-16 mx-auto max-w-sm bg-neutral-900 border border-neutral-800 rounded-xl p-4 text-left shadow-2xl">
                            <div className="flex items-center justify-between mb-3">
                                <span className="text-xs text-neutral-500 font-medium flex items-center gap-1.5">
                                    <Icon name="analytics" size={14} className="text-emerald-500" />
                                    Analysis Results
                                </span>
                                <span className="text-[10px] bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
                                    Advanced
                                </span>
                            </div>
                            <div className="flex justify-between items-end mb-2">
                                <span className="text-2xl font-bold text-emerald-400">Advanced</span>
                                <span className="text-sm text-neutral-400 font-mono">8 / 10</span>
                            </div>
                            <div className="w-full bg-neutral-800 h-1.5 rounded-full overflow-hidden mb-3">
                                <motion.div
                                    initial={{ width: 0 }}
                                    whileInView={{ width: '80%' }}
                                    viewport={{ once: true }}
                                    transition={{ duration: 0.9, ease: 'easeOut', delay: 0.6 }}
                                    className="h-full bg-emerald-500 rounded-full"
                                />
                            </div>
                            <div className="flex flex-wrap gap-2 mt-3">
                                <div className="px-2 py-1 bg-blue-500/10 border border-blue-500/30 rounded text-[10px] font-medium text-blue-400 flex items-center gap-1.5">
                                    <Icon name="pan_tool_alt" size={12} />
                                    Forehand Clear
                                </div>
                                <div className="px-2 py-1 bg-purple-500/10 border border-purple-500/30 rounded text-[10px] font-medium text-purple-400 flex items-center gap-1.5">
                                    <Icon name="explore" size={12} />
                                    Deep Lift
                                </div>
                                <div className="px-2 py-1 bg-rose-500/10 border border-rose-500/30 rounded text-[10px] font-medium text-rose-400 flex items-center gap-1.5">
                                    <Icon name="location_on" size={12} />
                                    Rear Court
                                </div>
                                <div className="px-2 py-1 bg-amber-500/10 border border-amber-500/30 rounded text-[10px] font-medium text-amber-400 flex items-center gap-1.5">
                                    <Icon name="psychology" size={12} />
                                    Offensive
                                </div>
                            </div>
                            <div className="mt-3 pt-3 border-t border-neutral-800">
                                <p className="text-xs text-neutral-300 leading-relaxed">
                                    <span className="text-emerald-500">•</span> Rotate your non-racket shoulder further back at the start of the swing to generate more power and increase shuttle speed.
                                </p>
                            </div>
                        </div>
                    </FadeUp>
                </div>
            </section>

            {/* Stats Bar */}
            <section className="border-y border-neutral-800 bg-neutral-900/40">
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
            <section id="features" className="py-16 md:py-28 px-6">
                <div className="max-w-5xl mx-auto">
                    <FadeUp className="text-center mb-16">
                        <span className="text-xs font-semibold text-emerald-500 uppercase tracking-widest">Features</span>
                        <h2 className="text-3xl md:text-4xl font-bold mt-3 tracking-tight">
                            Everything a coach sees,<br />in seconds
                        </h2>
                        <p className="text-neutral-400 mt-4 max-w-lg mx-auto leading-relaxed">
                            IsoCourt gives you the kind of detailed feedback that used to require a professional coach, available instantly after every clip.
                        </p>
                    </FadeUp>

                    <div className="grid md:grid-cols-3 gap-5">
                        {features.map(({ icon, label, description }, i) => (
                            <FadeUp key={label} delay={i * 0.1}>
                                <div className="h-full bg-neutral-900 border border-neutral-800 hover:border-neutral-700 rounded-xl p-6 transition-all duration-300">
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
                            See it in action
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
                            From clip to coaching<br />in three steps
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
                        Ready to level up<br />your game?
                    </h2>
                    <p className="text-neutral-400 mb-10 text-lg">
                        No account needed. Drop a clip and get instant feedback.
                    </p>
                    <motion.button
                        onClick={() => navigate('/analyze')}
                        whileHover={{ scale: 1.04 }}
                        whileTap={{ scale: 0.97 }}
                        className="inline-flex items-center gap-2 px-8 py-3.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg text-sm transition-colors shadow-lg shadow-emerald-900/30"
                    >
                        <Icon name="videocam" size={18} />
                        Analyze My Stroke
                    </motion.button>
                </FadeUp>
            </section>

            {/* Footer */}
            <footer className="border-t border-neutral-800 py-8 px-6">
                <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <Logo size={18} className="text-emerald-500" />
                        <span className="text-sm font-semibold">
                            Iso<span className="text-emerald-500">Court</span>
                        </span>
                    </div>
                    <p className="text-xs text-neutral-600">
                        AI-powered badminton stroke analysis · Built by a badminton player for badminton players
                    </p>
                </div>
            </footer>
        </div>
    )
}
