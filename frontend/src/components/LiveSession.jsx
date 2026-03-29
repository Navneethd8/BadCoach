import { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import ReactGA from 'react-ga4'
import Logo from './Logo'

function Icon({ name, size = 20, className = '' }) {
    return (
        <span className={`material-symbols-outlined ${className}`} style={{ fontSize: size }}>
            {name}
        </span>
    )
}

const FRAME_INTERVAL_MS = 200

export default function LiveSession() {
    const navigate = useNavigate()
    const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
    const wsBase = apiUrl.replace(/^http/, 'ws')

    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const wsRef = useRef(null)
    const streamRef = useRef(null)
    const intervalRef = useRef(null)
    const chatScrollRef = useRef(null)

    const [sessionId, setSessionId] = useState(null)
    const [status, setStatus] = useState('idle')   // idle | connecting | live | paused | error | capacity
    const [statusMsg, setStatusMsg] = useState(null)
    const [errorMsg, setErrorMsg] = useState(null)
    const [lastResult, setLastResult] = useState(null)
    const [onBreak, setOnBreak] = useState(null)
    const onBreakRef = useRef(null)
    const [cameraError, setCameraError] = useState(null)
    const [chatLog, setChatLog] = useState([])
    const [voiceEnabled, setVoiceEnabled] = useState(true)
    const voiceEnabledRef = useRef(true)

    const bestVoiceRef = useRef(null)
    const pickBestVoice = useCallback(() => {
        const voices = window.speechSynthesis?.getVoices() || []
        const en = voices.filter(v => v.lang.startsWith('en'))
        if (!en.length) return
        const ranked = [
            v => /Google UK English Male/i.test(v.name),
            v => /Google UK English Female/i.test(v.name),
            v => /Google US English/i.test(v.name),
            v => /Daniel.*Premium/i.test(v.name),
            v => /Samantha.*Enhanced/i.test(v.name),
            v => /\(Enhanced\)/i.test(v.name) || /\(Premium\)/i.test(v.name),
            v => /Microsoft.*Online/i.test(v.name) && v.lang.startsWith('en'),
            v => !v.localService,
        ]
        for (const test of ranked) {
            const match = en.find(test)
            if (match) { bestVoiceRef.current = match; return }
        }
        bestVoiceRef.current = en[0]
    }, [])

    useEffect(() => {
        pickBestVoice()
        window.speechSynthesis?.addEventListener('voiceschanged', pickBestVoice)
        return () => window.speechSynthesis?.removeEventListener('voiceschanged', pickBestVoice)
    }, [pickBestVoice])

    const speak = useCallback((text) => {
        if (!voiceEnabledRef.current || !window.speechSynthesis) return
        window.speechSynthesis.cancel()
        const utterance = new SpeechSynthesisUtterance(text)
        utterance.rate = 1.05
        utterance.pitch = 1.0
        utterance.volume = 1.0
        if (bestVoiceRef.current) utterance.voice = bestVoiceRef.current
        window.speechSynthesis.speak(utterance)
    }, [])

    const toggleVoice = useCallback(() => {
        setVoiceEnabled(prev => {
            const next = !prev
            voiceEnabledRef.current = next
            if (!next) window.speechSynthesis?.cancel()
            return next
        })
    }, [])

    useEffect(() => {
        const el = chatScrollRef.current
        if (!el) return
        el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
    }, [chatLog])

    const stopCamera = useCallback(() => {
        window.speechSynthesis?.cancel()
        if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
        if (wsRef.current) { wsRef.current.close(); wsRef.current = null }
        if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
        if (videoRef.current) { videoRef.current.srcObject = null }
    }, [])

    const pushChat = useCallback((type, text) => {
        setChatLog(prev => [...prev, { id: Date.now() + Math.random(), type, text, ts: new Date() }])
    }, [])

    const endSession = useCallback(async () => {
        stopCamera()
        if (sessionId) {
            try { await fetch(`${apiUrl}/live/sessions/${sessionId}`, { method: 'DELETE' }) } catch {}
        }
        setSessionId(null)
        setStatus('idle')
        setStatusMsg(null)
        setLastResult(null)
    }, [sessionId, apiUrl, stopCamera])

    function beginFrameLoop(video, canvas, ws) {
        if (intervalRef.current) clearInterval(intervalRef.current)
        intervalRef.current = setInterval(() => {
            if (!video || video.readyState < 2) return
            if (!ws || ws.readyState !== WebSocket.OPEN) return
            canvas.width = video.videoWidth || 640
            canvas.height = video.videoHeight || 480
            const ctx = canvas.getContext('2d')
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
            canvas.toBlob(blob => {
                if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob)
            }, 'image/jpeg', 0.6)
        }, FRAME_INTERVAL_MS)
    }

    const pauseSession = useCallback(() => {
        if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
        setStatus('paused')
        pushChat('system', 'Session paused — take a break.')
    }, [pushChat])

    const resumeSession = useCallback(() => {
        const v = videoRef.current, c = canvasRef.current, w = wsRef.current
        if (v && c && w) beginFrameLoop(v, c, w)
        setStatus('live')
        pushChat('system', 'Resumed — back on court!')
    }, [pushChat])

    useEffect(() => () => stopCamera(), [stopCamera])

    const startSession = async () => {
        setStatus('connecting')
        setErrorMsg(null)
        setCameraError(null)
        setLastResult(null)
        setChatLog([])
        setStatusMsg(null)

        ReactGA.event({ category: 'Live', action: 'Session Start Attempt' })

        let sid
        try {
            const res = await fetch(`${apiUrl}/live/sessions`, { method: 'POST' })
            if (res.status === 503) {
                const body = await res.json().catch(() => ({}))
                setStatus('capacity')
                setErrorMsg(body?.detail?.error || 'Live sessions are at capacity. Try again later.')
                ReactGA.event({ category: 'Live', action: 'Capacity Reached' })
                return
            }
            if (!res.ok) throw new Error(`Server error ${res.status}`)
            const data = await res.json()
            sid = data.session_id
            setSessionId(sid)
        } catch (e) {
            setStatus('error')
            setErrorMsg(e.message || 'Failed to start session')
            return
        }

        let stream
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
                audio: false,
            })
            streamRef.current = stream
            if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play() }
        } catch {
            setCameraError('Camera access denied or unavailable.')
            setStatus('error')
            try { await fetch(`${apiUrl}/live/sessions/${sid}`, { method: 'DELETE' }) } catch {}
            return
        }

        const ws = new WebSocket(`${wsBase}/live/sessions/${sid}/ws`)
        wsRef.current = ws

        ws.onopen = () => {
            setStatus('live')
            pushChat('system', 'Session started — point your camera at the court.')
            ReactGA.event({ category: 'Live', action: 'Session Started' })
            beginFrameLoop(videoRef.current, canvasRef.current, ws)
        }

        ws.onmessage = (msg) => {
            try {
                const data = JSON.parse(msg.data)
                if (data.event === 'status') {
                    setStatusMsg(data.message)
                    pushChat('system', data.message)
                } else if (data.event === 'break') {
                    setOnBreak(data.reason || 'game_break')
                    onBreakRef.current = data.reason || 'game_break'
                    setLastResult(null)
                    pushChat('system', data.message || 'Break detected.')
                } else if (data.event === 'analysis') {
                    if (onBreakRef.current) pushChat('system', 'Play resumed — analyzing...')
                    setOnBreak(null)
                    onBreakRef.current = null
                    setLastResult(data)
                    setStatusMsg(null)
                    const conf = data.confidence != null ? `${(data.confidence * 100).toFixed(0)}%` : ''
                    pushChat('analysis', `${data.label} ${conf}`)
                } else if (data.event === 'commentary') {
                    if (data.text) { pushChat('coach', data.text); speak(data.text) }
                } else if (data.event === 'error') {
                    setErrorMsg(data.error)
                    pushChat('system', `Error: ${data.error}`)
                }
            } catch {}
        }

        ws.onerror = () => { setStatus('error'); setErrorMsg('WebSocket connection failed.') }
        ws.onclose = () => { if (intervalRef.current) clearInterval(intervalRef.current) }
    }

    const getQualityColor = (q) => {
        const s = String(q).toLowerCase()
        if (s.includes('elite') || s.includes('expert')) return 'text-emerald-400'
        if (s.includes('advanced')) return 'text-emerald-500'
        if (s.includes('proficient')) return 'text-cyan-400'
        if (s.includes('competent')) return 'text-amber-400'
        if (s.includes('developing')) return 'text-orange-400'
        return 'text-red-400'
    }

    const fmtTime = (d) => `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`

    return (
        <div className="min-h-screen bg-neutral-950 text-neutral-100">
            <nav className="sticky top-0 z-50 border-b border-neutral-800/60 bg-neutral-950/80 backdrop-blur-md">
                <div className="mx-auto flex h-14 max-w-[min(1920px,calc(100vw-1.25rem))] items-center justify-between px-3 sm:px-5">
                    <button onClick={() => navigate('/')} className="flex items-center gap-2 focus:outline-none">
                        <Logo size={22} className="text-emerald-500" />
                        <span className="text-base font-semibold tracking-tight">Iso<span className="text-emerald-500">Court</span></span>
                    </button>
                    <button onClick={() => navigate('/analyze')} className="text-sm text-neutral-400 hover:text-white transition-colors">
                        Clip Analysis
                    </button>
                </div>
            </nav>

            <div className="mx-auto max-w-[min(1920px,calc(100vw-1.25rem))] px-3 py-6 sm:px-5">
                <h1 className="text-2xl font-bold mb-1 tracking-tight">Live <span className="text-emerald-500">Session</span></h1>
                <p className="text-sm text-neutral-500 mb-5">Point your camera at the court and get real-time feedback.</p>

                {/* Capacity / error banners */}
                {status === 'capacity' && (
                    <div className="mb-4 flex items-start gap-3 bg-amber-950/90 border border-amber-700/60 text-amber-200 text-sm px-5 py-3.5 rounded-xl">
                        <Icon name="hourglass_top" size={18} className="text-amber-400 flex-shrink-0 mt-0.5" />
                        <div>
                            <p className="font-semibold text-amber-300 mb-0.5">Live sessions at capacity</p>
                            <p className="text-xs text-amber-200/80">{errorMsg}</p>
                        </div>
                        <button onClick={() => { setStatus('idle'); setErrorMsg(null) }} className="ml-auto text-amber-400 hover:text-amber-200"><Icon name="close" size={16} /></button>
                    </div>
                )}
                {(status === 'error' || cameraError) && (
                    <div className="mb-4 flex items-start gap-3 bg-red-950/80 border border-red-700/50 text-red-200 text-sm px-5 py-3.5 rounded-xl">
                        <Icon name="error" size={18} className="text-red-400 flex-shrink-0 mt-0.5" />
                        <div>
                            <p className="font-semibold text-red-300 mb-0.5">Something went wrong</p>
                            <p className="text-xs text-red-200/80">{cameraError || errorMsg}</p>
                        </div>
                        <button onClick={() => { setStatus('idle'); setErrorMsg(null); setCameraError(null) }} className="ml-auto text-red-400 hover:text-red-200"><Icon name="close" size={16} /></button>
                    </div>
                )}

                <div className="flex flex-col gap-5 lg:flex-row lg:items-stretch lg:gap-4 xl:gap-5 min-[1700px]:gap-6">

                    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                        <div className="relative aspect-video w-full overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900 shadow-lg shadow-black/20">
                            <video ref={videoRef} className="w-full h-full object-cover" playsInline muted />
                            <canvas ref={canvasRef} className="hidden" />

                            {status === 'idle' && (
                                <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-neutral-900/90">
                                    <Icon name="videocam" size={48} className="text-neutral-700" />
                                    <motion.button onClick={startSession} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
                                        className="px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg text-sm transition-colors shadow-lg shadow-emerald-900/30">
                                        Start Game
                                    </motion.button>
                                </div>
                            )}

                            {status === 'connecting' && (
                                <div className="absolute inset-0 flex items-center justify-center bg-neutral-900/80">
                                    <div className="flex items-center gap-3 text-neutral-400">
                                        <span className="w-5 h-5 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
                                        <span className="text-sm">Starting session...</span>
                                    </div>
                                </div>
                            )}

                            {/* LIVE / PAUSED badge + status line */}
                            {(status === 'live' || status === 'paused') && (
                                <div className="absolute top-3 left-3 flex items-center gap-2">
                                    {status === 'live' ? (
                                        <div className="flex items-center gap-1.5 bg-red-600/90 text-white text-xs font-semibold px-3 py-1.5 rounded-full shadow-lg">
                                            <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
                                            LIVE
                                        </div>
                                    ) : (
                                        <div className="flex items-center gap-1.5 bg-amber-600/90 text-white text-xs font-semibold px-3 py-1.5 rounded-full shadow-lg">
                                            <Icon name="pause" size={12} />
                                            PAUSED
                                        </div>
                                    )}
                                    <AnimatePresence mode="wait">
                                        {statusMsg && status === 'live' && (
                                            <motion.div key={statusMsg} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0 }}
                                                className="bg-neutral-900/80 backdrop-blur-sm text-neutral-300 text-xs px-3 py-1.5 rounded-full">
                                                {statusMsg}
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            )}

                            {/* Paused overlay */}
                            {status === 'paused' && (
                                <div className="absolute inset-0 bg-neutral-950/50 flex items-center justify-center">
                                    <motion.button onClick={resumeSession} whileHover={{ scale: 1.06 }} whileTap={{ scale: 0.95 }}
                                        className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg text-sm transition-colors shadow-lg">
                                        <Icon name="play_arrow" size={16} className="align-middle mr-1" />Resume
                                    </motion.button>
                                </div>
                            )}

                            {(status === 'live' || status === 'paused') && (
                                <div className="absolute top-3 right-3 flex items-center gap-2">
                                    {status === 'live' && (
                                        <button onClick={pauseSession} className="bg-neutral-800/80 hover:bg-neutral-700 text-white text-xs font-medium px-3 py-1.5 rounded-lg transition-colors">
                                            <Icon name="pause" size={13} className="align-middle mr-0.5" />Pause
                                        </button>
                                    )}
                                    <button onClick={endSession} className="bg-red-900/60 hover:bg-red-800/80 text-red-200 text-xs font-medium px-3 py-1.5 rounded-lg transition-colors">
                                        End Session
                                    </button>
                                </div>
                            )}

                            {/* Break overlay */}
                            {onBreak && status === 'live' && (
                                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                                    className={`absolute bottom-0 inset-x-0 rounded-b-lg px-4 py-3 flex items-center justify-center gap-2 ${onBreak === 'no_badminton' ? 'bg-amber-950/80' : 'bg-neutral-950/80'} backdrop-blur-sm`}>
                                    <Icon name={onBreak === 'no_badminton' ? 'videocam_off' : 'pause_circle'} size={16}
                                        className={onBreak === 'no_badminton' ? 'text-amber-500' : 'text-neutral-500'} />
                                    <span className={`text-xs ${onBreak === 'no_badminton' ? 'text-amber-300' : 'text-neutral-400'}`}>
                                        {onBreak === 'no_badminton'
                                            ? 'No badminton detected — point camera at the court'
                                            : 'Break in play — waiting for action'}
                                    </span>
                                </motion.div>
                            )}

                            {/* Metrics HUD overlay */}
                            <AnimatePresence>
                                {lastResult && !onBreak && (status === 'live' || status === 'paused') && (
                                    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                                        className="absolute bottom-0 inset-x-0 rounded-b-lg bg-neutral-950/75 backdrop-blur-sm px-3 py-2.5">
                                        <div className="grid grid-cols-3 gap-x-3 gap-y-1.5 text-center">
                                            <div>
                                                <p className="text-[8px] text-neutral-500 uppercase tracking-wider">Stroke</p>
                                                <p className="text-xs font-bold text-white truncate">{lastResult.label}</p>
                                            </div>
                                            <div>
                                                <p className="text-[8px] text-neutral-500 uppercase tracking-wider">Technique</p>
                                                <p className="text-[11px] font-medium text-neutral-200 truncate">{lastResult.metrics?.technique?.label || '—'}</p>
                                            </div>
                                            <div>
                                                <p className="text-[8px] text-neutral-500 uppercase tracking-wider">Placement</p>
                                                <p className="text-[11px] font-medium text-neutral-200 truncate">{lastResult.metrics?.placement?.label || '—'}</p>
                                            </div>
                                            <div>
                                                <p className="text-[8px] text-neutral-500 uppercase tracking-wider">Position</p>
                                                <p className="text-[11px] font-medium text-neutral-200 truncate">{lastResult.metrics?.position?.label || '—'}</p>
                                            </div>
                                            <div>
                                                <p className="text-[8px] text-neutral-500 uppercase tracking-wider">Intent</p>
                                                <p className="text-[11px] font-medium text-neutral-200 truncate">{lastResult.metrics?.intent?.label || '—'}</p>
                                            </div>
                                            <div>
                                                <p className="text-[8px] text-neutral-500 uppercase tracking-wider">Quality</p>
                                                <p className={`text-[11px] font-semibold truncate ${getQualityColor(lastResult.metrics?.quality)}`}>{lastResult.metrics?.quality || '—'}</p>
                                            </div>
                                        </div>
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>
                    </div>

                    <aside
                        className="flex h-[min(18rem,42vh)] w-full shrink-0 flex-col overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900 sm:h-[min(20rem,44vh)] lg:min-h-0 lg:max-h-[min(40rem,calc(100dvh-9rem))] lg:w-72 lg:shrink-0 xl:w-80 min-[1700px]:w-96"
                        aria-label="AI Coach chat"
                    >
                        <div className="flex items-center gap-2 px-4 py-3 border-b border-neutral-800 flex-shrink-0">
                            <Icon name="smart_toy" size={16} className="text-emerald-500" />
                            <span className="text-xs font-semibold text-neutral-400 uppercase tracking-wider">AI Coach</span>
                            <div className="ml-auto flex items-center gap-2">
                                {chatLog.length > 0 && (
                                    <span className="text-[10px] text-neutral-600">{chatLog.length} messages</span>
                                )}
                                <button onClick={toggleVoice} title={voiceEnabled ? 'Mute voice' : 'Unmute voice'}
                                    className={`p-1 rounded transition-colors ${voiceEnabled ? 'text-emerald-500 hover:text-emerald-400' : 'text-neutral-600 hover:text-neutral-400'}`}>
                                    <Icon name={voiceEnabled ? 'volume_up' : 'volume_off'} size={16} />
                                </button>
                            </div>
                        </div>
                        <div
                            ref={chatScrollRef}
                            className="flex-1 min-h-0 overflow-y-auto overscroll-y-contain px-4 py-3 space-y-3 scroll-smooth"
                            role="log"
                            aria-live="polite"
                            aria-relevant="additions"
                        >
                            {chatLog.length === 0 && (
                                <div className="flex flex-col items-center justify-center h-full text-center gap-2 py-8">
                                    <Icon name="sports" size={28} className="text-neutral-800" />
                                    <p className="text-xs text-neutral-600">Start a game to see live coaching commentary here.</p>
                                </div>
                            )}
                            <AnimatePresence initial={false}>
                                {chatLog.map(msg => (
                                    <motion.div key={msg.id} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}
                                        className="flex gap-2.5 items-start">
                                        <span className="text-[10px] text-neutral-700 font-mono mt-1 flex-shrink-0 w-12">{fmtTime(msg.ts)}</span>
                                        {msg.type === 'coach' ? (
                                            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg px-3 py-2.5 text-xs text-emerald-300 leading-relaxed flex-1 min-w-0">
                                                {msg.text}
                                            </div>
                                        ) : msg.type === 'analysis' ? (
                                            <div className="bg-neutral-800/60 rounded-lg px-3 py-1.5 text-[11px] text-neutral-400 flex-1 min-w-0">
                                                <Icon name="sports_tennis" size={11} className="text-neutral-600 align-middle mr-1" />{msg.text}
                                            </div>
                                        ) : (
                                            <div className="text-[11px] text-neutral-500 italic flex-1 min-w-0">{msg.text}</div>
                                        )}
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>
                    </aside>

                </div>
            </div>
        </div>
    )
}
