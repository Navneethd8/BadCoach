import { useState, useCallback, useRef, useEffect } from 'react'
import Logo from './components/Logo';
import { useDropzone } from 'react-dropzone'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { motion, useReducedMotion } from 'framer-motion'
import ReactGA from "react-ga4"

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

export default function App() {
    const navigate = useNavigate()
    const [file, setFile] = useState(null)
    const [result, setResult] = useState(null)
    const [loading, setLoading] = useState(false)
    const [preview, setPreview] = useState(null)
    const [loadingStep, setLoadingStep] = useState(-1)
    const [capacityError, setCapacityError] = useState(null)   // 503 at-capacity state
    const videoRef = useRef(null)
    const loadingTimers = useRef([])
    const shouldReduceMotion = useReducedMotion()
    const retryTimerRef = useRef(null)

    // Mobile detection
    const [isMobile, setIsMobile] = useState(window.innerWidth < 768)
    useEffect(() => {
        const handleResize = () => setIsMobile(window.innerWidth < 768)
        window.addEventListener('resize', handleResize)
        return () => window.removeEventListener('resize', handleResize)
    }, [])

    // Camera recording state (Laptop)
    const [inputMode, setInputMode] = useState('upload') // 'upload' | 'record'
    const [isRecording, setIsRecording] = useState(false)
    const [cameraError, setCameraError] = useState(null)
    const [recordingSeconds, setRecordingSeconds] = useState(0)
    const cameraPreviewRef = useRef(null)
    const mediaRecorderRef = useRef(null)
    const cameraStreamRef = useRef(null)
    const recordedChunksRef = useRef([])
    const recordingTimerRef = useRef(null)
    const [isFullScreen, setIsFullScreen] = useState(false)

    // Native Camera Reference
    const nativeVideoInputRef = useRef(null)

    const loadingSteps = [
        { icon: 'movie_filter', label: 'Splitting clip into frames' },
        { icon: 'directions_run', label: 'Tracing poses' },
        { icon: 'query_stats', label: 'Analyzing strokes' },
        { icon: 'rate_review', label: 'Generating feedback' },
    ]

    // Weighted distribution: frame splitting is fast, pose tracing and stroke analysis are the bulk
    const stepWeights = [0.10, 0.35, 0.45, 0.10]

    const startLoadingSteps = () => {
        // Get video duration from the element
        const duration = videoRef.current?.duration || 3
        // Rough estimate: ~2s processing per second of video, minimum 4s total
        const estimatedTime = Math.max(4, duration * 2) * 1000

        setLoadingStep(0)

        let elapsed = 0
        loadingTimers.current = []
        for (let i = 1; i < loadingSteps.length; i++) {
            elapsed += estimatedTime * stepWeights[i - 1]
            const timer = setTimeout(() => setLoadingStep(i), elapsed)
            loadingTimers.current.push(timer)
        }
    }

    const stopLoadingSteps = () => {
        loadingTimers.current.forEach(clearTimeout)
        loadingTimers.current = []
        setLoadingStep(-1)
    }

    const onDrop = useCallback((acceptedFiles) => {
        const file = acceptedFiles[0]
        setFile(file)
        setPreview(URL.createObjectURL(file))
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'video/*': [] }
    })

    const handleNativeVideoSelect = (e) => {
        const file = e.target.files?.[0]
        if (file) {
            setFile(file)
            setPreview(URL.createObjectURL(file))
        }
    }

    // ── Laptop Camera helpers ──────────────────────────────────────────
    const openCamera = async () => {
        setCameraError(null)
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
            cameraStreamRef.current = stream
            if (cameraPreviewRef.current) {
                cameraPreviewRef.current.srcObject = stream
                cameraPreviewRef.current.play().catch(e => console.error("Error playing camera stream:", e))
            }
        } catch (err) {
            if (err.name === 'NotAllowedError') {
                setCameraError('Camera permission denied. Please allow camera access and try again.')
            } else if (err.name === 'NotFoundError') {
                setCameraError('No camera found on this device.')
            } else {
                setCameraError('Could not access camera: ' + err.message)
            }
        }
    }

    const closeCamera = () => {
        if (cameraStreamRef.current) {
            cameraStreamRef.current.getTracks().forEach(t => t.stop())
            cameraStreamRef.current = null
        }
        if (cameraPreviewRef.current) {
            cameraPreviewRef.current.srcObject = null
        }
        if (recordingTimerRef.current) clearInterval(recordingTimerRef.current)
        setIsRecording(false)
        setRecordingSeconds(0)
    }

    const startRecording = () => {
        if (!cameraStreamRef.current) return
        recordedChunksRef.current = []
        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
            ? 'video/webm;codecs=vp9'
            : MediaRecorder.isTypeSupported('video/webm')
                ? 'video/webm'
                : 'video/mp4'
        const recorder = new MediaRecorder(cameraStreamRef.current, { mimeType })
        recorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) recordedChunksRef.current.push(e.data)
        }
        recorder.onstop = () => {
            const ext = mimeType.includes('mp4') ? 'mp4' : 'webm'
            const blob = new Blob(recordedChunksRef.current, { type: mimeType })
            const recorded = new File([blob], `recording.${ext}`, { type: mimeType })
            setFile(recorded)
            setPreview(URL.createObjectURL(blob))
            closeCamera()
            ReactGA.event({ category: 'Video', action: 'Camera Recording Captured', label: `${recordingSeconds}s` })
        }
        mediaRecorderRef.current = recorder
        recorder.start(250) // collect chunks every 250 ms
        setIsRecording(true)
        setRecordingSeconds(0)
        recordingTimerRef.current = setInterval(() => setRecordingSeconds(s => s + 1), 1000)
    }

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop()
        }
        if (recordingTimerRef.current) clearInterval(recordingTimerRef.current)
        setIsRecording(false)
    }

    const switchMode = (mode) => {
        if (mode === inputMode) return
        if (inputMode === 'record') {
            stopRecording()
            closeCamera()
        }
        setFile(null)
        setPreview(null)
        setResult(null)
        setCameraError(null)
        setInputMode(mode)
    }

    useEffect(() => {
        if (!isMobile && inputMode === 'record') {
            openCamera()
        }
        return () => {
            if (!isMobile && inputMode === 'record') {
                stopRecording()
                closeCamera()
            }
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [inputMode, isMobile])

    useEffect(() => {
        if (isMobile) {
            closeCamera()
        }
        return () => {
            closeCamera()
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isMobile])

    const toggleFullScreen = () => {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch((err) => {
                console.error(`Error attempting to enable fullscreen: ${err.message}`);
            });
            setIsFullScreen(true)
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
                setIsFullScreen(false)
            }
        }
    }

    useEffect(() => {
        const handleFullscreenChange = () => {
            setIsFullScreen(!!document.fullscreenElement)
        }
        document.addEventListener('fullscreenchange', handleFullscreenChange)
        return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }, [])

    const formatSeconds = (s) => {
        const m = Math.floor(s / 60).toString().padStart(2, '0')
        const sec = (s % 60).toString().padStart(2, '0')
        return `${m}:${sec}`
    }

    const handleSubmit = async () => {
        if (!file) return

        setLoading(true)
        setCapacityError(null)
        startLoadingSteps()
        const formData = new FormData()
        formData.append('file', file)

        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
            const response = await axios.post(`${apiUrl}/analyze`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            })

            if (response.data.validation_failed) {
                const isOverDuration = response.data.over_duration_limit || false
                setResult({
                    validation_error: true,
                    error_message: response.data.error,
                    over_duration_limit: isOverDuration,
                    validation_details: response.data.validation_details
                })
                if (isOverDuration) {
                    ReactGA.event({ category: "Video", action: "Clip Too Long", label: file?.name })
                } else {
                    ReactGA.event({ category: "Video", action: "Validation Failed", label: response.data.error })
                }
            } else {
                setResult(response.data)
                ReactGA.event({
                    category: "Video",
                    action: "Clip Analyzed",
                    label: response.data.quality_label,
                    value: response.data.quality_numeric
                });
            }
        } catch (error) {
            console.error("Error uploading file:", error)

            // 503 — server at capacity
            if (error.response?.status === 503) {
                const retryAfter = error.response?.data?.detail?.retry_after || 30
                setCapacityError(retryAfter)
                if (retryTimerRef.current) clearTimeout(retryTimerRef.current)
                retryTimerRef.current = setTimeout(() => setCapacityError(null), retryAfter * 1000)
                ReactGA.event({ category: "Video", action: "Server At Capacity", label: file?.name })
            } else if (error.response?.data?.validation_failed) {
                const isOverDuration = error.response.data.over_duration_limit || false
                setResult({
                    validation_error: true,
                    error_message: error.response.data.error,
                    over_duration_limit: isOverDuration,
                    validation_details: error.response.data.validation_details
                })
                if (isOverDuration) {
                    ReactGA.event({ category: "Video", action: "Clip Too Long", label: file?.name })
                } else {
                    ReactGA.event({ category: "Video", action: "Validation Failed", label: error.response.data.error })
                }
            } else {
                const errorMessage = error.response?.data?.detail || error.message || "Error analyzing video"
                ReactGA.event({ category: "Video", action: "Analysis Failed", label: errorMessage })
                alert(`Analysis failed: ${errorMessage}`)
            }
        } finally {
            setLoading(false)
            stopLoadingSteps()
        }
    }

    // Stream-based analysis using /analyze/stream (SSE).
    // Fires GA events for each key phase: start, each window received, done, error.
    const handleStreamAnalysis = async () => {
        if (!file) return

        setLoading(true)
        setCapacityError(null)
        startLoadingSteps()

        const apiUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
        const formData = new FormData()
        formData.append('file', file)

        ReactGA.event({ category: 'Video', action: 'Stream Started', label: file.name })

        let windowCount = 0

        try {
            const response = await fetch(`${apiUrl}/analyze/stream`, {
                method: 'POST',
                body: formData,
            })

            if (response.status === 503) {
                const body = await response.json().catch(() => ({}))
                const retryAfter = body?.detail?.retry_after || 30
                setCapacityError(retryAfter)
                if (retryTimerRef.current) clearTimeout(retryTimerRef.current)
                retryTimerRef.current = setTimeout(() => setCapacityError(null), retryAfter * 1000)
                ReactGA.event({ category: 'Video', action: 'Server At Capacity', label: file.name })
                return
            }

            const reader = response.body.getReader()
            const decoder = new TextDecoder()
            let buffer = ''
            const streamedTimeline = []

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                buffer += decoder.decode(value, { stream: true })
                const lines = buffer.split('\n\n')
                buffer = lines.pop() // keep incomplete chunk

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue
                    let parsed
                    try { parsed = JSON.parse(line.slice(6)) } catch { continue }

                    if (parsed.event === 'progress') {
                        windowCount++
                        streamedTimeline.push(parsed)
                        // Fire GA every 5 windows to avoid flooding analytics
                        if (windowCount % 5 === 1) {
                            ReactGA.event({
                                category: 'Video',
                                action: 'Stream Window Received',
                                label: parsed.label,
                                value: windowCount,
                            })
                        }
                    } else if (parsed.event === 'done') {
                        const summary = parsed.summary || {}
                        setResult({ ...summary, timeline: streamedTimeline })
                        ReactGA.event({
                            category: 'Video',
                            action: 'Stream Complete',
                            label: summary.action || 'Unknown',
                            value: windowCount,
                        })
                    } else if (parsed.event === 'error') {
                        const isOverDuration = parsed.over_duration_limit || false
                        if (isOverDuration) {
                            setResult({ validation_error: true, error_message: parsed.error, over_duration_limit: true })
                            ReactGA.event({ category: 'Video', action: 'Clip Too Long', label: file.name })
                        } else {
                            ReactGA.event({ category: 'Video', action: 'Stream Error', label: parsed.error })
                            setResult({ validation_error: true, error_message: parsed.error, over_duration_limit: false })
                        }
                    }
                }
            }
        } catch (err) {
            console.error('Stream error:', err)
            ReactGA.event({ category: 'Video', action: 'Analysis Failed', label: err.message })
        } finally {
            setLoading(false)
            stopLoadingSteps()
        }
    }

    const handleTimelineClick = (timestamp) => {
        if (!videoRef.current) return

        const parts = timestamp.split(':')
        let seconds = 0
        if (parts.length === 2) {
            seconds = parseInt(parts[0]) * 60 + parseInt(parts[1])
        } else if (parts.length === 1) {
            seconds = parseInt(parts[0])
        }

        videoRef.current.currentTime = seconds
        videoRef.current.play()
        videoRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }

    const getQualityColor = (quality) => {
        // Updated for 10-point scale
        const q = String(quality).toLowerCase()
        if (q.includes('elite') || q.includes('expert')) return 'text-emerald-400'
        if (q.includes('advanced')) return 'text-emerald-500' // New cyan/emerald
        if (q.includes('proficient')) return 'text-cyan-400'
        if (q.includes('competent')) return 'text-amber-400'
        if (q.includes('developing') || q.includes('emerging')) return 'text-orange-400'
        return 'text-rose-400'
    }

    const getQualityBarColor = (quality) => {
        // Updated for 10-point scale
        const q = String(quality).toLowerCase()
        if (q.includes('elite') || q.includes('expert')) return 'bg-emerald-500'
        if (q.includes('advanced')) return 'bg-emerald-600'
        if (q.includes('proficient')) return 'bg-cyan-500'
        if (q.includes('competent')) return 'bg-amber-500'
        if (q.includes('developing') || q.includes('emerging')) return 'bg-orange-500'
        return 'bg-rose-500'
    }

    return (
        <div className="min-h-screen w-screen bg-neutral-950 text-neutral-100 p-6 md:p-10">
            <div className="max-w-2xl mx-auto">

                {/* Capacity toast banner */}
                {capacityError !== null && (
                    <div className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex items-start gap-3 bg-amber-950/90 border border-amber-700/60 text-amber-200 text-sm px-5 py-3.5 rounded-xl shadow-2xl backdrop-blur-sm max-w-sm w-full">
                        <Icon name="hourglass_top" size={18} className="text-amber-400 flex-shrink-0 mt-0.5" />
                        <div>
                            <p className="font-semibold text-amber-300 mb-0.5">IsoCourt is fully loaded right now 🏸</p>
                            <p className="text-xs text-amber-200/80">We're popular! Please try again in ~{capacityError}s. Your clip is worth the wait.</p>
                        </div>
                        <button onClick={() => setCapacityError(null)} className="ml-auto text-amber-400 hover:text-amber-200">
                            <Icon name="close" size={16} />
                        </button>
                    </div>
                )}

                <header className="flex items-center gap-2 mb-8">
                    {true && (
                        <button
                            onClick={() => navigate('/')}
                            className="mr-1 -ml-1 p-1 rounded text-neutral-600 hover:text-neutral-300 transition-colors"
                            aria-label="Back to home"
                        >
                            <Icon name="arrow_back" size={18} />
                        </button>
                    )}
                    <Logo size={24} className="text-emerald-500" />
                    <h1 className="text-xl font-semibold tracking-tight">
                        Iso<span className="text-emerald-500">Court</span>
                    </h1>
                </header>

                {/* Upload / Record */}
                <section className="bg-neutral-900 border border-neutral-800 rounded-lg p-6 mb-6">

                    {/* Tab switcher */}
                    <div className="flex items-center gap-1 mb-4 p-1 bg-neutral-800/60 rounded-lg w-fit">
                        <button
                            onClick={() => switchMode('upload')}
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 ${inputMode === 'upload'
                                ? 'bg-neutral-700 text-neutral-100 shadow-sm'
                                : 'text-neutral-500 hover:text-neutral-300'
                                }`}
                        >
                            <Icon name="upload" size={14} />
                            Upload
                        </button>
                        <button
                            onClick={() => switchMode('record')}
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-200 ${inputMode === 'record'
                                ? 'bg-neutral-700 text-neutral-100 shadow-sm'
                                : 'text-neutral-500 hover:text-neutral-300'
                                }`}
                        >
                            <Icon name="videocam" size={14} />
                            Record
                        </button>
                    </div>

                    {inputMode === 'upload' ? (
                        /* ── Upload mode ─── */
                        <div
                            {...getRootProps()}
                            className={`
                                border border-dashed rounded-md min-h-[220px] flex flex-col items-center justify-center cursor-pointer transition-colors
                                ${isDragActive
                                    ? 'border-emerald-500 bg-emerald-500/5'
                                    : 'border-neutral-700 hover:border-neutral-500'
                                }
                            `}
                        >
                            <input {...getInputProps()} />
                            {preview ? (
                                <div className="w-full rounded-md overflow-hidden bg-black">
                                    <video
                                        ref={videoRef}
                                        src={preview}
                                        className="w-full max-h-[300px] object-contain bg-black"
                                        controls
                                    />
                                    <div className="flex items-center justify-between px-3 py-2 bg-neutral-900 border-t border-neutral-800/50">
                                        <span className="text-xs text-neutral-400 flex items-center gap-1.5">
                                            <Icon name="check_circle" size={13} className="text-emerald-500" />
                                            Clip ready to analyze
                                        </span>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null) }}
                                            className="text-xs text-neutral-500 hover:text-neutral-300 flex items-center gap-1 transition-colors"
                                        >
                                            <Icon name="replay" size={13} /> Change Video
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-center text-neutral-500 p-6">
                                    <Icon name="video_file" size={36} className="block mx-auto mb-3 text-neutral-600" />
                                    <p className="text-sm font-medium text-neutral-300">Drag &amp; drop video here</p>
                                    <p className="text-xs mt-1 text-neutral-500">or click to select file</p>
                                </div>
                            )}
                        </div>
                    ) : (
                        /* ── Record mode ─── */
                        <div className="rounded-md overflow-hidden border border-neutral-800 min-h-[220px] flex flex-col bg-black relative">
                            {isMobile ? (
                                /* Native Mobile Camera Button View */
                                preview ? (
                                    /* Recorded clip preview */
                                    <div className="w-full bg-black">
                                        <video
                                            ref={videoRef}
                                            src={preview}
                                            className="w-full max-h-[300px] object-contain bg-black"
                                            controls
                                        />
                                        <div className="flex items-center justify-between px-3 py-2 bg-neutral-900/80 border-t border-neutral-800/50">
                                            <span className="text-xs text-neutral-400 flex items-center gap-1.5">
                                                <Icon name="check_circle" size={13} className="text-emerald-500" />
                                                Clip ready
                                            </span>
                                            <button
                                                onClick={() => { setFile(null); setPreview(null); nativeVideoInputRef.current?.click() }}
                                                className="text-xs text-neutral-500 hover:text-neutral-300 flex items-center gap-1 transition-colors"
                                            >
                                                <Icon name="replay" size={13} /> Re-record
                                            </button>
                                        </div>
                                        {/* Hidden Native Input */}
                                        <input
                                            type="file"
                                            accept="video/*"
                                            capture="environment"
                                            ref={nativeVideoInputRef}
                                            onChange={handleNativeVideoSelect}
                                            className="hidden"
                                        />
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center flex-1 gap-4 p-8 bg-neutral-900/50 text-center min-h-[220px]">
                                        <div className="text-neutral-400">
                                            <Icon name="photo_camera" size={48} className="opacity-50 mb-2 block mx-auto" />
                                            <p className="text-sm font-medium text-neutral-300 mb-1">Record on Device</p>
                                            <p className="text-xs text-neutral-500 mb-4 max-w-[200px]">Use your device's native camera for the best quality and zoom.</p>
                                        </div>

                                        <button
                                            onClick={() => nativeVideoInputRef.current?.click()}
                                            className="flex items-center justify-center gap-2 w-full py-3 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border border-neutral-700 rounded-md transition-colors shadow-sm"
                                        >
                                            <Icon name="videocam" size={18} />
                                            <span className="text-sm font-medium">Open Camera</span>
                                        </button>

                                        {/* Hidden Native Input */}
                                        <input
                                            type="file"
                                            accept="video/*"
                                            capture="environment"
                                            ref={nativeVideoInputRef}
                                            onChange={handleNativeVideoSelect}
                                            className="hidden"
                                        />
                                    </div>
                                )
                            ) : (
                                /* Custom Desktop Web Camera View */
                                cameraError ? (
                                    /* Permission / device error */
                                    <div className="flex flex-col items-center justify-center flex-1 gap-3 p-6 text-center min-h-[220px]">
                                        <Icon name="videocam_off" size={36} className="text-neutral-600" />
                                        <p className="text-sm text-neutral-400">{cameraError}</p>
                                        <button
                                            onClick={openCamera}
                                            className="mt-1 px-4 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-white text-xs rounded-md transition-colors"
                                        >
                                            Try again
                                        </button>
                                    </div>
                                ) : preview ? (
                                    /* Recorded clip preview */
                                    <div className="w-full">
                                        <video
                                            ref={videoRef}
                                            src={preview}
                                            className="w-full max-h-[300px] object-contain bg-black"
                                            controls
                                        />
                                        <div className="flex items-center justify-between px-3 py-2 bg-neutral-900/80">
                                            <span className="text-xs text-neutral-400 flex items-center gap-1.5">
                                                <Icon name="check_circle" size={13} className="text-emerald-500" />
                                                Clip ready to analyze
                                            </span>
                                            <button
                                                onClick={() => { setFile(null); setPreview(null); openCamera() }}
                                                className="text-xs text-neutral-500 hover:text-neutral-300 flex items-center gap-1 transition-colors"
                                            >
                                                <Icon name="replay" size={13} /> Re-record
                                            </button>
                                        </div>
                                    </div>
                                ) : (
                                    /* Live camera preview */
                                    <div className={`relative ${isFullScreen ? 'fixed inset-0 z-50 bg-black flex flex-col justify-center' : 'flex-1'}`}>

                                        {isFullScreen && (
                                            <button
                                                onClick={toggleFullScreen}
                                                className="absolute top-6 left-6 z-[60] p-2 bg-black/50 hover:bg-black/70 rounded-full text-white backdrop-blur-md"
                                            >
                                                <Icon name="close" size={24} />
                                            </button>
                                        )}

                                        <video
                                            ref={cameraPreviewRef}
                                            autoPlay
                                            muted
                                            playsInline
                                            onClick={toggleFullScreen}
                                            className={`w-full object-contain bg-black cursor-pointer ${isFullScreen ? 'h-[100dvh]' : 'max-h-[300px] min-h-[180px]'}`}
                                        />

                                        {/* Fullscreen hint when not recording and not fullscreen */}
                                        {!isRecording && !isFullScreen && cameraStreamRef.current?.active && (
                                            <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-black/60 text-white/70 text-[10px] px-2 py-1 rounded-md backdrop-blur-sm pointer-events-none">
                                                <Icon name="fullscreen" size={14} /> Tap to expand
                                            </div>
                                        )}

                                        {/* Recording timer badge */}
                                        {isRecording && (
                                            <div className={`absolute left-1/2 -translate-x-1/2 flex items-center gap-1.5 bg-black/70 border border-rose-500/50 text-rose-400 font-mono px-3 py-1 rounded-full backdrop-blur-sm ${isFullScreen ? 'top-12 text-sm px-4 py-1.5' : 'top-3 text-xs px-2.5 py-1'}`}>
                                                <span className="w-2.5 h-2.5 bg-rose-500 rounded-full animate-pulse" />
                                                {formatSeconds(recordingSeconds)}
                                            </div>
                                        )}

                                        {/* Fullscreen recording controls */}
                                        {isFullScreen && (
                                            <div className="absolute bottom-12 left-0 right-0 flex justify-center pb-safe">
                                                {isRecording ? (
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); stopRecording() }}
                                                        className="flex items-center justify-center w-20 h-20 bg-rose-600/90 text-white rounded-full transition-transform active:scale-90 border-4 border-white/20"
                                                    >
                                                        <span className="w-6 h-6 bg-white rounded-sm" />
                                                    </button>
                                                ) : (
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); startRecording() }}
                                                        className="flex items-center justify-center w-20 h-20 bg-rose-600/90 text-white rounded-full transition-transform active:scale-90 border-4 border-white/20"
                                                    >
                                                        <span className="w-6 h-6 bg-white rounded-full" />
                                                    </button>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                )
                            )}

                            {/* Standard Record / Stop button for Desktop (hidden if full screen) */}
                            {!isMobile && !cameraError && !preview && !isFullScreen && (
                                <div className="flex justify-center py-3 bg-neutral-900/90 border-t border-neutral-800">
                                    {isRecording ? (
                                        <button
                                            onClick={stopRecording}
                                            className="flex items-center gap-2 px-5 py-2 bg-rose-600 hover:bg-rose-700 text-white text-sm font-medium rounded-full transition-colors shadow-lg"
                                        >
                                            <span className="w-2.5 h-2.5 bg-white rounded-sm" />
                                            Stop
                                        </button>
                                    ) : (
                                        <button
                                            onClick={startRecording}
                                            className="flex items-center gap-2 px-5 py-2 bg-rose-600 hover:bg-rose-700 text-white text-sm font-medium rounded-full transition-colors shadow-lg"
                                        >
                                            <span className="w-2.5 h-2.5 bg-white rounded-full" />
                                            Record
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    <button
                        onClick={handleStreamAnalysis}
                        disabled={!file || loading}
                        className="w-full mt-4 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium py-2.5 rounded-md transition-colors"
                    >
                        {loadingStep >= 0 ? (
                            <span className="flex items-center justify-center gap-2">
                                <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                Analyzing...
                            </span>
                        ) : 'Analyze Stroke'}
                    </button>
                </section>

                {/* Results */}
                <section className="bg-neutral-900 border border-neutral-800 rounded-lg p-6">
                    <h2 className="text-sm font-medium text-neutral-400 mb-4 flex items-center gap-2">
                        <Icon name="analytics" size={18} />
                        Analysis Results
                    </h2>

                    {loadingStep >= 0 ? (
                        <div className="py-12 px-4">
                            <div className="space-y-4">
                                {loadingSteps.map((step, idx) => {
                                    const isActive = idx === loadingStep
                                    const isDone = idx < loadingStep
                                    return (
                                        <div
                                            key={idx}
                                            className={`flex items-center gap-3 py-2 px-3 rounded-md transition-all duration-300 ${isActive ? 'bg-neutral-800' : ''
                                                }`}
                                        >
                                            <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-300 ${isDone ? 'bg-emerald-600' : isActive ? 'bg-neutral-700' : 'bg-neutral-800'
                                                }`}>
                                                {isDone ? (
                                                    <Icon name="check" size={14} className="text-white" />
                                                ) : isActive ? (
                                                    <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                                                ) : (
                                                    <span className="w-1.5 h-1.5 bg-neutral-600 rounded-full" />
                                                )}
                                            </div>
                                            <Icon
                                                name={step.icon}
                                                size={18}
                                                className={`transition-colors duration-300 ${isDone ? 'text-emerald-500' : isActive ? 'text-neutral-200' : 'text-neutral-600'
                                                    }`}
                                            />
                                            <span className={`text-sm transition-colors duration-300 ${isDone ? 'text-neutral-400' : isActive ? 'text-neutral-200 font-medium' : 'text-neutral-600'
                                                }`}>
                                                {step.label}{isActive ? '...' : ''}
                                            </span>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    ) : result?.validation_error ? (
                        <div className="py-8 px-4">
                            <div className={`border rounded-lg p-6 ${result.over_duration_limit
                                ? 'bg-amber-950/30 border-amber-800/50'
                                : 'bg-red-950/30 border-red-900/50'
                                }`}>
                                <div className="flex items-start gap-4">
                                    <div className="flex-shrink-0">
                                        <Icon
                                            name={result.over_duration_limit ? 'schedule' : 'error'}
                                            size={32}
                                            className={result.over_duration_limit ? 'text-amber-400' : 'text-red-500'}
                                        />
                                    </div>
                                    <div className="flex-1">
                                        <h3 className={`text-lg font-semibold mb-2 ${result.over_duration_limit ? 'text-amber-300' : 'text-red-400'
                                            }`}>
                                            {result.over_duration_limit ? 'Video Too Long' : 'Not a Badminton Video'}
                                        </h3>
                                        <p className="text-sm text-neutral-300 mb-4">
                                            {result.error_message}
                                        </p>

                                        {result.over_duration_limit && (
                                            <div className="mt-2 mb-4 p-3 bg-amber-900/20 border border-amber-700/30 rounded-lg">
                                                <p className="text-xs text-amber-200/80 leading-relaxed">
                                                    <span className="font-semibold text-amber-300">💡 Tip:</span> For full-game analysis, upload each rally or quarter separately. This keeps analysis times fast and results more accurate for each play.
                                                </p>
                                            </div>
                                        )}

                                        {!result.over_duration_limit && result.validation_details && (
                                            <div className="mt-4 p-3 bg-neutral-950/50 rounded border border-neutral-800">
                                                <span className="text-xs text-neutral-500 block mb-2">Detection Details</span>
                                                <div className="space-y-1.5 text-xs">
                                                    {result.validation_details.pose_confidence !== undefined && (
                                                        <div className="flex justify-between">
                                                            <span className="text-neutral-400">Pose Detection Score:</span>
                                                            <span className={result.validation_details.pose_confidence > 0.3 ? 'text-green-400' : 'text-red-400'}>
                                                                {(result.validation_details.pose_confidence * 100).toFixed(1)}%
                                                            </span>
                                                        </div>
                                                    )}
                                                    {result.validation_details.model_confidence !== undefined && (
                                                        <div className="flex justify-between">
                                                            <span className="text-neutral-400">Model Confidence:</span>
                                                            <span className={result.validation_details.model_confidence > 0.5 ? 'text-green-400' : 'text-red-400'}>
                                                                {(result.validation_details.model_confidence * 100).toFixed(1)}%
                                                            </span>
                                                        </div>
                                                    )}
                                                    {result.validation_details.overhead_score !== undefined && (
                                                        <div className="flex justify-between">
                                                            <span className="text-neutral-400">Overhead Motion:</span>
                                                            <span className={result.validation_details.overhead_score > 0.3 ? 'text-green-400' : 'text-red-400'}>
                                                                {(result.validation_details.overhead_score * 100).toFixed(1)}%
                                                            </span>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        )}

                                        <button
                                            onClick={() => { setResult(null); setFile(null); setPreview(null); }}
                                            className="mt-4 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-sm rounded transition-colors"
                                        >
                                            Try Another Video
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ) : result ? (
                        <div className="space-y-4">
                            {/* Cache hit badge */}
                            {result.cache_hit && (
                                <div className="flex items-center gap-2 text-[11px] text-emerald-400/70 font-medium px-1">
                                    <Icon name="bolt" size={13} className="text-emerald-500" />
                                    Instant result — same clip analyzed before
                                </div>
                            )}
                            <div className="space-y-3">
                                {/* Performance & Tactical Analysis */}
                                <div className="p-4 bg-neutral-950 rounded-md border border-neutral-800">
                                    <div className="flex justify-between items-start mb-4">
                                        <div>
                                            <span className="text-xs text-neutral-500 block mb-1">Execution Quality</span>
                                            <div className={`text-xl font-bold ${getQualityColor(result.quality)}`}>
                                                {result.quality}
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <span className="text-xs font-mono text-neutral-400 block mb-1">Score</span>
                                            <div className="text-lg font-semibold text-white">{result.quality_numeric || 0} / 10</div>
                                        </div>
                                    </div>

                                    <div className="w-full bg-neutral-800 h-1.5 rounded-full mb-6 overflow-hidden">
                                        <motion.div
                                            initial={{ width: shouldReduceMotion ? `${((result.quality_numeric || 0) / 10) * 100}%` : 0 }}
                                            animate={{ width: `${((result.quality_numeric || 0) / 10) * 100}%` }}
                                            transition={{ duration: shouldReduceMotion ? 0 : 0.8, ease: "easeOut" }}
                                            className={`h-full rounded-full ${getQualityBarColor(result.quality)}`}
                                        />
                                    </div>

                                    {result.tactical_analysis && (
                                        <div className="pt-4 border-t border-neutral-800/50">
                                            <span className="text-xs text-neutral-500 block mb-3">Tactical Metrics</span>
                                            <div className="flex flex-wrap gap-2">
                                                <div className="px-2 py-1 bg-blue-500/10 border border-blue-500/30 rounded text-[10px] font-medium text-blue-400 flex items-center gap-1.5">
                                                    <Icon name="pan_tool_alt" size={12} />
                                                    {result.tactical_analysis.technique?.label || 'Unknown'}
                                                    {result.tactical_analysis.technique?.confidence > 0 && (
                                                        <span className="text-[9px] opacity-60">{(result.tactical_analysis.technique.confidence * 100).toFixed(0)}%</span>
                                                    )}
                                                </div>
                                                <div className="px-2 py-1 bg-purple-500/10 border border-purple-500/30 rounded text-[10px] font-medium text-purple-400 flex items-center gap-1.5">
                                                    <Icon name="explore" size={12} />
                                                    {result.tactical_analysis.placement?.label || 'Unknown'}
                                                    {result.tactical_analysis.placement?.confidence > 0 && (
                                                        <span className="text-[9px] opacity-60">{(result.tactical_analysis.placement.confidence * 100).toFixed(0)}%</span>
                                                    )}
                                                </div>
                                                <div className="px-2 py-1 bg-rose-500/10 border border-purple-500/30 rounded text-[10px] font-medium text-rose-400 flex items-center gap-1.5">
                                                    <Icon name="location_on" size={12} />
                                                    {result.tactical_analysis.position?.label || 'Unknown'}
                                                    {result.tactical_analysis.position?.confidence > 0 && (
                                                        <span className="text-[9px] opacity-60">{(result.tactical_analysis.position.confidence * 100).toFixed(0)}%</span>
                                                    )}
                                                </div>
                                                <div className="px-2 py-1 bg-amber-500/10 border border-amber-500/30 rounded text-[10px] font-medium text-amber-400 flex items-center gap-1.5">
                                                    <Icon name="psychology" size={12} />
                                                    {result.tactical_analysis.intent?.label || 'None'}
                                                    {result.tactical_analysis.intent?.confidence > 0 && (
                                                        <span className="text-[9px] opacity-60">{(result.tactical_analysis.intent.confidence * 100).toFixed(0)}%</span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Coach's Recommendations */}
                            {result.recommendations && result.recommendations.length > 0 && (
                                <div className="p-4 bg-emerald-950/20 rounded-md border border-emerald-900/40">
                                    <span className="text-xs text-emerald-500/80 flex items-center gap-1.5 mb-3 font-medium">
                                        <Icon name="tips_and_updates" size={14} />
                                        Coach's Recommendations
                                    </span>
                                    <ul className="space-y-2">
                                        {result.recommendations.map((tip, idx) => (
                                            <li key={idx} className="text-sm text-neutral-300 flex items-start gap-2">
                                                <span className="text-emerald-500 mt-1">•</span>
                                                {tip}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* Timeline Breakdown */}
                            {result.timeline && (
                                <div className="p-4 bg-neutral-950 rounded-md border border-neutral-800">
                                    <span className="text-xs text-neutral-500 flex items-center gap-1.5 mb-4">
                                        <Icon name="timeline" size={14} />
                                        Play-by-Play Breakdown
                                    </span>

                                    {/* Mobile: vertical text timeline */}
                                    <div className="block md:hidden relative border-l border-neutral-800 ml-2 space-y-6 py-1 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                                        {result.timeline.map((event, idx) => (
                                            <div
                                                key={idx}
                                                onClick={() => handleTimelineClick(event.timestamp)}
                                                className="relative pl-6 py-2 rounded hover:bg-neutral-900 transition-colors cursor-pointer group"
                                            >
                                                <div className={`absolute -left-[5.5px] top-4 w-2.5 h-2.5 rounded-full border-2 ${event.label === 'Other'
                                                    ? 'bg-neutral-950 border-neutral-600'
                                                    : 'bg-neutral-950 border-emerald-500'
                                                    }`} />
                                                <div className="flex flex-col gap-3">
                                                    <div className="flex items-start justify-between">
                                                        <div>
                                                            <span className="text-xs font-mono text-neutral-500 block">{event.timestamp}</span>
                                                            <span className={`text-base font-semibold ${event.label === 'Other' ? 'text-neutral-500' : 'text-neutral-100'}`}>
                                                                {event.label.replace(/_/g, ' ')}
                                                            </span>
                                                            <span className="text-[10px] text-neutral-600 ml-2">{(event.confidence * 100).toFixed(0)}%</span>
                                                        </div>
                                                        {event.pose_image && (
                                                            <div className="w-24 h-16 rounded overflow-hidden bg-black/50 border border-neutral-800 flex-shrink-0">
                                                                <img
                                                                    src={`data:image/jpeg;base64,${event.pose_image}`}
                                                                    alt={event.label}
                                                                    className="w-full h-full object-contain"
                                                                />
                                                            </div>
                                                        )}
                                                    </div>

                                                    {event.metrics && (
                                                        <div className="flex flex-wrap gap-1.5">
                                                            <span className="px-1.5 py-0.5 bg-blue-500/5 text-blue-400/70 border border-blue-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                                                                {event.metrics.technique?.label || event.metrics.technique || 'Unknown'}
                                                            </span>
                                                            <span className="px-1.5 py-0.5 bg-purple-500/5 text-purple-400/70 border border-purple-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                                                                {event.metrics.placement?.label || event.metrics.placement || 'Unknown'}
                                                            </span>
                                                            <span className="px-1.5 py-0.5 bg-rose-500/5 text-rose-400/70 border border-rose-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                                                                {event.metrics.position?.label || event.metrics.position || 'Unknown'}
                                                            </span>
                                                            <span className="px-1.5 py-0.5 bg-amber-500/5 text-amber-400/70 border border-amber-500/20 rounded text-[9px] uppercase tracking-wider font-semibold">
                                                                {event.metrics.intent?.label || event.metrics.intent || 'None'}
                                                            </span>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>

                                    {/* Desktop: horizontal skeleton strip */}
                                    <div className="hidden md:flex gap-4 overflow-x-auto pb-6 pt-2 custom-scrollbar">
                                        {result.timeline.map((event, idx) => (
                                            <div
                                                key={idx}
                                                onClick={() => handleTimelineClick(event.timestamp)}
                                                className="flex-shrink-0 cursor-pointer group w-44"
                                            >
                                                <div className={`w-44 h-32 rounded overflow-hidden bg-black border transition-all duration-300 ${event.pose_image ? 'border-neutral-800 group-hover:border-emerald-600 group-hover:scale-[1.02]' : 'border-neutral-900 flex items-center justify-center'}`}>
                                                    {event.pose_image ? (
                                                        <img
                                                            src={`data:image/jpeg;base64,${event.pose_image}`}
                                                            alt={event.label}
                                                            className="w-full h-full object-contain"
                                                        />
                                                    ) : (
                                                        <Icon name="hide_image" size={24} className="text-neutral-800" />
                                                    )}
                                                </div>
                                                <div className="mt-3 space-y-2 px-1">
                                                    <div className="flex justify-between items-center">
                                                        <span className="text-[10px] font-mono text-neutral-500">{event.timestamp}</span>
                                                        <span className="text-[10px] text-neutral-600">{(event.confidence * 100).toFixed(0)}%</span>
                                                    </div>
                                                    <span className={`text-sm block truncate group-hover:text-emerald-400 transition-colors ${event.label === 'Other' ? 'text-neutral-500' : 'text-neutral-200 font-semibold'}`}>
                                                        {event.label.replace(/_/g, ' ')}
                                                    </span>

                                                    {event.metrics && (
                                                        <div className="grid grid-cols-2 gap-1 mt-2 border-t border-neutral-800/30 pt-2">
                                                            <div className="flex items-center gap-1 text-[8px] text-neutral-500 uppercase font-bold tracking-tighter truncate">
                                                                <Icon name="pan_tool_alt" size={10} className="text-blue-500/50" />
                                                                {event.metrics.technique?.label || event.metrics.technique || '???'}
                                                            </div>
                                                            <div className="flex items-center gap-1 text-[8px] text-neutral-500 uppercase font-bold tracking-tighter truncate">
                                                                <Icon name="explore" size={10} className="text-purple-500/50" />
                                                                {event.metrics.placement?.label || event.metrics.placement || '???'}
                                                            </div>
                                                            <div className="flex items-center gap-1 text-[8px] text-neutral-500 uppercase font-bold tracking-tighter truncate">
                                                                <Icon name="location_on" size={10} className="text-rose-500/50" />
                                                                {event.metrics.position?.label || event.metrics.position || '???'}
                                                            </div>
                                                            <div className="flex items-center gap-1 text-[8px] text-neutral-500 uppercase font-bold tracking-tighter truncate">
                                                                <Icon name="psychology" size={10} className="text-amber-500/50" />
                                                                {event.metrics.intent?.label || event.metrics.intent || '???'}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center py-16 text-neutral-600">
                            <Icon name="pending" size={32} className="mb-3 text-neutral-700" />
                            <p className="text-sm">Upload a clip to get started</p>
                        </div>
                    )}
                </section>
            </div>
        </div>
    )
}
