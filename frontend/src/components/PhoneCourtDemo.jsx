import { useEffect, useRef, useState } from 'react'

export default function PhoneCourtDemo({ video, poster, frame, label }) {
    const videoRef = useRef(null)
    const [playing, setPlaying] = useState(false)

    useEffect(() => {
        const el = videoRef.current
        if (!el || !playing) return undefined
        el.play().catch(() => {})
        return undefined
    }, [playing])

    return (
        <div className="figma-phone-frame figma-phone-frame--split">
            <div className="figma-phone-screen">
                {playing ? (
                    <video
                        ref={videoRef}
                        src={video}
                        poster={poster}
                        autoPlay
                        loop
                        muted
                        playsInline
                        preload="metadata"
                        aria-label={label}
                    />
                ) : null}
                {!playing ? (
                    <button
                        type="button"
                        className="figma-phone-play"
                        onClick={() => setPlaying(true)}
                        aria-label={`Play demo: ${label}`}
                    >
                        Play demo
                    </button>
                ) : null}
            </div>
            <img src={frame} alt="" className="figma-phone-court-frame figma-phone-mockup" aria-hidden />
        </div>
    )
}
