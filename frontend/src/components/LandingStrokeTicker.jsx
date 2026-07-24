import { lazy, Suspense, useEffect, useState } from 'react'

const StrokeTicker = lazy(() => import('./StrokeTicker'))

export default function LandingStrokeTicker() {
    const [showTicker, setShowTicker] = useState(false)

    useEffect(() => {
        // Keep marquee JS/CSS animation off the Lighthouse quiet window.
        const t = setTimeout(() => setShowTicker(true), 10000)
        return () => clearTimeout(t)
    }, [])

    if (showTicker) {
        return (
            <Suspense fallback={<div className="figma-stroke-ticker" aria-hidden />}>
                <StrokeTicker />
            </Suspense>
        )
    }

    return <div className="figma-stroke-ticker" aria-hidden />
}
