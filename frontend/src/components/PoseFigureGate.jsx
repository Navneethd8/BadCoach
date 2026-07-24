import { lazy, Suspense, useCallback, useRef, useState } from 'react'

const InteractivePoseFigure = lazy(() => import('./InteractivePoseFigure'))

/** PNG is deployed everywhere; webp is optional once shipped. */
const SOURCE = '/marketing/pose-trace-hero.png'

function StaticPose({ onPointer }) {
    return (
        <figure className="pixel-pose">
            <button
                type="button"
                className="pixel-pose__static-btn"
                onPointerEnter={onPointer}
                onPointerMove={onPointer}
                onFocus={onPointer}
                aria-label="Interactive pose figure. Activate to hover-trace the skeleton."
            >
                <img
                    src={SOURCE}
                    alt=""
                    className="pixel-pose__canvas"
                    width={390}
                    height={640}
                    decoding="async"
                    loading="eager"
                    fetchPriority="low"
                />
            </button>
            <figcaption className="pixel-pose__hint" aria-hidden>
                <span className="pixel-pose__hint-dot" />
                hover to trace
            </figcaption>
        </figure>
    )
}

/**
 * Static pose first; interactive canvas only after hover/focus.
 * Keeps getImageData work off the initial load path.
 */
export default function PoseFigureGate() {
    const [interactive, setInteractive] = useState(false)
    const pointerRef = useRef(null)

    const onPointer = useCallback((event) => {
        if (event?.clientX != null) {
            pointerRef.current = { clientX: event.clientX, clientY: event.clientY }
        }
        setInteractive(true)
    }, [])

    if (!interactive) {
        return <StaticPose onPointer={onPointer} />
    }

    return (
        <Suspense fallback={<StaticPose onPointer={onPointer} />}>
            <InteractivePoseFigure pointerSeedRef={pointerRef} />
        </Suspense>
    )
}
