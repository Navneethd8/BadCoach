import { lazy, Suspense, useState } from 'react'

const InteractivePoseFigure = lazy(() => import('./InteractivePoseFigure'))

function StaticPose({ onActivate }) {
    return (
        <figure className="pixel-pose">
            <button
                type="button"
                className="pixel-pose__static-btn"
                onPointerEnter={onActivate}
                onFocus={onActivate}
                aria-label="Interactive pose figure. Activate to hover-trace the skeleton."
            >
                <img
                    src="/marketing/pose-trace-hero.webp"
                    alt=""
                    className="pixel-pose__canvas"
                    width={390}
                    height={640}
                    decoding="async"
                    loading="lazy"
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

    if (!interactive) {
        return <StaticPose onActivate={() => setInteractive(true)} />
    }

    return (
        <Suspense fallback={<StaticPose onActivate={() => {}} />}>
            <InteractivePoseFigure />
        </Suspense>
    )
}
