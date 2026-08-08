const STROKES = [
    'smash',
    'overhead clear',
    'net drop',
    'drive',
    'lift',
    'backhand push',
    'cross-court drop',
    'jump smash',
]

function TickerRow() {
    return (
        <>
            {STROKES.map((label, i) => (
                <span key={label} className="figma-stroke-ticker__label">
                    {label}
                    {i < STROKES.length - 1 ? (
                        <span className="figma-stroke-ticker__sep" aria-hidden>
                            ·
                        </span>
                    ) : null}
                </span>
            ))}
        </>
    )
}

/** CSS-only marquee — no ResizeObserver (keeps mobile TBT down). */
export default function StrokeTicker() {
    return (
        <div className="figma-stroke-ticker figma-stroke-ticker--animate" aria-hidden>
            <div className="figma-stroke-ticker__viewport">
                <div className="figma-stroke-ticker__track" style={{ '--ticker-copies': 3 }}>
                    {['a', 'b', 'c'].map((copyId) => (
                        <div key={copyId} className="figma-stroke-ticker__inner">
                            <TickerRow />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
