/**
 * Hero background layers positioned on a 1512×870px Figma artboard (node 216-334).
 * Scales uniformly to viewport width so net + racket match the design file.
 */
export default function HeroFigmaBackdrop() {
    return (
        <div className="figma-hero-scale-wrap" aria-hidden>
            <div className="figma-hero-artboard">
                <div className="figma-layer-net">
                    <img src="/net-shenanigans.svg" alt="" className="figma-layer-net-img" />
                </div>

                <div className="figma-layer-racket-wrap">
                    <div className="figma-layer-racket-rot">
                        <div className="figma-layer-racket-v2">
                            <div className="figma-racket-shaft-slot">
                                <img src="/hero-racket-shaft.svg" alt="" className="figma-racket-part" />
                            </div>
                            <div className="figma-racket-head-slot">
                                <img src="/hero-racket-head.svg" alt="" className="figma-racket-part" />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
