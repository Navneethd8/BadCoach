import HeroRacketSvg from './HeroRacketSvg'

/** Instant hero art — no framer-motion. Scroll animation upgrades later. */
export default function HeroStaticBackdrop() {
    return (
        <>
            <div className="figma-layer-net" aria-hidden>
                <img src="/net-shenanigans.svg" alt="" width={3648} height={4188} decoding="async" />
            </div>
            <div className="figma-layer-racket-glass" aria-hidden />
            <div className="figma-layer-racket" aria-hidden>
                <div className="figma-layer-racket-rot">
                    <HeroRacketSvg />
                </div>
            </div>
        </>
    )
}
