import { LazyMotion, domAnimation, m, useReducedMotion, useScroll, useTransform } from 'framer-motion'
import HeroRacketSvg from './HeroRacketSvg'

/**
 * Hero backdrop — Figma 216-334 artboard layers (net + racket).
 *
 * Scroll-driven racket choreography:
 *   progress 0.00 — horizontal, head centered around the heading + CTAs
 *                   so the copy lives inside the empty racket face
 *   progress 0.50 — rotated to fully vertical (90°), then locks in place
 *   progress 1.00 — held vertical, faded out
 *
 * `scrollTarget` is a ref to the hero <section>; useScroll measures how far
 * the user has scrolled the hero through the top of the viewport.
 */
export default function HeroFigmaBackdrop({ scrollTarget }) {
    const reduceMotion = useReducedMotion()

    const { scrollYProgress } = useScroll({
        target: scrollTarget,
        offset: ['start start', 'end start'],
    })

    const rotate = useTransform(scrollYProgress, [0, 0.5, 1], [0, 90, 90])
    const opacity = useTransform(scrollYProgress, [0, 0.55, 0.85, 1], [1, 1, 1, 0])

    const racketStyle = reduceMotion ? undefined : { rotate, opacity }

    return (
        <LazyMotion features={domAnimation} strict>
            <div className="figma-layer-net" aria-hidden>
                <img src="/net-shenanigans.svg" alt="" width={3648} height={4188} decoding="async" />
            </div>

            <div className="figma-layer-racket-glass" aria-hidden />

            <div className="figma-layer-racket" aria-hidden>
                <m.div className="figma-layer-racket-rot" style={racketStyle}>
                    <HeroRacketSvg />
                </m.div>
            </div>
        </LazyMotion>
    )
}
