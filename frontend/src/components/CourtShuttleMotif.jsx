import { motion, useReducedMotion } from 'framer-motion'

/**
 * Decorative court lines + shuttlecock motifs for the landing hero.
 * Smash loop + drift respect prefers-reduced-motion.
 */

function TrajectoryArc({ className = '', steep = false }) {
    const d = steep
        ? 'M 28 12 Q 160 118 388 96'
        : 'M 20 95 Q 120 8 360 78'
    return (
        <svg className={className} viewBox="0 0 400 120" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
            <path
                className="shuttle-trajectory-stroke"
                d={d}
                stroke="currentColor"
                strokeWidth={steep ? '1.35' : '1.25'}
                strokeLinecap="round"
                strokeDasharray={steep ? '5 12' : '6 10'}
                opacity={steep ? 0.55 : 0.45}
            />
        </svg>
    )
}

/** Top-down doubles court slice */
function CourtFloorSvg({ className = '' }) {
    return (
        <svg
            className={className}
            viewBox="0 0 440 220"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden
        >
            <g stroke="currentColor" strokeWidth="1.15" strokeLinecap="square" opacity="0.9">
                <rect x="14" y="18" width="412" height="184" />
                <line x1="220" y1="18" x2="220" y2="202" strokeWidth="1.35" />
                <line x1="14" y1="78" x2="426" y2="78" opacity="0.75" />
                <line x1="14" y1="142" x2="426" y2="142" opacity="0.75" />
                <line x1="62" y1="18" x2="62" y2="202" opacity="0.55" strokeDasharray="4 5" />
                <line x1="378" y1="18" x2="378" y2="202" opacity="0.55" strokeDasharray="4 5" />
                <line x1="62" y1="46" x2="378" y2="46" opacity="0.4" strokeDasharray="3 6" />
                <line x1="62" y1="174" x2="378" y2="174" opacity="0.4" strokeDasharray="3 6" />
                <line x1="220" y1="78" x2="220" y2="142" opacity="0.65" />
            </g>
        </svg>
    )
}

/** Feather shuttle — skirt reads like cork + feathers, not a random triangle */
function ShuttleGlyph({ className = '' }) {
    return (
        <svg
            className={className}
            viewBox="0 0 48 64"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden
        >
            <ellipse cx="24" cy="50" rx="8.5" ry="11" className="fill-white dark:fill-white" opacity="0.96" />
            <ellipse cx="24" cy="50" rx="6" ry="8" fill="rgba(234,88,12,0.45)" />
            <g stroke="rgba(120,113,108,0.35)" strokeWidth="0.5">
                {[0, 45, 90, 135, 180, 225, 270, 315].map((deg, i) => (
                    <line
                        key={i}
                        x1="24"
                        y1="18"
                        x2={24 + 14 * Math.cos((deg * Math.PI) / 180)}
                        y2={18 + 14 * Math.sin((deg * Math.PI) / 180)}
                    />
                ))}
            </g>
            <g fill="rgba(255,255,255,0.92)" stroke="rgba(255,255,255,0.35)" strokeWidth="0.4">
                {[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map((deg, i) => {
                    const rad = (deg * Math.PI) / 180
                    const x = 24 + 11 * Math.cos(rad)
                    const y = 18 + 11 * Math.sin(rad)
                    return <circle key={i} cx={x} cy={y} r="2.2" />
                })}
            </g>
            <ellipse cx="24" cy="18" rx="5" ry="4" fill="rgba(255,255,255,0.85)" />
        </svg>
    )
}

const shuttleEase = [0.22, 1, 0.36, 1]

export function HeroCourtShuttleLayer({ className = '' }) {
    const reduce = useReducedMotion()

    return (
        <div className={`pointer-events-none absolute inset-0 overflow-hidden ${className}`} aria-hidden>
            <div className="absolute -right-4 top-[8%] w-[min(92vw,520px)] text-orange-100/22 md:right-[6%] md:top-[12%] md:w-[480px]">
                <CourtFloorSvg className="h-auto w-full" />
            </div>
            <div className="absolute -left-8 bottom-[18%] hidden w-[min(85vw,380px)] rotate-[8deg] text-teal-200/14 sm:block md:left-[4%] md:bottom-[22%]">
                <CourtFloorSvg className="h-auto w-full scale-x-[-1]" />
            </div>

            <TrajectoryArc className="absolute left-[4%] top-[22%] h-24 w-[min(70vw,360px)] text-orange-200/45 md:left-[8%]" />
            <TrajectoryArc className="absolute bottom-[28%] right-[2%] h-20 w-[min(65vw,320px)] rotate-12 scale-x-[-1] text-teal-200/30 md:right-[6%]" />
            <TrajectoryArc
                steep
                className="absolute right-[6%] top-[14%] h-28 w-[min(72vw,380px)] text-orange-100/55 md:right-[10%]"
            />

            {/* Rear-court attack — steep arc */}
            <div className="absolute right-[8%] top-[18%] h-14 w-11 md:right-[12%] md:top-[16%]">
                {!reduce && (
                    <div className="relative h-full w-full">
                        <ShuttleGlyph className="shuttle-smash-sprite h-full w-full drop-shadow-[0_3px_14px_rgba(0,0,0,0.55)]" />
                    </div>
                )}
                {reduce && (
                    <ShuttleGlyph className="h-full w-full opacity-90 drop-shadow-[0_3px_14px_rgba(0,0,0,0.55)]" />
                )}
            </div>

            <motion.div
                className="absolute left-[10%] top-[30%] h-12 w-10 md:left-[14%] md:top-[28%]"
                animate={
                    reduce
                        ? undefined
                        : {
                              y: [0, -14, 4, 0],
                              x: [0, 10, -4, 0],
                              rotate: [-6, 4, -2, -6],
                          }
                }
                transition={reduce ? undefined : { duration: 7.5, repeat: Infinity, ease: shuttleEase }}
            >
                <ShuttleGlyph className="h-full w-full drop-shadow-[0_2px_8px_rgba(0,0,0,0.45)]" />
            </motion.div>
            <motion.div
                className="absolute right-[16%] top-[38%] h-10 w-8 md:right-[22%]"
                animate={
                    reduce
                        ? undefined
                        : {
                              y: [0, 12, -6, 0],
                              x: [0, -16, 8, 0],
                              rotate: [8, -10, 6, 8],
                          }
                }
                transition={reduce ? undefined : { duration: 8.8, repeat: Infinity, ease: shuttleEase, delay: 0.8 }}
            >
                <ShuttleGlyph className="h-full w-full drop-shadow-[0_2px_8px_rgba(0,0,0,0.5)]" />
            </motion.div>
            <motion.div
                className="absolute bottom-[32%] left-[22%] h-9 w-7 md:bottom-[36%]"
                animate={
                    reduce
                        ? undefined
                        : {
                              y: [0, -10, 8, 0],
                              x: [0, 22, -6, 0],
                              rotate: [-4, 12, -8, -4],
                          }
                }
                transition={reduce ? undefined : { duration: 9.5, repeat: Infinity, ease: shuttleEase, delay: 1.6 }}
            >
                <ShuttleGlyph className="h-full w-full opacity-90 drop-shadow-[0_2px_8px_rgba(0,0,0,0.4)]" />
            </motion.div>
        </div>
    )
}

export function ShuttleInline({ className = '' }) {
    return <ShuttleGlyph className={`inline-block h-[1.1rem] w-[0.85rem] align-[-3px] opacity-90 ${className}`} />
}
