/**
 * Full-bleed indoor hall strip — green mat + white tape reads as badminton instantly.
 * Decorative labels cite real markings (doubles box, service lines).
 */

import { hallBand } from '../brand/isoCourtVoice.js'

function DoublesCourtFloorSvg({ className = '' }) {
    /* Proportional to 13.4 m × 6.1 m; net at centre; short service 1.98 m from net */
    return (
        <svg
            className={className}
            viewBox="0 0 610 134"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            aria-hidden
        >
            <g stroke="currentColor" strokeWidth="2" strokeLinejoin="miter" opacity="0.95">
                {/* Doubles court */}
                <rect x="4" y="4" width="602" height="126" />
                {/* Net */}
                <line x1="305" y1="4" x2="305" y2="130" strokeWidth="2.5" />
                {/* Short service lines */}
                <line x1="215" y1="4" x2="215" y2="130" opacity="0.88" />
                <line x1="395" y1="4" x2="395" y2="130" opacity="0.88" />
                {/* Centre service line */}
                <line x1="305" y1="46" x2="305" y2="88" opacity="0.85" />
                {/* Singles sidelines (tramlines) */}
                <line x1="52" y1="4" x2="52" y2="130" strokeDasharray="5 6" opacity="0.55" />
                <line x1="558" y1="4" x2="558" y2="130" strokeDasharray="5 6" opacity="0.55" />
                {/* Long service lines (rear court singles) */}
                <line x1="52" y1="27" x2="558" y2="27" strokeDasharray="4 7" opacity="0.38" />
                <line x1="52" y1="107" x2="558" y2="107" strokeDasharray="4 7" opacity="0.38" />
            </g>
        </svg>
    )
}

export default function HallCourtBand() {
    return (
        <section
            className="relative overflow-hidden border-y border-white/10 bg-court-mat text-court-on"
            aria-labelledby="hall-court-heading"
        >
            <div
                className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_85%_70%_at_50%_35%,rgba(255,255,255,0.07),transparent_55%),linear-gradient(180deg,rgba(0,0,0,0.12),transparent_40%,rgba(0,0,0,0.18))]"
                aria-hidden
            />
            {/* PVC-style horizon sheen */}
            <div
                className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-white/[0.06] to-transparent"
                aria-hidden
            />

            <div className="relative mx-auto max-w-6xl px-5 py-12 md:py-16">
                <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                    <div>
                        <p className="font-mono text-[10px] uppercase tracking-[0.38em] text-white/50">{hallBand.kicker}</p>
                        <h2 id="hall-court-heading" className="font-display mt-3 max-w-xl text-2xl leading-snug text-white md:text-[1.85rem]">
                            {hallBand.headline}
                        </h2>
                    </div>
                    <p className="max-w-xs font-mono text-[11px] leading-relaxed text-white/45 md:text-right">{hallBand.aside}</p>
                </div>

                <div className="mt-10 text-court-line drop-shadow-[0_1px_0_rgba(0,0,0,0.25)]">
                    <DoublesCourtFloorSvg className="h-auto w-full max-h-[min(28vw,200px)] md:max-h-[220px]" />
                </div>

                <dl className="mt-8 grid gap-6 border-t border-white/15 pt-8 font-mono text-[10px] uppercase tracking-[0.22em] text-white/40 sm:grid-cols-3">
                    <div>
                        <dt className="text-white/55">{hallBand.dlCourt}</dt>
                        <dd className="mt-2 font-sans normal-case tracking-normal text-[13px] text-white/75">{hallBand.dlCourtDd}</dd>
                    </div>
                    <div>
                        <dt className="text-white/55">{hallBand.dlService}</dt>
                        <dd className="mt-2 font-sans normal-case tracking-normal text-[13px] text-white/75">{hallBand.dlServiceDd}</dd>
                    </div>
                    <div>
                        <dt className="text-white/55">{hallBand.dlSingles}</dt>
                        <dd className="mt-2 font-sans normal-case tracking-normal text-[13px] text-white/75">{hallBand.dlSinglesDd}</dd>
                    </div>
                </dl>
            </div>
        </section>
    )
}
