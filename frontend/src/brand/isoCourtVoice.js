/**
 * IsoCourt voice — one product idea, repeated everywhere (marketing + app chrome).
 * Same pattern as strong product landings: every section ladders back to the spine.
 */

/** Full spine — use in hero, footer, ribbon repeats */
export const spine =
    'IsoCourt reads your rally in hall language—between sessions, beside the tape—not like enterprise video analytics.'

/** Short spine — ribbons, subheads, stats intro */
export const spineShort = 'Hall language, not enterprise dashboards.'

export const clipLab = {
    kicker: 'Clip lab',
    /** One line under the logo — same promise as marketing */
    promise: 'Bring a clip. Leave with a read.',
}

export const nav = {
    analyze: 'Analyze',
    analyzeClip: 'Analyze clip',
    live: 'Live',
    homeAria: 'IsoCourt — home',
    clipLabAria: 'Open clip lab — read your rally',
    liveAria: 'Live coaching between points',
}

export const cta = {
    /** Primary action — same label on landing + clip lab */
    primary: 'Read your clip',
    primaryBusy: 'Reading your rally…',
    secondary: 'Live between points',
    liveFooter: 'Live between points',
}

export const analyzeUi = {
    resultsTitle: 'Your read',
    clipReady: 'Clip ready for a read',
    changeVideo: 'Swap clip',
    dropTitle: 'Drop rally footage here',
    dropHint: 'or click to browse — phone on a tripod is fine',
    loadingSteps: [
        { icon: 'movie_filter', label: 'Pulling frames from your rally' },
        { icon: 'directions_run', label: 'Tracing how you move on court' },
        { icon: 'query_stats', label: 'Calling strokes like a sideline read' },
        { icon: 'rate_review', label: 'Turning it into cues you can use' },
    ],
}

export const landing = {
    heroKicker: 'Badminton · clip reads',
    /** Opening paragraph — must echo spine */
    heroLead:
        'Upload a clip — doubles scrape or singles grind. IsoCourt reads footwork, contact height, and stroke the way you’d talk after rubber: what broke, where the shuttle went, what to try next.',
    sampleReadLabel: 'Sample read',
    statsIntro: 'Numbers worth posting on a clubhouse wall — not a slide.',
    featuresKicker: 'What IsoCourt reads',
    featuresHeadline: 'Split-step timing, contact height — without someone yelling from the next court.',
    featuresLead:
        'Same three beats in every rubber: body, shuttle, plain cue. No quarterly deck; just what you’d hash out over a split crate of shuttles.',
    filmKicker: 'Replay',
    filmHeadline: 'Watch the read on real hall footage.',
    filmLead: 'Same pipeline you get when you upload — shown on tape so you know what “read the rally” looks like.',
    flowKicker: 'Between sessions',
    flowHeadline: 'Clip in. Read out. Back to training.',
    closingHeadline: 'Same hall tomorrow.',
    closingSub: 'Fewer blind spots before you tie on next time.',
    closingLead: spineShort,
    feedbackKicker: 'Sideline mail',
    feedbackHeadline: 'Notes from the hall.',
    feedbackLead:
        'Wrong call, half-baked idea, or “this fixed my clear.” We answer between sessions — not from a ticket queue.',
    footerLead: 'For people who know what a feather costs. ' + spineShort,
    contactLink: 'Sideline mail',
}

/** Timeline — mirrors flowKicker / flowHeadline */
export const flowSteps = [
    {
        n: '01',
        icon: 'upload',
        title: 'Clip in',
        description: 'One smash, a messy rally, or footwork drills. Steady framing — shuttle has to be visible.',
    },
    {
        n: '02',
        icon: 'model_training',
        title: 'We read it',
        description: 'Pose trace and stroke calls compile while you towel off. No labelling frames yourself.',
    },
    {
        n: '03',
        icon: 'emoji_events',
        title: 'Back to the hall',
        description: 'What broke in the rally, plus a few cues before you tie on next time.',
    },
]

export const hallBand = {
    kicker: 'Where IsoCourt lives',
    headline: 'The doubles box you already argue about.',
    aside: 'Feather shuttle, PVC squeak, centre-line serves — not tennis, not pickleball.',
    dlCourt: 'Court',
    dlCourtDd: '13.4 m by 6.1 m — net 1.55 m at the tape.',
    dlService: 'Service',
    dlServiceDd: 'Short line 1.98 m from the net; T split dead centre.',
    dlSingles: 'Singles',
    dlSinglesDd: 'Tramlines in — the alley’s gone when you walk in for singles.',
}
