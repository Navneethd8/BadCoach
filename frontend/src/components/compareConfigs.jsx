/** Comparison page configs. Claims stick to public positioning; keep IsoCourt voice. */

export const badmintonPeakCompare = {
    path: '/compare/badmintonpeak',
    seoTitle: 'IsoCourt vs BadmintonPeak · AI Badminton Analysis',
    seoDescription:
        'Compare IsoCourt and BadmintonPeak for AI badminton video analysis: pose timelines, live camera coaching, courses, pricing, and who each tool fits.',
    heroTitle: (
        <>
            isocourt vs <span className="text-brand">badmintonpeak</span>
        </>
    ),
    lead: 'Both read your badminton video. They aim at different training loops.',
    competitorName: 'BadmintonPeak',
    competitorUrl: 'https://badmintonpeak.com/en',
    updated: 'July 23, 2026',
    intro:
        'BadmintonPeak leans into structured courses and one clear correction per video. IsoCourt is a browser coach: upload a rally or go live, get pose skeletons, stroke labels, scores, and short Birdzo tips tied to timestamps.',
    rows: [
        {
            topic: 'Best for',
            us: 'Club players who want a fast clip read or live camera feedback in the browser',
            them: 'Players following drill courses with step-by-step video correction',
        },
        {
            topic: 'How you train',
            us: 'Drop a rally or open Live. Timeline of strokes + skeleton frames',
            them: 'Film a course drill, get one instruction, validate progress in the course',
        },
        {
            topic: 'Live camera',
            us: 'Yes, in-browser live session',
            them: 'Upload / send video focused (check their app for latest)',
        },
        {
            topic: 'Pose view',
            us: 'Skeleton overlays on analysis frames you can inspect',
            them: 'Frame-by-frame move read with angle-style feedback in their product',
        },
        {
            topic: 'Courses',
            us: 'No curriculum track yet. Cues per clip you bring',
            them: 'Course library (fundamentals, attack, serve paths) with progress levels',
        },
        {
            topic: 'Pricing shape',
            us: 'Try on the web; capacity limits when busy',
            them: 'Free starter plus paid Essential / Basic / Pro minute packs (public pricing on their site)',
        },
        {
            topic: 'Platform',
            us: 'Web (desktop and phone browser)',
            them: 'Product site + analysis workflow (see their site for apps)',
        },
    ],
    pickThem: [
        'You want a curriculum that tells you which drill to film next.',
        'You like progress framed as completed courses and levels.',
        'You want paid minute packs and priority queues laid out as subscriptions.',
    ],
    pickUs: [
        'You already know the shot you want to fix and just need a fast read.',
        'You want live camera coaching without installing a course app first.',
        'You care about stroke timeline + pose frames you can scrub yourself.',
    ],
    faqs: [
        {
            q: 'Is IsoCourt a BadmintonPeak alternative?',
            a: 'Yes, if you want AI badminton video feedback. It is not a clone of their course product. IsoCourt is stronger for ad-hoc rally analysis and live browser sessions.',
        },
        {
            q: 'Which is better for beginners?',
            a: 'If you need a guided course path, BadmintonPeak\'s curriculum is built for that. If you already train with a coach or club and want clip feedback between sessions, IsoCourt fits better.',
        },
        {
            q: 'Does IsoCourt replace a human coach?',
            a: 'No. Neither tool should. IsoCourt tips are automated training insight only.',
        },
        {
            q: 'Can I use both?',
            a: 'Sure. Some players use a course product for structure and IsoCourt when they want a quick pose and stroke read on a messy rally.',
        },
    ],
}

export const kreedaCompare = {
    path: '/compare/kreeda',
    seoTitle: 'IsoCourt vs Kreeda · AI Badminton Coaching',
    seoDescription:
        'Compare IsoCourt and Kreeda for AI badminton coaching: browser live analysis vs phone app metrics, communities, coaches, and who each product fits.',
    heroTitle: (
        <>
            isocourt vs <span className="text-brand">kreeda</span>
        </>
    ),
    lead: 'Phone athletic intelligence vs browser stroke coach. Different jobs.',
    competitorName: 'Kreeda',
    competitorUrl: 'https://kreeda.tech/',
    updated: 'July 23, 2026',
    intro:
        'Kreeda positions as a phone app for match metrics, AI score, community, and optional human coaches. IsoCourt is web-first: analyze a clip or go live, with pose tracing, stroke labels, and Birdzo tips on a timeline.',
    rows: [
        {
            topic: 'Best for',
            us: 'Players who want stroke + pose feedback in the browser, including live',
            them: 'Players who want phone match analytics, scores, and community loops',
        },
        {
            topic: 'Core output',
            us: 'Stroke timeline, skeleton frames, quality score, coaching tips',
            them: 'AI score, shot mix, speed/coverage style metrics, session pulse (per their marketing)',
        },
        {
            topic: 'Live coaching',
            us: 'In-browser live camera session',
            them: 'Record/upload workflow with app-centric experience',
        },
        {
            topic: 'Community / coaches',
            us: 'Feedback form today; no marketplace yet',
            them: 'Community challenges and paths to send reviews to coaches',
        },
        {
            topic: 'Setup',
            us: 'Open the site, allow camera or upload a file',
            them: 'App install / account flow (Google login on their site)',
        },
        {
            topic: 'Doubles / full court',
            us: 'Works best with a clear single-player clip in frame',
            them: 'Markets full-court recording guidance from behind the baseline',
        },
    ],
    pickThem: [
        'You want an app-native dashboard and community rankings.',
        'You care about match-style metrics (speed, coverage, shot mix) first.',
        'You want a path to human coach review inside the same product.',
    ],
    pickUs: [
        'You want pose skeletons and stroke labels you can inspect frame by frame.',
        'You want live browser feedback without installing another sports app.',
        'You bring short rallies or drills and want tips tied to timestamps.',
    ],
    faqs: [
        {
            q: 'Is IsoCourt a Kreeda alternative?',
            a: 'For AI badminton feedback, yes. For community leaderboards and coach marketplace features, Kreeda is built for that today.',
        },
        {
            q: 'Do I need a phone mount for IsoCourt?',
            a: 'Helpful, not required. Keep the shuttle in frame and the camera steady. Live mode needs browser camera permission.',
        },
        {
            q: 'Which is better for doubles?',
            a: 'Full-court doubles tracking is hard for any AI. Kreeda\'s marketing leans into full-court phone setups. IsoCourt is strongest when one player and the shuttle are clear in frame.',
        },
        {
            q: 'Is my video private on IsoCourt?',
            a: 'Clips are processed for analysis. See the Privacy page for the current summary.',
        },
    ],
}
