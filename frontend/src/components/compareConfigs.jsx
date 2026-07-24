/** Single alternatives comparison page. Claims stick to public positioning. */

export const alternativesCompare = {
    path: '/compare',
    seoTitle: 'IsoCourt vs BadmintonPeak vs Kreeda · AI Badminton Tools',
    seoDescription:
        'Compare IsoCourt, BadmintonPeak, and Kreeda for AI badminton coaching: browser pose analysis and live camera vs courses vs phone match metrics.',
    heroTitle: (
        <>
            isocourt vs <span className="text-brand">alternatives</span>
        </>
    ),
    lead: 'Three AI badminton tools. Different jobs: clip coach, course path, phone match app.',
    competitors: [
        { name: 'BadmintonPeak', url: 'https://badmintonpeak.com/en' },
        { name: 'Kreeda', url: 'https://kreeda.tech/' },
    ],
    updated: 'July 23, 2026',
    intro:
        'BadmintonPeak leans into structured courses and one clear correction per video. Kreeda positions as a phone app for match metrics, AI score, community, and optional human coaches. IsoCourt is web-first: analyze a clip or go live, with pose tracing, stroke labels, and Birdzo tips on a timeline.',
    rows: [
        {
            topic: 'Best for',
            us: 'Club players who want a fast clip read or live camera feedback in the browser',
            peak: 'Players following drill courses with step-by-step video correction',
            kreeda: 'Players who want phone match analytics, scores, and community loops',
        },
        {
            topic: 'Core output',
            us: 'Stroke timeline, skeleton frames, quality score, coaching tips',
            peak: 'One instruction per filmed course drill; progress in the course',
            kreeda: 'AI score, shot mix, speed/coverage-style metrics (per their marketing)',
        },
        {
            topic: 'Live camera',
            us: 'Yes — in-browser live session',
            peak: 'Upload / send video focused (check their app for latest)',
            kreeda: 'Record/upload workflow; app-centric experience',
        },
        {
            topic: 'Pose / form view',
            us: 'Skeleton overlays on analysis frames you can inspect',
            peak: 'Frame-by-frame move read with angle-style feedback',
            kreeda: 'Match metrics first; less pose-timeline focused in public positioning',
        },
        {
            topic: 'Courses / community',
            us: 'No curriculum track yet. Cues per clip you bring',
            peak: 'Course library with progress levels',
            kreeda: 'Community challenges and paths to coach review',
        },
        {
            topic: 'Setup',
            us: 'Open the site, allow camera or upload a file',
            peak: 'Product site + analysis / course workflow',
            kreeda: 'App install / account flow (Google login on their site)',
        },
        {
            topic: 'Pricing shape',
            us: 'Try on the web; capacity limits when busy',
            peak: 'Free starter plus paid minute packs (see their site)',
            kreeda: 'App plans / access (see their site)',
        },
    ],
    pickPeak: [
        'You want a curriculum that tells you which drill to film next.',
        'You like progress framed as completed courses and levels.',
        'You want paid minute packs and priority queues laid out as subscriptions.',
    ],
    pickKreeda: [
        'You want an app-native dashboard and community rankings.',
        'You care about match-style metrics (speed, coverage, shot mix) first.',
        'You want a path to human coach review inside the same product.',
    ],
    pickUs: [
        'You already know the shot you want to fix and just need a fast read.',
        'You want pose skeletons and stroke labels you can scrub yourself.',
        'You want live browser feedback without installing another sports app.',
    ],
    faqs: [
        {
            q: 'Is IsoCourt an alternative to BadmintonPeak or Kreeda?',
            a: 'Yes, if you want AI badminton video feedback. It is not a course product like BadmintonPeak, and not a phone match-community app like Kreeda. IsoCourt is strongest for ad-hoc rally analysis and live browser sessions.',
        },
        {
            q: 'Which is better for beginners?',
            a: 'If you need a guided course path, BadmintonPeak is built for that. If you want phone match scores and community, look at Kreeda. If you already train with a coach or club and want clip feedback between sessions, IsoCourt fits better.',
        },
        {
            q: 'Does IsoCourt replace a human coach?',
            a: 'No. None of these tools should. IsoCourt tips are automated training insight only.',
        },
        {
            q: 'Can I use more than one?',
            a: 'Sure. Some players use a course or match app for structure, and IsoCourt when they want a quick pose and stroke read on a messy rally.',
        },
    ],
}
