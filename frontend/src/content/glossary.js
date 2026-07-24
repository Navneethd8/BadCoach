/** Shared glossary — GlossaryPage + static prerender. */

export const TERMS = [
    {
        term: 'Birdzo',
        def: 'The AI coaching persona inside IsoCourt. Birdzo turns stroke scores and pose cues into short tips you can take back on court.',
    },
    {
        term: 'Clear',
        def: 'A high, deep shot usually hit from the rear court to push an opponent back. IsoCourt labels clears among other stroke types when the model sees them.',
    },
    {
        term: 'Contact point',
        def: 'Where racket meets shuttle relative to your body. Contact too far behind often shows up as a weak clear or mistimed smash in the timeline.',
    },
    {
        term: 'Drop shot',
        def: 'A soft shot that lands early in the opponent\'s forecourt. Often paired with clears from the same setup to disguise intent.',
    },
    {
        term: 'IsoCourt',
        def: 'The product: a web app for badminton clip analysis and live camera coaching. Birdzo is the coach voice inside it.',
    },
    {
        term: 'Live session',
        def: 'Browser mode that uses your camera for real-time feedback while you train, instead of uploading a file first.',
    },
    {
        term: 'Pose tracing',
        def: 'Skeleton overlay on analysis frames so you can see body position at contact and during footwork, not just the shuttle path.',
    },
    {
        term: 'Quality score',
        def: 'A 0–10 style rating of how clean the execution looked for a window or clip summary. It is model judgment, not a tournament ranking.',
    },
    {
        term: 'Smash',
        def: 'A steep, attacking overhead. Analysis looks at timing, contact, and related cues when labeling smash windows.',
    },
    {
        term: 'Split-step',
        def: 'The small hop or load before you move to the shuttle. A late split-step is a common miss pose tracing makes obvious.',
    },
    {
        term: 'Stroke read',
        def: 'IsoCourt\'s label for what you hit (and related tags like technique or court position) on a timeline window.',
    },
]
