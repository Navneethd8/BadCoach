/** Shared FAQ copy — used by FaqPage + static prerender. Keep answers short and unique. */

export const FAQS = [
    {
        q: 'What is IsoCourt?',
        a: 'A browser tool that reads badminton video. You upload a clip or open Live; it returns pose skeletons, stroke labels, a score, and short tips from Birdzo.',
    },
    {
        q: 'Who is Birdzo?',
        a: 'The coach voice inside IsoCourt, not a second app. Birdzo turns the model output into cues you can try on the next rally.',
    },
    {
        q: 'What video should I upload?',
        a: 'One stroke, a short rally, or a drill. Shuttle in frame, camera steady. Full matches are slower and usually worse than rally clips.',
    },
    {
        q: 'How is this different from watching my own footage?',
        a: 'You get a stroke timeline, skeleton frames, and tips tied to timestamps. Less scrubbing, less guessing.',
    },
    {
        q: 'What is pose tracing?',
        a: 'A skeleton drawn on your body in analysis frames. Late split-steps and contact behind the body show up without frame-by-frame guesswork. Definitions: glossary.',
    },
    {
        q: 'Does live coaching work in the browser?',
        a: 'Yes. Open Live, allow the camera, point it at the court. Feedback runs while you train. Busy periods may hit capacity limits.',
    },
    {
        q: 'Is IsoCourt a substitute for a coach?',
        a: 'No. Automated training insight only. Not medical advice. Not a replacement for a qualified coach.',
    },
    {
        q: 'Is my video private?',
        a: 'Clips are processed for pose and stroke analysis. See Privacy for retention. Ask via the feedback form if you need a deletion request.',
    },
]
