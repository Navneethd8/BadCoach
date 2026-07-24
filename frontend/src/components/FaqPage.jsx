import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import LegalLayout from './LegalLayout'
import { usePageSeo } from '../seo/usePageSeo'
import { absoluteUrl } from '../seo/site'
import { injectJsonLd } from '../seo/injectJsonLd'

const FAQS = [
    {
        q: 'What is IsoCourt?',
        a: 'IsoCourt is a web app for badminton video analysis. Upload a clip or go live with your camera. You get pose skeletons, stroke labels, a quality score, and short coaching tips from Birdzo, the in-product coach.',
    },
    {
        q: 'Who is Birdzo?',
        a: 'Birdzo is the coaching persona inside IsoCourt, not a separate product. Think of Birdzo as the voice that turns stroke scores into cues you can try on the next rally.',
    },
    {
        q: 'What video should I upload?',
        a: 'A single stroke, a short rally, or a drill works best. Keep the shuttle in frame and the camera steady. Long full-match uploads are slower and often less useful than rally-by-rally clips.',
    },
    {
        q: 'How is this different from watching my own footage?',
        a: 'Phone scrubbing eats practice time and you still guess. IsoCourt labels strokes, shows a skeleton on key frames, and gives specific tips tied to timestamps so you know what to fix next.',
    },
    {
        q: 'What is pose tracing?',
        a: 'Pose tracing draws a skeleton over your body on analysis frames. You can see late split-steps, contact behind the body, and other form misses without scrubbing blindly. See the glossary for related terms.',
    },
    {
        q: 'Does live coaching work in the browser?',
        a: 'Yes. Open Live, allow camera access, and point it at the court. You get real-time feedback while you train. Capacity limits may apply when the server is busy.',
    },
    {
        q: 'Is IsoCourt a substitute for a coach?',
        a: 'No. Tips are automated training insight only. They are not medical advice and not a replacement for a qualified coach.',
    },
    {
        q: 'Is my video private?',
        a: 'Clips are processed so we can run pose and stroke analysis. Retention depends on how the backend is configured. See the Privacy page for the current summary, and use the feedback form for deletion questions.',
    },
]

export default function FaqPage() {
    usePageSeo('/faq')

    useEffect(() => {
        return injectJsonLd('isocourt-faq-jsonld', {
            '@context': 'https://schema.org',
            '@type': 'FAQPage',
            mainEntity: FAQS.map(({ q, a }) => ({
                '@type': 'Question',
                name: q,
                acceptedAnswer: {
                    '@type': 'Answer',
                    text: a,
                },
            })),
            url: absoluteUrl('/faq'),
        })
    }, [])

    return (
        <LegalLayout
            hero
            title={
                <>
                    frequently <span className="text-brand">asked</span>
                </>
            }
            lead={
                <>
                    Straight answers about IsoCourt, Birdzo, clips, and live coaching. Term definitions live in the{' '}
                    <Link to="/glossary" className="text-brand underline-offset-2 hover:underline">
                        glossary
                    </Link>
                    .
                </>
            }
        >
            <div className="space-y-6">
                {FAQS.map(({ q, a }) => (
                    <section key={q}>
                        <h2 className="text-base font-semibold text-[var(--text)] mb-2">{q}</h2>
                        <p>{a}</p>
                    </section>
                ))}
            </div>
            <p className="pt-4">
                Ready to try it?{' '}
                <Link to="/analyze" className="text-brand underline-offset-2 hover:underline">
                    Analyze a clip
                </Link>{' '}
                or{' '}
                <Link to="/live" className="text-brand underline-offset-2 hover:underline">
                    go live
                </Link>
                .
            </p>
        </LegalLayout>
    )
}
