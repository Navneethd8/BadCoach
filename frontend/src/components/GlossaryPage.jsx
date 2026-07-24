import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import LegalLayout from './LegalLayout'
import { usePageSeo } from '../seo/usePageSeo'
import { absoluteUrl } from '../seo/site'
import { injectJsonLd } from '../seo/injectJsonLd'

const TERMS = [
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

export default function GlossaryPage() {
    usePageSeo('/glossary')

    useEffect(() => {
        return injectJsonLd('isocourt-glossary-jsonld', {
            '@context': 'https://schema.org',
            '@type': 'DefinedTermSet',
            name: 'IsoCourt badminton glossary',
            url: absoluteUrl('/glossary'),
            hasDefinedTerm: TERMS.map(({ term, def }) => ({
                '@type': 'DefinedTerm',
                name: term,
                description: def,
                inDefinedTermSet: absoluteUrl('/glossary'),
            })),
        })
    }, [])

    return (
        <LegalLayout
            hero
            title={
                <>
                    court <span className="text-brand">glossary</span>
                </>
            }
            lead={
                <>
                    Plain-language terms used across IsoCourt. Product questions live in the{' '}
                    <Link to="/faq" className="text-brand underline-offset-2 hover:underline">
                        FAQ
                    </Link>
                    .
                </>
            }
        >
            <dl className="space-y-5">
                {TERMS.map(({ term, def }) => (
                    <div key={term} id={term.toLowerCase().replace(/\s+/g, '-')}>
                        <dt className="text-base font-semibold text-[var(--text)]">{term}</dt>
                        <dd className="mt-1">{def}</dd>
                    </div>
                ))}
            </dl>
            <p className="pt-4 text-[var(--text-subtle)] text-xs">
                Missing a term? Tell us on the{' '}
                <Link to="/#feedback" className="text-brand underline-offset-2 hover:underline">
                    feedback form
                </Link>
                .
            </p>
        </LegalLayout>
    )
}
