import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import LegalLayout from './LegalLayout'
import { usePageSeo } from '../seo/usePageSeo'
import { absoluteUrl } from '../seo/site'
import { injectJsonLd } from '../seo/injectJsonLd'
import { FAQS } from '../content/faq'
import { WHAT_IS_PATH } from '../content/whatIs'

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
                    Short answers about IsoCourt and Birdzo. For the full explainer, see{' '}
                    <Link to={WHAT_IS_PATH} className="text-brand underline-offset-2 hover:underline">
                        what AI stroke analysis is
                    </Link>
                    . Terms live in the{' '}
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
                Ready?{' '}
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
