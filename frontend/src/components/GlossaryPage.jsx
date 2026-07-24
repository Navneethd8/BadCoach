import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import LegalLayout from './LegalLayout'
import { usePageSeo } from '../seo/usePageSeo'
import { absoluteUrl } from '../seo/site'
import { injectJsonLd } from '../seo/injectJsonLd'
import { TERMS } from '../content/glossary'

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
