import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import LegalLayout from './LegalLayout'
import { usePageSeo } from '../seo/usePageSeo'
import { absoluteUrl } from '../seo/site'
import { injectJsonLd } from '../seo/injectJsonLd'
import { WHAT_IS, WHAT_IS_PATH } from '../content/whatIs'

export default function WhatIsStrokeAnalysisPage() {
    const { seoTitle, seoDescription, lead, updated, sections } = WHAT_IS

    usePageSeo(WHAT_IS_PATH, {
        title: seoTitle,
        description: seoDescription,
        path: WHAT_IS_PATH,
    })

    useEffect(() => {
        const cleanupArticle = injectJsonLd('isocourt-what-is-article', {
            '@context': 'https://schema.org',
            '@type': 'Article',
            headline: 'What is AI badminton stroke analysis?',
            description: seoDescription,
            dateModified: '2026-07-23',
            author: { '@type': 'Organization', name: 'IsoCourt' },
            publisher: { '@type': 'Organization', name: 'IsoCourt', url: absoluteUrl('/') },
            mainEntityOfPage: absoluteUrl(WHAT_IS_PATH),
            about: [
                { '@type': 'Thing', name: 'Badminton stroke analysis' },
                { '@type': 'SoftwareApplication', name: 'IsoCourt' },
            ],
        })
        const cleanupFaq = injectJsonLd('isocourt-what-is-faq', {
            '@context': 'https://schema.org',
            '@type': 'FAQPage',
            mainEntity: [
                {
                    '@type': 'Question',
                    name: 'What is AI badminton stroke analysis?',
                    acceptedAnswer: {
                        '@type': 'Answer',
                        text: sections[0].p[0],
                    },
                },
                {
                    '@type': 'Question',
                    name: 'Does IsoCourt replace a human coach?',
                    acceptedAnswer: {
                        '@type': 'Answer',
                        text: 'No. Tips are automated training insight only, not medical advice and not a replacement for a qualified coach.',
                    },
                },
            ],
            url: absoluteUrl(WHAT_IS_PATH),
        })
        return () => {
            cleanupArticle()
            cleanupFaq()
        }
    }, [seoDescription, sections])

    return (
        <LegalLayout
            hero
            title={
                <>
                    what is AI badminton <span className="text-brand">stroke analysis</span>?
                </>
            }
            lead={lead}
        >
            <p className="text-xs text-[var(--text-subtle)]">Last updated {updated}.</p>

            {sections.map(({ id, h, p }) => (
                <section key={id} id={id} className="space-y-2">
                    <h2 className="text-base font-semibold text-[var(--text)] pt-2">{h}</h2>
                    {p.map((para) => (
                        <p key={para}>{para}</p>
                    ))}
                </section>
            ))}

            <div className="flex flex-wrap gap-3 pt-4">
                <Link to="/analyze" className="figma-cta figma-cta--primary">
                    Drop a clip
                </Link>
                <Link to="/live" className="figma-cta figma-cta--secondary">
                    Try live
                </Link>
            </div>
            <p className="text-xs text-[var(--text-subtle)] pt-2">
                Also:{' '}
                <Link to="/faq" className="text-brand underline-offset-2 hover:underline">
                    FAQ
                </Link>
                {' · '}
                <Link to="/glossary" className="text-brand underline-offset-2 hover:underline">
                    Glossary
                </Link>
                {' · '}
                <Link to="/compare" className="text-brand underline-offset-2 hover:underline">
                    Compare
                </Link>
                .
            </p>
        </LegalLayout>
    )
}
