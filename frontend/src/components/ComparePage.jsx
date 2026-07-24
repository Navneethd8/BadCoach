import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import LegalLayout from './LegalLayout'
import { usePageSeo } from '../seo/usePageSeo'
import { absoluteUrl } from '../seo/site'
import { injectJsonLd } from '../seo/injectJsonLd'
import { alternativesCompare } from './compareConfigs'

/**
 * Single alternatives comparison page (IsoCourt vs BadmintonPeak vs Kreeda).
 * Keep claims fair and based on public positioning; IsoCourt voice in VOICE.md.
 */
export default function ComparePage() {
    const {
        path,
        seoTitle,
        seoDescription,
        heroTitle,
        lead,
        competitors,
        intro,
        rows,
        pickPeak,
        pickKreeda,
        pickUs,
        faqs,
        updated,
    } = alternativesCompare

    usePageSeo(path, { title: seoTitle, description: seoDescription, path })

    useEffect(() => {
        const cleanupFaq = injectJsonLd('isocourt-compare-faq', {
            '@context': 'https://schema.org',
            '@type': 'FAQPage',
            mainEntity: faqs.map(({ q, a }) => ({
                '@type': 'Question',
                name: q,
                acceptedAnswer: { '@type': 'Answer', text: a },
            })),
            url: absoluteUrl(path),
        })
        const cleanupWeb = injectJsonLd('isocourt-compare-web', {
            '@context': 'https://schema.org',
            '@type': 'WebPage',
            name: seoTitle,
            description: seoDescription,
            url: absoluteUrl(path),
            dateModified: updated,
            about: [
                { '@type': 'SoftwareApplication', name: 'IsoCourt' },
                ...competitors.map(({ name, url }) => ({
                    '@type': 'SoftwareApplication',
                    name,
                    url,
                })),
            ],
        })
        return () => {
            cleanupFaq()
            cleanupWeb()
        }
    }, [path, seoTitle, seoDescription, competitors, faqs, updated])

    return (
        <LegalLayout hero title={heroTitle} lead={lead}>
            <p>{intro}</p>
            <p className="text-xs text-[var(--text-subtle)]">
                Last updated {updated}. Based on public product pages. Features change; verify on their sites before you buy.
            </p>

            <h2 className="text-base font-semibold text-[var(--text)] pt-2">Side by side</h2>
            <div className="overflow-x-auto rounded-lg border border-[var(--border)]">
                <table className="w-full text-left text-sm min-w-[36rem]">
                    <thead className="bg-[var(--surface-inset)] text-[var(--text-subtle)] text-xs uppercase tracking-wider">
                        <tr>
                            <th className="px-3 py-2.5 font-medium">Topic</th>
                            <th className="px-3 py-2.5 font-medium">IsoCourt</th>
                            <th className="px-3 py-2.5 font-medium">BadmintonPeak</th>
                            <th className="px-3 py-2.5 font-medium">Kreeda</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map(({ topic, us, peak, kreeda }) => (
                            <tr key={topic} className="border-t border-[var(--border)]">
                                <th scope="row" className="px-3 py-3 font-medium text-[var(--text)] align-top whitespace-nowrap">
                                    {topic}
                                </th>
                                <td className="px-3 py-3 align-top">{us}</td>
                                <td className="px-3 py-3 align-top">{peak}</td>
                                <td className="px-3 py-3 align-top">{kreeda}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <h2 className="text-base font-semibold text-[var(--text)] pt-2">Pick BadmintonPeak if…</h2>
            <ul className="list-disc pl-5 space-y-1.5">
                {pickPeak.map((item) => (
                    <li key={item}>{item}</li>
                ))}
            </ul>

            <h2 className="text-base font-semibold text-[var(--text)] pt-2">Pick Kreeda if…</h2>
            <ul className="list-disc pl-5 space-y-1.5">
                {pickKreeda.map((item) => (
                    <li key={item}>{item}</li>
                ))}
            </ul>

            <h2 className="text-base font-semibold text-[var(--text)] pt-2">Pick IsoCourt if…</h2>
            <ul className="list-disc pl-5 space-y-1.5">
                {pickUs.map((item) => (
                    <li key={item}>{item}</li>
                ))}
            </ul>

            <h2 className="text-base font-semibold text-[var(--text)] pt-2">FAQ</h2>
            <div className="space-y-5">
                {faqs.map(({ q, a }) => (
                    <section key={q}>
                        <h3 className="text-sm font-semibold text-[var(--text)] mb-1.5">{q}</h3>
                        <p>{a}</p>
                    </section>
                ))}
            </div>

            <div className="flex flex-wrap gap-3 pt-4">
                <Link to="/analyze" className="figma-cta figma-cta--primary">
                    Drop a clip on IsoCourt
                </Link>
                <Link to="/live" className="figma-cta figma-cta--secondary">
                    Try live
                </Link>
            </div>
            <p className="text-xs text-[var(--text-subtle)]">
                Also see the{' '}
                <Link to="/faq" className="text-brand underline-offset-2 hover:underline">FAQ</Link>
                {' '}and{' '}
                <Link to="/glossary" className="text-brand underline-offset-2 hover:underline">glossary</Link>
                . Competitor sites:{' '}
                {competitors.map(({ name, url }, i) => (
                    <span key={url}>
                        {i > 0 ? ' · ' : null}
                        <a href={url} target="_blank" rel="noopener noreferrer" className="text-brand underline-offset-2 hover:underline">
                            {name}
                        </a>
                    </span>
                ))}
                .
            </p>
        </LegalLayout>
    )
}
