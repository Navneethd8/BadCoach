import { useEffect } from 'react'
import { DEFAULT_DESCRIPTION, SITE_NAME, absoluteUrl } from './site'

const SCRIPT_ID = 'isocourt-jsonld'

/**
 * Injects Organization + WebSite + SoftwareApplication JSON-LD for AI/search entities.
 */
export default function JsonLd() {
    useEffect(() => {
        const origin = absoluteUrl('/')
        const data = [
            {
                '@context': 'https://schema.org',
                '@type': 'Organization',
                name: SITE_NAME,
                alternateName: ['Birdzo', 'IsoCourt Birdzo'],
                url: origin || undefined,
                logo: absoluteUrl('/logo.svg'),
                description: DEFAULT_DESCRIPTION,
            },
            {
                '@context': 'https://schema.org',
                '@type': 'WebSite',
                name: SITE_NAME,
                alternateName: 'Birdzo',
                url: origin || undefined,
                description: DEFAULT_DESCRIPTION,
                publisher: { '@type': 'Organization', name: SITE_NAME },
            },
            {
                '@context': 'https://schema.org',
                '@type': 'SoftwareApplication',
                name: SITE_NAME,
                alternateName: 'Birdzo',
                applicationCategory: 'SportsApplication',
                operatingSystem: 'Web',
                description: DEFAULT_DESCRIPTION,
                url: origin || undefined,
                offers: {
                    '@type': 'Offer',
                    price: '0',
                    priceCurrency: 'USD',
                },
            },
        ]

        let el = document.getElementById(SCRIPT_ID)
        if (!el) {
            el = document.createElement('script')
            el.id = SCRIPT_ID
            el.type = 'application/ld+json'
            document.head.appendChild(el)
        }
        el.textContent = JSON.stringify(data)

        return () => {
            el?.remove()
        }
    }, [])

    return null
}
