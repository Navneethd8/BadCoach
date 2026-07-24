/**
 * Site-wide SEO config for the Vite SPA.
 * Set VITE_SITE_URL at build time (e.g. https://isocourt.example) for absolute
 * canonical/OG URLs and sitemap generation. Falls back to window.origin in the browser.
 */

export const SITE_NAME = 'IsoCourt'
export const DEFAULT_TITLE = 'IsoCourt · AI Badminton Coach'
export const DEFAULT_DESCRIPTION =
    'IsoCourt: AI-powered badminton stroke analysis. Upload a clip and get instant pose tracing, stroke scoring, and personalised coaching tips.'
export const DEFAULT_OG_IMAGE = '/marketing/pose-trace-hero.png'
export const TWITTER_HANDLE = '' // set when available, e.g. @isocourt

/** Prefer www — Vercel 307s apex → https://www.isocourt.fit */
const PRODUCTION_ORIGIN = 'https://www.isocourt.fit'

/** Absolute site origin without trailing slash. */
export function getSiteUrl() {
    const fromEnv = (import.meta.env.VITE_SITE_URL || '').replace(/\/$/, '')
    if (fromEnv) return fromEnv
    if (typeof window !== 'undefined' && window.location?.origin) {
        const origin = window.location.origin.replace(/\/$/, '')
        // Prefer production canonical when developing on localhost
        if (origin.includes('localhost') || origin.includes('127.0.0.1')) {
            return PRODUCTION_ORIGIN
        }
        return origin
    }
    return PRODUCTION_ORIGIN
}

export function absoluteUrl(path = '/') {
    const origin = getSiteUrl()
    const p = path.startsWith('/') ? path : `/${path}`
    return origin ? `${origin}${p}` : p
}

/** Per-route SEO (path → meta). */
export const ROUTE_SEO = {
    '/': {
        title: 'IsoCourt · AI Badminton Stroke Analysis',
        description:
            'Upload a badminton rally or stroke. IsoCourt\'s Birdzo coach reads footwork, contact, and shot type: pose tracing, scoring, and coaching tips.',
        path: '/',
        ogType: 'website',
    },
    '/analyze': {
        title: 'Analyze a Clip · IsoCourt AI Badminton Coach',
        description:
            'Upload or record a badminton clip for AI stroke analysis: pose skeletons, shot labels, quality scores, and coach recommendations.',
        path: '/analyze',
        ogType: 'website',
    },
    '/live': {
        title: 'Live Badminton Coaching · IsoCourt',
        description:
            'Point your camera at the court for real-time IsoCourt feedback — live stroke reads and AI coaching cues while you train.',
        path: '/live',
        ogType: 'website',
    },
    '/privacy': {
        title: 'Privacy · IsoCourt',
        description:
            'How IsoCourt handles uploaded video clips, feedback form data, and Google Analytics on the AI badminton coaching site.',
        path: '/privacy',
        ogType: 'website',
        noindex: false,
    },
    '/terms': {
        title: 'Terms of Use · IsoCourt',
        description:
            'IsoCourt terms of use: AI analysis and tips are for training insight only and are not a substitute for a qualified coach or medical advice.',
        path: '/terms',
        ogType: 'website',
        noindex: false,
    },
    '/faq': {
        title: 'FAQ · IsoCourt AI Badminton Coach',
        description:
            'Answers about IsoCourt, Birdzo, clip uploads, pose tracing, live coaching, privacy, and how AI badminton stroke analysis works.',
        path: '/faq',
        ogType: 'website',
    },
    '/glossary': {
        title: 'Glossary · IsoCourt Badminton Terms',
        description:
            'Plain-language glossary for IsoCourt: pose tracing, stroke reads, split-step, smash, clear, drop, quality score, Birdzo, and more.',
        path: '/glossary',
        ogType: 'website',
    },
    '/404': {
        title: 'Page Not Found · IsoCourt',
        description: 'This page does not exist on IsoCourt.',
        path: '/404',
        ogType: 'website',
        noindex: true,
    },
}

export function seoForPath(pathname) {
    const clean = (pathname || '/').replace(/\/$/, '') || '/'
    return ROUTE_SEO[clean] || ROUTE_SEO['/404']
}

export const INDEXABLE_PATHS = ['/', '/analyze', '/live', '/faq', '/glossary', '/privacy', '/terms']
