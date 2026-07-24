import { useEffect } from 'react'
import {
    DEFAULT_DESCRIPTION,
    DEFAULT_OG_IMAGE,
    DEFAULT_TITLE,
    SITE_NAME,
    TWITTER_HANDLE,
    absoluteUrl,
    seoForPath,
} from './site'

function upsertMeta(attr, key, content) {
    if (content == null || content === '') return
    let el = document.head.querySelector(`meta[${attr}="${key}"]`)
    if (!el) {
        el = document.createElement('meta')
        el.setAttribute(attr, key)
        document.head.appendChild(el)
    }
    el.setAttribute('content', content)
}

function upsertLink(rel, href) {
    if (!href) return
    let el = document.head.querySelector(`link[rel="${rel}"]`)
    if (!el) {
        el = document.createElement('link')
        el.setAttribute('rel', rel)
        document.head.appendChild(el)
    }
    el.setAttribute('href', href)
}

/**
 * Vite SPA per-route SEO: mutates document head after navigation.
 * Pair with prerender (vite-plugin-prerender / vpr) for crawler-visible HTML.
 */
export function usePageSeo(pathname, overrides = {}) {
    const {
        title: titleOverride,
        description: descriptionOverride,
        path: pathOverride,
        ogType: ogTypeOverride,
        noindex: noindexOverride,
        image: imageOverride,
    } = overrides

    useEffect(() => {
        const base = seoForPath(pathname)
        const title = titleOverride || base.title || DEFAULT_TITLE
        const description = descriptionOverride || base.description || DEFAULT_DESCRIPTION
        const path = pathOverride || base.path || pathname || '/'
        const ogType = ogTypeOverride || base.ogType || 'website'
        const noindex = noindexOverride ?? base.noindex ?? false
        const image = absoluteUrl(imageOverride || DEFAULT_OG_IMAGE)
        const url = absoluteUrl(path)

        document.title = title

        upsertMeta('name', 'description', description)
        upsertMeta('name', 'robots', noindex ? 'noindex, nofollow' : 'index, follow')
        upsertMeta('name', 'theme-color', '#0a0a0a')
        upsertMeta('name', 'application-name', SITE_NAME)

        upsertMeta('property', 'og:type', ogType)
        upsertMeta('property', 'og:site_name', SITE_NAME)
        upsertMeta('property', 'og:title', title)
        upsertMeta('property', 'og:description', description)
        upsertMeta('property', 'og:url', url)
        upsertMeta('property', 'og:image', image)

        upsertMeta('name', 'twitter:card', 'summary_large_image')
        upsertMeta('name', 'twitter:title', title)
        upsertMeta('name', 'twitter:description', description)
        upsertMeta('name', 'twitter:image', image)
        if (TWITTER_HANDLE) {
            upsertMeta('name', 'twitter:site', TWITTER_HANDLE)
        }

        upsertLink('canonical', url)
    }, [
        pathname,
        titleOverride,
        descriptionOverride,
        pathOverride,
        ogTypeOverride,
        noindexOverride,
        imageOverride,
    ])
}
