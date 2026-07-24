/** Upsert a JSON-LD script tag by id. */
export function injectJsonLd(id, data) {
    if (typeof document === 'undefined') return () => {}
    let el = document.getElementById(id)
    if (!el) {
        el = document.createElement('script')
        el.id = id
        el.type = 'application/ld+json'
        document.head.appendChild(el)
    }
    el.textContent = JSON.stringify(data)
    return () => {
        el?.remove()
    }
}
