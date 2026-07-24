import AppShell from './AppShell'
import LiveSessionView from './LiveSessionView'
import { usePageSeo } from '../seo/usePageSeo'
import { useLiveSession } from '../useLiveSession'

import '@fontsource-variable/material-symbols-outlined/standard.css'

export default function LiveSession() {
    usePageSeo('/live')
    const live = useLiveSession()

    return (
        <AppShell active="live" mainClassName="mx-auto max-w-[min(1920px,calc(100vw-1.25rem))] px-3 py-6 sm:px-5 sm:py-8">
            <h1 className="app-page-title">
                live <span className="text-brand">session</span>
            </h1>
            <p className="app-page-lead mb-5">Point your camera at the court and get real-time feedback.</p>
            <LiveSessionView {...live} />
        </AppShell>
    )
}
