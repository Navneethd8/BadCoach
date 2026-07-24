import { usePageSeo } from '../seo/usePageSeo'
import LandingFeedbackSection from './LandingFeedbackSection'
import LandingFeatures from './LandingFeatures'
import LandingFinalCta from './LandingFinalCta'
import LandingFooter from './LandingFooter'
import LandingHeader from './LandingHeader'
import LandingHero from './LandingHero'
import LandingProcess from './LandingProcess'
import LandingResultsSection from './LandingResultsSection'
import LandingStrokeTicker from './LandingStrokeTicker'

export default function LandingPage() {
    usePageSeo('/')

    return (
        <>
            <LandingHeader />

            <main className="figma-landing figma-page-body theme-page min-h-screen w-full">
                <div className="figma-top-bar-spacer" aria-hidden />

                <LandingHero />
                <LandingStrokeTicker />
                <LandingResultsSection />
                <LandingFeatures />
                <LandingProcess />
                <LandingFinalCta />
                <LandingFeedbackSection />
                <LandingFooter />
            </main>
        </>
    )
}
