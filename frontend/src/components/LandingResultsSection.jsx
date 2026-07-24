import LandingResultsPreview from './LandingResultsPreview'
import PoseFigureGate from './PoseFigureGate'

export default function LandingResultsSection() {
    return (
        <section id="results-preview" className="figma-split figma-split--results scroll-mt-20">
            <div className="figma-split__panel figma-split__panel--copy">
                <div className="figma-split__inner">
                    <h2 className="figma-split-title">
                        read the rally. <span className="figma-brand-accent">see the gap.</span>
                    </h2>
                    <LandingResultsPreview />
                </div>
            </div>
            <div className="figma-split__panel figma-split__panel--visual figma-split__panel--pose">
                <PoseFigureGate />
            </div>
        </section>
    )
}
