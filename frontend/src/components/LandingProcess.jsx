import PhoneCourtDemo from './PhoneCourtDemo'
import { FULL_FLOW_POSTER, FULL_FLOW_VIDEO, processSteps } from './landingData'

export default function LandingProcess() {
    return (
        <section id="how-it-works" className="figma-split figma-split--process scroll-mt-20">
            <div className="figma-split__panel figma-split__panel--copy">
                <div className="figma-split__inner">
                    <h2 className="figma-split-title">
                        three steps. <span className="figma-brand-accent">smarter court.</span>
                    </h2>
                    <ol className="figma-split-steps">
                        {processSteps.map((step) => (
                            <li key={step.num} className="figma-split-step">
                                <span className="figma-split-step__num">{step.num}</span>
                                <div>
                                    <h3 className="figma-split-step__title">{step.title}</h3>
                                    <p className="figma-split-step__body">{step.body}</p>
                                </div>
                            </li>
                        ))}
                    </ol>
                </div>
            </div>
            <div className="figma-split__panel figma-split__panel--visual figma-split__panel--video">
                <PhoneCourtDemo
                    video={FULL_FLOW_VIDEO}
                    poster={FULL_FLOW_POSTER}
                    frame="/phone-mockup.svg"
                    label="Full IsoCourt flow: upload, analyze, and review results"
                />
            </div>
        </section>
    )
}
