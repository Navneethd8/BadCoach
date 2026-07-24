import SvgIcon from './SvgIcon'
import { features } from './landingData'

export default function LandingFeatures() {
    return (
        <section id="features" className="figma-section px-5 sm:px-8 scroll-mt-20">
            <div className="figma-section-inner">
                <h2 className="figma-section-title text-center">core features</h2>
                <div className="figma-feature-grid">
                    {features.map(({ icon, title, description }) => (
                        <article key={title} className="figma-feature-card">
                            <div className="figma-icon-badge rounded-lg">
                                <SvgIcon name={icon} size={24} className="" />
                            </div>
                            <h3 className="figma-feature-title">{title}</h3>
                            <p className="figma-feature-desc">{description}</p>
                        </article>
                    ))}
                </div>
            </div>
        </section>
    )
}
