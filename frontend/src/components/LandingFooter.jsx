import { Link } from 'react-router-dom'
import Logo from './Logo'

export default function LandingFooter() {
    return (
        <footer className="figma-footer px-5 py-8 text-center sm:text-left">
            <div className="figma-section-inner flex flex-col sm:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-2">
                    <Logo size={20} className="text-brand" />
                    <span className="font-display text-sm font-bold">
                        Iso<span className="figma-brand-accent">Court</span>
                    </span>
                </div>
                <nav className="figma-footer-nav flex flex-wrap justify-center gap-x-6 gap-y-2" aria-label="Learn and legal">
                    <Link to="/faq" className="figma-footer-link">
                        FAQ
                    </Link>
                    <Link to="/glossary" className="figma-footer-link">
                        Glossary
                    </Link>
                    <Link to="/what-is-ai-badminton-stroke-analysis" className="figma-footer-link">
                        What is
                    </Link>
                    <Link to="/compare" className="figma-footer-link">
                        Compare
                    </Link>
                    <Link to="/privacy" className="figma-footer-link">
                        Privacy
                    </Link>
                    <Link to="/terms" className="figma-footer-link">
                        Terms
                    </Link>
                </nav>
            </div>
        </footer>
    )
}
