import { Icon } from './AnalyzeIcon'

export default function AnalyzeRecommendations({ recommendations }) {
    return (
        <div className="app-coach-panel">
            <span className="app-coach-panel__title">
                <Icon name="tips_and_updates" size={14} />
                Coach's Recommendations
            </span>
            <ul className="space-y-2">
                {recommendations.map((tip) => (
                    <li key={tip} className="app-coach-panel__item">
                        <span className="text-brand mt-1">•</span>
                        {tip}
                    </li>
                ))}
            </ul>
        </div>
    )
}
