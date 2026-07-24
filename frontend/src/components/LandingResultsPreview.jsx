/** Static marketing preview — matches analyze results UI. */
import SvgIcon from './SvgIcon'

const TACTICAL = [
    { icon: 'pan_tool_alt', label: 'Backhand', pct: 54, tone: 'blue' },
    { icon: 'explore', label: 'Cross-court', pct: 62, tone: 'purple' },
    { icon: 'location_on', label: 'Mid_Front', pct: 55, tone: 'rose' },
    { icon: 'psychology', label: 'To_Create_Depth', pct: 35, tone: 'amber' },
]

const COACH_TIPS = [
    {
        title: 'Tighten your grip:',
        body: 'Ensure a relaxed thumb-dominant grip (beveled) to maximize pronation and snap during the final contact phase.',
    },
    {
        title: 'Minimize your backswing:',
        body: 'Keep your racquet prep compact near your waist to prevent telegraphing and ensure you catch the shuttle at its highest possible point.',
    },
]

function Icon({ name, size = 14 }) {
    return <SvgIcon name={name} size={size} />
}

export default function LandingResultsPreview() {
    return (
        <div className="landing-preview">
            <div className="app-result-panel">
                <div className="flex justify-between items-start mb-4">
                    <div>
                        <span className="text-xs text-[var(--text-subtle)] block mb-1">Execution Quality</span>
                        <div className="text-xl font-bold text-cyan-700 dark:text-cyan-300">Proficient</div>
                    </div>
                    <div className="text-right">
                        <span className="text-xs font-mono text-[var(--text-subtle)] block mb-1">Score</span>
                        <div className="app-result-score">6 / 10</div>
                    </div>
                </div>

                <div className="app-result-meter">
                    <div className="h-full rounded-full bg-cyan-500" style={{ width: '60%' }} />
                </div>

                <div className="pt-4 app-tactical-divider">
                    <span className="text-xs text-[var(--text-subtle)] block mb-3">Tactical Metrics</span>
                    <div className="flex flex-wrap gap-2">
                        {TACTICAL.map(({ icon, label, pct, tone }) => (
                            <div key={label} className={`app-tactical-chip app-tactical-chip--${tone}`}>
                                <Icon name={icon} size={12} />
                                {label.replace(/_/g, ' ')}
                                <span className="text-[9px] opacity-60">{pct}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="app-coach-panel">
                <span className="app-coach-panel__title">
                    <Icon name="tips_and_updates" size={14} />
                    Coach&apos;s Recommendations
                </span>
                <p className="mb-2 text-sm leading-relaxed text-[var(--text-secondary)]">
                    As your coach, here are three technical adjustments to elevate your backhand cross-court lob:
                </p>
                <ul className="space-y-2">
                    {COACH_TIPS.map(({ title, body }) => (
                        <li key={title} className="app-coach-panel__item">
                            <span className="text-brand mt-1">•</span>
                            <span>
                                <strong className="font-semibold text-[var(--text)]">{title}</strong> {body}
                            </span>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    )
}
