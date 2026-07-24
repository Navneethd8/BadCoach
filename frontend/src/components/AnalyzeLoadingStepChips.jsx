import { Icon } from './AnalyzeIcon'

export default function AnalyzeLoadingStepChips({ loadingSteps, loadingStep }) {
    return (
        <div className="flex flex-wrap gap-2 px-1">
            {loadingSteps.map((step, idx) => {
                const isActive = idx === loadingStep
                const isDone = idx < loadingStep
                return (
                    <span
                        key={step.label}
                        className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] ${isDone ? 'bg-brand/15 text-brand' : isActive ? 'bg-[var(--surface-inset)] text-[var(--text)]' : 'text-[var(--text-muted)]'}`}
                    >
                        <Icon name={step.icon} size={12} />
                        {step.label}{isActive ? '…' : ''}
                    </span>
                )
            })}
        </div>
    )
}
