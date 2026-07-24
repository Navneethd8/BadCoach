import { Icon } from './AnalyzeIcon'

export default function AnalyzeLoadingSteps({ loadingSteps, loadingStep }) {
    return (
        <div className="py-12 px-4">
            <div className="space-y-4">
                {loadingSteps.map((step, idx) => {
                    const isActive = idx === loadingStep
                    const isDone = idx < loadingStep
                    return (
                        <div
                            key={step.label}
                            className={`flex items-center gap-3 py-2 px-3 rounded-md transition-all duration-300 ${isActive ? 'app-loading-step--active' : ''
                                }`}
                        >
                            <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-300 ${isDone ? 'app-loading-step__ring--done' : isActive ? 'app-loading-step__ring--active' : 'app-loading-step__ring--pending'
                                }`}>
                                {isDone ? (
                                    <Icon name="check" size={14} className="text-white" />
                                ) : isActive ? (
                                    <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
                                ) : (
                                    <span className="w-1.5 h-1.5 bg-neutral-400 rounded-full" />
                                )}
                            </div>
                            <Icon
                                name={step.icon}
                                size={18}
                                className={`transition-colors duration-300 ${isDone ? 'text-brand' : isActive ? 'app-loading-step__icon--active' : 'app-loading-step__icon--pending'
                                    }`}
                            />
                            <span className={`text-sm transition-colors duration-300 ${isDone ? 'app-loading-step__label--done' : isActive ? 'app-loading-step__label--active' : 'app-loading-step__label--pending'
                                }`}>
                                {step.label}{isActive ? '...' : ''}
                            </span>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
