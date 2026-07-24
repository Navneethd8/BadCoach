import { Icon } from './AnalyzeIcon'

export default function AnalyzeResultSummary({ result, getQualityColor, getQualityBarColor }) {
    return (
        <div className="space-y-3">
            <div className="app-result-panel">
                <div className="flex justify-between items-start mb-4">
                    <div>
                        <span className="text-xs text-[var(--text-subtle)] block mb-1">Execution Quality</span>
                        <div className={`text-xl font-bold ${getQualityColor(result.quality)}`}>
                            {result.quality}
                        </div>
                    </div>
                    <div className="text-right">
                        <span className="text-xs font-mono text-[var(--text-subtle)] block mb-1">Score</span>
                        <div className="app-result-score">{result.quality_numeric || 0} / 10</div>
                    </div>
                </div>

                <div className="app-result-meter">
                    <div
                        className={`h-full rounded-full ${getQualityBarColor(result.quality)}`}
                        style={{ width: `${((result.quality_numeric || 0) / 10) * 100}%` }}
                    />
                </div>

                {result.tactical_analysis && (
                    <div className="pt-4 app-tactical-divider">
                        <span className="text-xs text-[var(--text-subtle)] block mb-3">Tactical Metrics</span>
                        <div className="flex flex-wrap gap-2">
                            <div className="app-tactical-chip app-tactical-chip--blue">
                                <Icon name="pan_tool_alt" size={12} />
                                {result.tactical_analysis.technique?.label || 'Unknown'}
                                {result.tactical_analysis.technique?.confidence > 0 && (
                                    <span className="text-[9px] opacity-60">{(result.tactical_analysis.technique.confidence * 100).toFixed(0)}%</span>
                                )}
                            </div>
                            <div className="app-tactical-chip app-tactical-chip--purple">
                                <Icon name="explore" size={12} />
                                {result.tactical_analysis.placement?.label || 'Unknown'}
                                {result.tactical_analysis.placement?.confidence > 0 && (
                                    <span className="text-[9px] opacity-60">{(result.tactical_analysis.placement.confidence * 100).toFixed(0)}%</span>
                                )}
                            </div>
                            <div className="app-tactical-chip app-tactical-chip--rose">
                                <Icon name="location_on" size={12} />
                                {result.tactical_analysis.position?.label || 'Unknown'}
                                {result.tactical_analysis.position?.confidence > 0 && (
                                    <span className="text-[9px] opacity-60">{(result.tactical_analysis.position.confidence * 100).toFixed(0)}%</span>
                                )}
                            </div>
                            <div className="app-tactical-chip app-tactical-chip--amber">
                                <Icon name="psychology" size={12} />
                                {result.tactical_analysis.intent?.label || 'None'}
                                {result.tactical_analysis.intent?.confidence > 0 && (
                                    <span className="text-[9px] opacity-60">{(result.tactical_analysis.intent.confidence * 100).toFixed(0)}%</span>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
