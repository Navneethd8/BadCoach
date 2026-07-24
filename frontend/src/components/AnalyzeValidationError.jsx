import { Icon } from './AnalyzeIcon'

export default function AnalyzeValidationError({ result, setResult, setFile, setPreview }) {
    return (
        <div className="py-8 px-4">
            <div className={`border rounded-lg p-6 ${result.over_duration_limit
                ? 'bg-amber-950/30 border-amber-800/50'
                : 'bg-red-950/30 border-red-900/50'
                }`}>
                <div className="flex items-start gap-4">
                    <div className="flex-shrink-0">
                        <Icon
                            name={result.over_duration_limit ? 'schedule' : 'error'}
                            size={32}
                            className={result.over_duration_limit ? 'text-amber-400' : 'text-red-500'}
                        />
                    </div>
                    <div className="flex-1">
                        <h3 className={`text-lg font-semibold mb-2 ${result.over_duration_limit ? 'text-amber-300' : 'text-red-400'
                            }`}>
                            {result.over_duration_limit ? 'Video Too Long' : 'Not a Badminton Video'}
                        </h3>
                        <p className="text-sm text-neutral-300 mb-4">
                            {result.error_message}
                        </p>

                        {result.over_duration_limit && (
                            <div className="mt-2 mb-4 p-3 bg-amber-900/20 border border-amber-700/30 rounded-lg">
                                <p className="text-xs text-amber-200/80 leading-relaxed">
                                    <span className="font-semibold text-amber-300">💡 Tip:</span> For full-game analysis, upload each rally or quarter separately. This keeps analysis times fast and results more accurate for each play.
                                </p>
                            </div>
                        )}

                        {!result.over_duration_limit && result.validation_details && (
                            <div className="mt-4 p-3 bg-neutral-950/50 rounded border border-neutral-800">
                                <span className="text-xs text-neutral-500 block mb-2">Detection Details</span>
                                <div className="space-y-1.5 text-xs">
                                    {result.validation_details.pose_confidence !== undefined && (
                                        <div className="flex justify-between">
                                            <span className="text-neutral-400">Pose Detection Score:</span>
                                            <span className={result.validation_details.pose_confidence > 0.3 ? 'text-green-400' : 'text-red-400'}>
                                                {(result.validation_details.pose_confidence * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    )}
                                    {result.validation_details.model_confidence !== undefined && (
                                        <div className="flex justify-between">
                                            <span className="text-neutral-400">Model Confidence:</span>
                                            <span className={result.validation_details.model_confidence > 0.5 ? 'text-green-400' : 'text-red-400'}>
                                                {(result.validation_details.model_confidence * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    )}
                                    {result.validation_details.overhead_score !== undefined && (
                                        <div className="flex justify-between">
                                            <span className="text-neutral-400">Overhead Motion:</span>
                                            <span className={result.validation_details.overhead_score > 0.3 ? 'text-green-400' : 'text-red-400'}>
                                                {(result.validation_details.overhead_score * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        <button type="button"
                            onClick={() => { setResult(null); setFile(null); setPreview(null); }}
                            className="mt-4 px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-sm rounded transition-colors"
                        >
                            Try Another Video
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}
