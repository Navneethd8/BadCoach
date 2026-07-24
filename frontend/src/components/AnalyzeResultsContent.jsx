import { Icon } from './AnalyzeIcon'
import AnalyzeLoadingStepChips from './AnalyzeLoadingStepChips'
import AnalyzeResultSummary from './AnalyzeResultSummary'
import AnalyzeRecommendations from './AnalyzeRecommendations'
import AnalyzeTimeline from './AnalyzeTimeline'

export default function AnalyzeResultsContent({
    result,
    loadingStep,
    loadingSteps,
    displayTimeline,
    preview,
    videoRef,
    getQualityColor,
    getQualityBarColor,
    timelineHasPoseFrames,
    handleTimelineClick,
    openFrameAnalysis,
}) {
    return (
        <div className="space-y-4">
            {loadingStep >= 0 && (
                <AnalyzeLoadingStepChips loadingSteps={loadingSteps} loadingStep={loadingStep} />
            )}

            {result?.cache_hit && (
                <div className="flex items-center gap-2 text-[11px] text-brand/70 font-medium px-1">
                    <Icon name="bolt" size={13} className="text-brand" />
                    Instant result: same clip analyzed before
                </div>
            )}
            {result?.cache_hit && result.timeline?.length > 0 && !timelineHasPoseFrames(result.timeline) && (
                <p className="text-xs text-amber-700 dark:text-amber-400/90 px-1">
                    Cached result has no pose frames. Trim a second off the clip or wait ~1 hour, then analyze again for skeleton views.
                </p>
            )}

            {preview && (
                <div className="overflow-hidden rounded-md border border-[var(--border)] bg-black">
                    <video
                        ref={videoRef}
                        src={preview}
                        className="max-h-[280px] w-full object-contain"
                        controls
                        playsInline
                    />
                    <p className="border-t border-[var(--border)] px-3 py-2 text-[11px] text-[var(--text-muted)]">
                        Tap a skeleton below to inspect that frame; timestamps seek this clip.
                    </p>
                </div>
            )}

            {result && (
                <AnalyzeResultSummary
                    result={result}
                    getQualityColor={getQualityColor}
                    getQualityBarColor={getQualityBarColor}
                />
            )}

            {result?.recommendations && result.recommendations.length > 0 && (
                <AnalyzeRecommendations recommendations={result.recommendations} />
            )}

            {displayTimeline.length > 0 && (
                <AnalyzeTimeline
                    displayTimeline={displayTimeline}
                    loadingStep={loadingStep}
                    result={result}
                    handleTimelineClick={handleTimelineClick}
                    openFrameAnalysis={openFrameAnalysis}
                />
            )}
        </div>
    )
}
