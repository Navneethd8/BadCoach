import { Icon } from './AnalyzeIcon'
import AnalyzeUploadPanel from './AnalyzeUploadPanel'
import AnalyzeRecordPanel from './AnalyzeRecordPanel'

export default function AnalyzeInputSection({
    file,
    loading,
    loadingStep,
    inputMode,
    switchMode,
    handleStreamAnalysis,
    uploadPanelProps,
    recordPanelProps,
}) {
    return (
        <section className="app-card mb-6">
            <div className="app-tabs">
                <button type="button"
                    onClick={() => switchMode('upload')}
                    className={`app-tab ${inputMode === 'upload' ? 'app-tab--active' : ''}`}
                >
                    <Icon name="upload" size={14} />
                    Upload
                </button>
                <button type="button"
                    onClick={() => switchMode('record')}
                    className={`app-tab ${inputMode === 'record' ? 'app-tab--active' : ''}`}
                >
                    <Icon name="videocam" size={14} />
                    Record
                </button>
            </div>

            {inputMode === 'upload' ? (
                <AnalyzeUploadPanel {...uploadPanelProps} />
            ) : (
                <AnalyzeRecordPanel {...recordPanelProps} />
            )}

            <button type="button"
                onClick={handleStreamAnalysis}
                disabled={!file || loading}
                className="app-btn-primary app-btn-primary--block mt-4"
            >
                {loadingStep >= 0 ? (
                    <span className="flex items-center justify-center gap-2">
                        <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Analyzing...
                    </span>
                ) : 'Analyze Stroke'}
            </button>
        </section>
    )
}
