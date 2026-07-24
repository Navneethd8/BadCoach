import { Icon } from './AnalyzeIcon'

export default function AnalyzeUploadPanel({
    preview,
    result,
    videoRef,
    setFile,
    setPreview,
    getRootProps,
    getInputProps,
    isDragActive,
}) {
    return (
        <div
            {...getRootProps()}
            className={`app-dropzone ${isDragActive ? 'app-dropzone--active' : ''}`}
        >
            <input {...getInputProps()} />
            {preview ? (
                <div className="w-full rounded-md overflow-hidden bg-black">
                    <video
                        ref={result ? undefined : videoRef}
                        src={preview}
                        className="w-full max-h-[300px] object-contain bg-black"
                        controls
                    />
                    <div className="flex items-center justify-between px-3 py-2 bg-neutral-900 border-t border-neutral-800/50">
                        <span className="text-xs text-neutral-400 flex items-center gap-1.5">
                            <Icon name="check_circle" size={13} className="text-brand" />
                            Clip ready to analyze
                        </span>
                        <button type="button"
                            onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null) }}
                            className="text-xs text-neutral-500 hover:text-neutral-300 flex items-center gap-1 transition-colors"
                        >
                            <Icon name="replay" size={13} /> Change Video
                        </button>
                    </div>
                </div>
            ) : (
                <div className="text-center text-[var(--text-muted)] p-6">
                    <Icon name="video_file" size={36} className="block mx-auto mb-3 text-[var(--text-subtle)]" />
                    <p className="text-sm font-medium text-[var(--text-secondary)]">Drag &amp; drop video here</p>
                    <p className="text-xs mt-1 text-[var(--text-muted)]">or click to select file</p>
                </div>
            )}
        </div>
    )
}
