import { Icon } from './AnalyzeIcon'

export default function AnalyzeDesktopRecordPanel({
    preview,
    result,
    videoRef,
    setFile,
    setPreview,
    cameraError,
    cameraPreviewRef,
    cameraStreamRef,
    isRecording,
    isFullScreen,
    recordingSeconds,
    openCamera,
    toggleFullScreen,
    startRecording,
    stopRecording,
    formatSeconds,
}) {
    if (cameraError) {
        return (
            <div className="flex flex-col items-center justify-center flex-1 gap-3 p-6 text-center min-h-[220px]">
                <Icon name="videocam_off" size={36} className="text-neutral-600" />
                <p className="text-sm text-neutral-400">{cameraError}</p>
                <button type="button"
                    onClick={openCamera}
                    className="mt-1 px-4 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-white text-xs rounded-md transition-colors"
                >
                    Try again
                </button>
            </div>
        )
    }

    if (preview) {
        return (
            <div className="w-full">
                <video
                    ref={result ? undefined : videoRef}
                    src={preview}
                    className="w-full max-h-[300px] object-contain bg-black"
                    controls
                />
                <div className="flex items-center justify-between px-3 py-2 bg-neutral-900/80">
                    <span className="text-xs text-neutral-400 flex items-center gap-1.5">
                        <Icon name="check_circle" size={13} className="text-brand" />
                        Clip ready to analyze
                    </span>
                    <button type="button"
                        onClick={() => { setFile(null); setPreview(null); openCamera() }}
                        className="text-xs text-neutral-500 hover:text-neutral-300 flex items-center gap-1 transition-colors"
                    >
                        <Icon name="replay" size={13} /> Re-record
                    </button>
                </div>
            </div>
        )
    }

    return (
        <>
            <div className={`relative ${isFullScreen ? 'fixed inset-0 z-50 bg-black flex flex-col justify-center' : 'flex-1'}`}>
                {isFullScreen && (
                    <button type="button"
                        onClick={toggleFullScreen}
                        className="absolute top-6 left-6 z-[60] p-2 bg-black/50 hover:bg-black/70 rounded-full text-white backdrop-blur-md"
                        aria-label="Exit fullscreen"
                    >
                        <Icon name="close" size={24} />
                    </button>
                )}

                <video
                    ref={cameraPreviewRef}
                    autoPlay
                    muted
                    playsInline
                    onClick={toggleFullScreen}
                    className={`w-full object-contain bg-black cursor-pointer ${isFullScreen ? 'h-[100dvh]' : 'max-h-[300px] min-h-[180px]'}`}
                />

                {!isRecording && !isFullScreen && cameraStreamRef.current?.active && (
                    <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-black/60 text-white/70 text-[10px] px-2 py-1 rounded-md backdrop-blur-sm pointer-events-none">
                        <Icon name="fullscreen" size={14} /> Tap to expand
                    </div>
                )}

                {isRecording && (
                    <div className={`absolute left-1/2 -translate-x-1/2 flex items-center gap-1.5 bg-black/70 border border-rose-500/50 text-rose-400 font-mono px-3 py-1 rounded-full backdrop-blur-sm ${isFullScreen ? 'top-12 text-sm px-4 py-1.5' : 'top-3 text-xs px-2.5 py-1'}`}>
                        <span className="w-2.5 h-2.5 bg-rose-500 rounded-full animate-pulse" />
                        {formatSeconds(recordingSeconds)}
                    </div>
                )}

                {isFullScreen && (
                    <div className="absolute bottom-12 left-0 right-0 flex justify-center pb-safe">
                        {isRecording ? (
                            <button type="button"
                                onClick={(e) => { e.stopPropagation(); stopRecording() }}
                                className="flex items-center justify-center w-20 h-20 bg-rose-600/90 text-white rounded-full transition-transform active:scale-90 border-4 border-white/20"
                                aria-label="Stop recording"
                            >
                                <span className="w-6 h-6 bg-white rounded-sm" />
                            </button>
                        ) : (
                            <button type="button"
                                onClick={(e) => { e.stopPropagation(); startRecording() }}
                                className="flex items-center justify-center w-20 h-20 bg-rose-600/90 text-white rounded-full transition-transform active:scale-90 border-4 border-white/20"
                                aria-label="Start recording"
                            >
                                <span className="w-6 h-6 bg-white rounded-full" />
                            </button>
                        )}
                    </div>
                )}
            </div>

            {!cameraError && !preview && !isFullScreen && (
                <div className="flex justify-center py-3 bg-neutral-900/90 border-t border-neutral-800">
                    {isRecording ? (
                        <button type="button"
                            onClick={stopRecording}
                            className="flex items-center gap-2 px-5 py-2 bg-rose-600 hover:bg-rose-700 text-white text-sm font-medium rounded-full transition-colors shadow-lg"
                        >
                            <span className="w-2.5 h-2.5 bg-white rounded-sm" />
                            Stop
                        </button>
                    ) : (
                        <button type="button"
                            onClick={startRecording}
                            className="flex items-center gap-2 px-5 py-2 bg-rose-600 hover:bg-rose-700 text-white text-sm font-medium rounded-full transition-colors shadow-lg"
                        >
                            <span className="w-2.5 h-2.5 bg-white rounded-full" />
                            Record
                        </button>
                    )}
                </div>
            )}
        </>
    )
}
