import { Icon } from './AnalyzeIcon'

export default function AnalyzeMobileRecordPanel({
    preview,
    result,
    videoRef,
    setFile,
    setPreview,
    nativeVideoInputRef,
    handleNativeVideoSelect,
}) {
    if (preview) {
        return (
            <div className="w-full bg-black">
                <video
                    ref={result ? undefined : videoRef}
                    src={preview}
                    className="w-full max-h-[300px] object-contain bg-black"
                    controls
                />
                <div className="flex items-center justify-between px-3 py-2 bg-neutral-900/80 border-t border-neutral-800/50">
                    <span className="text-xs text-neutral-400 flex items-center gap-1.5">
                        <Icon name="check_circle" size={13} className="text-brand" />
                        Clip ready
                    </span>
                    <button type="button"
                        onClick={() => { setFile(null); setPreview(null); nativeVideoInputRef.current?.click() }}
                        className="text-xs text-neutral-500 hover:text-neutral-300 flex items-center gap-1 transition-colors"
                    >
                        <Icon name="replay" size={13} /> Re-record
                    </button>
                </div>
                <input
                    type="file"
                    accept="video/*"
                    capture="environment"
                    ref={nativeVideoInputRef}
                    onChange={handleNativeVideoSelect}
                    className="hidden"
                />
            </div>
        )
    }

    return (
        <div className="flex flex-col items-center justify-center flex-1 gap-4 p-8 bg-neutral-900/50 text-center min-h-[220px]">
            <div className="text-neutral-400">
                <Icon name="photo_camera" size={48} className="opacity-50 mb-2 block mx-auto" />
                <p className="text-sm font-medium text-neutral-300 mb-1">Record on Device</p>
                <p className="text-xs text-neutral-500 mb-4 max-w-[200px]">Use your device's native camera for the best quality and zoom.</p>
            </div>

            <button type="button"
                onClick={() => nativeVideoInputRef.current?.click()}
                className="flex items-center justify-center gap-2 w-full py-3 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border border-neutral-700 rounded-md transition-colors shadow-sm"
            >
                <Icon name="videocam" size={18} />
                <span className="text-sm font-medium">Open Camera</span>
            </button>

            <input
                type="file"
                accept="video/*"
                capture="environment"
                ref={nativeVideoInputRef}
                onChange={handleNativeVideoSelect}
                className="hidden"
            />
        </div>
    )
}
