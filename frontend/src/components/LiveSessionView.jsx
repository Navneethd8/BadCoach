function Icon({ name, size = 20, className = '' }) {
    return (
        <span className={`material-symbols-outlined ${className}`} style={{ fontSize: size }}>
            {name}
        </span>
    )
}

function getQualityColor(quality) {
    const value = String(quality).toLowerCase()
    if (value.includes('elite') || value.includes('expert') || value.includes('advanced')) return 'text-brand'
    if (value.includes('proficient')) return 'text-cyan-400'
    if (value.includes('competent')) return 'text-amber-400'
    if (value.includes('developing')) return 'text-orange-400'
    return 'text-red-400'
}

function fmtTime(date) {
    return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`
}

export default function LiveSessionView({
    cameraError, canvasRef, chatLog, chatScrollRef, dismissCapacity, dismissError,
    endSession, errorMsg, isExpanded, isImmersive, isNativeFullscreen, lastResult,
    onBreak, overlayMessages, overlayScrollRef, pauseSession, resumeSession,
    startSession, status, statusMsg, toggleFullscreen, toggleVoice, videoContainerRef,
    videoRef, voiceEnabled,
}) {
    return (
        <>
            {status === 'capacity' && (
                <div className="mb-4 flex items-start gap-3 bg-amber-950/90 border border-amber-700/60 text-amber-200 text-sm px-5 py-3.5 rounded-xl">
                    <Icon name="hourglass_top" size={18} className="text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                        <p className="font-semibold text-amber-300 mb-0.5">Live sessions at capacity</p>
                        <p className="text-xs text-amber-200/80">{errorMsg}</p>
                    </div>
                    <button type="button" onClick={dismissCapacity} className="ml-auto text-amber-400 hover:text-amber-200" aria-label="Dismiss capacity notice"><Icon name="close" size={16} /></button>
                </div>
            )}
            {(status === 'error' || cameraError) && (
                <div className="mb-4 flex items-start gap-3 bg-red-950/80 border border-red-700/50 text-red-200 text-sm px-5 py-3.5 rounded-xl">
                    <Icon name="error" size={18} className="text-red-400 flex-shrink-0 mt-0.5" />
                    <div>
                        <p className="font-semibold text-red-300 mb-0.5">Something went wrong</p>
                        <p className="text-xs text-red-200/80">{cameraError || errorMsg}</p>
                    </div>
                    <button type="button" onClick={dismissError} className="ml-auto text-red-400 hover:text-red-200" aria-label="Dismiss error notice"><Icon name="close" size={16} /></button>
                </div>
            )}

            <div className="flex flex-col gap-5 lg:flex-row lg:items-stretch lg:gap-4 xl:gap-5 min-[1700px]:gap-6">
                <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                    <div
                        ref={videoContainerRef}
                        className={`app-panel-dark app-live-video relative aspect-video w-full overflow-hidden touch-manipulation ${isImmersive ? 'app-live-video--immersive' : ''} ${isNativeFullscreen ? 'app-live-video--fullscreen' : ''}`}
                    >
                        <video ref={videoRef} className="w-full h-full object-cover" playsInline muted />
                        <canvas ref={canvasRef} className="hidden" />

                        {(status === 'live' || status === 'paused') && (
                            <button type="button" className="absolute inset-0 z-[1] cursor-pointer border-0 bg-transparent p-0 touch-manipulation" onClick={toggleFullscreen} aria-label={isExpanded ? 'Exit fullscreen' : 'Enter fullscreen'} />
                        )}

                        {status === 'idle' && (
                            <div className="app-live-idle absolute inset-0 flex flex-col items-center justify-center gap-5 px-6">
                                <Icon name="videocam" size={48} className="app-live-idle__icon" />
                                <button type="button" onClick={startSession} className="figma-cta figma-cta--primary">start game</button>
                                <p className="app-live-idle__hint">Camera on, court in frame — Birdzo handles the rest.</p>
                            </div>
                        )}

                        {status === 'connecting' && (
                            <div className="absolute inset-0 flex items-center justify-center bg-neutral-900/80">
                                <div className="flex items-center gap-3 text-neutral-400">
                                    <span className="w-5 h-5 border-2 border-brand/30 border-t-brand rounded-full animate-spin" />
                                    <span className="text-sm">Starting session...</span>
                                </div>
                            </div>
                        )}

                        {(status === 'live' || status === 'paused') && (
                            <div className="app-live-video__hud-top absolute top-3 left-3 z-[3] flex max-w-[calc(100%-5.5rem)] items-center gap-2 pointer-events-none sm:max-w-none">
                                {status === 'live' ? (
                                    <div className="flex items-center gap-1.5 bg-red-600/90 text-white text-xs font-semibold px-3 py-1.5 rounded-full shadow-lg"><span className="w-2 h-2 bg-white rounded-full" />LIVE</div>
                                ) : (
                                    <div className="flex items-center gap-1.5 bg-amber-600/90 text-white text-xs font-semibold px-3 py-1.5 rounded-full shadow-lg"><Icon name="pause" size={12} />PAUSED</div>
                                )}
                                {statusMsg && status === 'live' && <div key={statusMsg} className="bg-neutral-900/80 backdrop-blur-sm text-neutral-300 text-xs px-3 py-1.5 rounded-full">{statusMsg}</div>}
                            </div>
                        )}

                        {status === 'paused' && (
                            <div className="absolute inset-0 z-[4] pointer-events-none bg-neutral-950/50 flex items-center justify-center">
                                <button type="button" onClick={resumeSession} className="figma-cta figma-cta--primary pointer-events-auto"><Icon name="play_arrow" size={18} className="align-middle -ml-0.5 mr-1" />resume</button>
                            </div>
                        )}

                        {(status === 'live' || status === 'paused') && (
                            <div className="app-live-video__hud-top app-live-video__controls absolute top-3 right-3 z-[3] flex items-center gap-1.5 sm:gap-2">
                                <button type="button" onClick={toggleFullscreen} className="app-live-control-btn bg-neutral-800/80 hover:bg-neutral-700 text-white text-xs font-medium rounded-lg transition-colors pointer-events-auto" title={isExpanded ? 'Exit fullscreen' : 'Fullscreen'} aria-label={isExpanded ? 'Exit fullscreen' : 'Enter fullscreen'}><Icon name={isExpanded ? 'fullscreen_exit' : 'fullscreen'} size={16} /></button>
                                {status === 'live' && <button type="button" onClick={pauseSession} className="app-live-control-btn bg-neutral-800/80 hover:bg-neutral-700 text-white text-xs font-medium rounded-lg transition-colors pointer-events-auto" aria-label="Pause session"><Icon name="pause" size={16} /><span className="hidden sm:inline sm:ml-0.5">Pause</span></button>}
                                <button type="button" onClick={endSession} className="app-live-control-btn bg-red-900/60 hover:bg-red-800/80 text-red-200 text-xs font-medium rounded-lg transition-colors pointer-events-auto" aria-label="End session"><Icon name="stop_circle" size={16} className="sm:hidden" /><span className="hidden sm:inline">End Session</span><span className="sm:hidden sr-only">End session</span></button>
                            </div>
                        )}

                        {(status === 'live' || status === 'paused') && overlayMessages.length > 0 && (
                            <div className="app-live-commentary absolute right-2 top-14 bottom-[4.5rem] z-[2] pointer-events-none sm:right-3">
                                <div className="app-live-commentary__fade" aria-hidden="true" />
                                <div ref={overlayScrollRef} className="app-live-commentary__scroll h-full overflow-hidden">
                                    <div className="flex min-h-full flex-col justify-end gap-1.5 px-1.5 py-2 sm:px-2">
                                        {overlayMessages.map(msg => (
                                            <div key={msg.id} className="app-live-commentary__msg">
                                                {msg.type === 'coach' ? <><span className="app-live-commentary__author">Coach</span><p className="app-live-commentary__text">{msg.text}</p></> : msg.type === 'analysis' ? <><span className="app-live-commentary__author app-live-commentary__author--stroke">Stroke</span><p className="app-live-commentary__text app-live-commentary__text--muted">{msg.text}</p></> : <p className="app-live-commentary__text app-live-commentary__text--system">{msg.text}</p>}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        {!isExpanded && (status === 'live' || status === 'paused') && !lastResult && !onBreak && <div className="absolute bottom-3 left-3 z-[2] pointer-events-none flex items-center gap-1.5 rounded-full bg-black/45 px-2.5 py-1 text-[10px] text-white/70 backdrop-blur-sm"><Icon name="touch_app" size={12} />Tap video for fullscreen</div>}

                        {onBreak && status === 'live' && (
                            <div className={`absolute bottom-0 inset-x-0 z-[2] rounded-b-lg px-4 py-3 flex items-center justify-center gap-2 pointer-events-none ${onBreak === 'no_badminton' ? 'bg-amber-950/80' : 'bg-neutral-950/80'} backdrop-blur-sm`}>
                                <Icon name={onBreak === 'no_badminton' ? 'videocam_off' : 'pause_circle'} size={16} className={onBreak === 'no_badminton' ? 'text-amber-500' : 'text-neutral-500'} />
                                <span className={`text-xs ${onBreak === 'no_badminton' ? 'text-amber-300' : 'text-neutral-400'}`}>{onBreak === 'no_badminton' ? 'No badminton detected. Point camera at the court' : 'Break in play. Waiting for action'}</span>
                            </div>
                        )}

                        {lastResult && !onBreak && (status === 'live' || status === 'paused') && (
                            <div className="absolute bottom-0 inset-x-0 z-[2] rounded-b-lg bg-neutral-950/75 backdrop-blur-sm px-3 py-2.5 pointer-events-none">
                                <div className="grid grid-cols-3 gap-x-3 gap-y-1.5 text-center">
                                    <Metric label="Stroke" value={lastResult.label} bold />
                                    <Metric label="Technique" value={lastResult.metrics?.technique?.label || '-'} />
                                    <Metric label="Placement" value={lastResult.metrics?.placement?.label || '-'} />
                                    <Metric label="Position" value={lastResult.metrics?.position?.label || '-'} />
                                    <Metric label="Intent" value={lastResult.metrics?.intent?.label || '-'} />
                                    <Metric label="Quality" value={lastResult.metrics?.quality || '-'} className={getQualityColor(lastResult.metrics?.quality)} />
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                <aside className="app-card flex h-[min(18rem,42vh)] w-full shrink-0 flex-col overflow-hidden !p-0 sm:h-[min(20rem,44vh)] lg:min-h-0 lg:max-h-[min(40rem,calc(100dvh-9rem))] lg:w-72 lg:shrink-0 xl:w-80 min-[1700px]:w-96" aria-label="AI Coach chat">
                    <div className="app-chat-header flex items-center gap-2 px-4 py-3 flex-shrink-0">
                        <Icon name="smart_toy" size={16} className="text-brand" />
                        <span className="text-xs font-semibold text-[var(--text-subtle)] uppercase tracking-wider font-mono">AI coach</span>
                        <div className="ml-auto flex items-center gap-2">
                            {chatLog.length > 0 && <span className="text-[10px] text-[var(--text-muted)]">{chatLog.length} messages</span>}
                            <button type="button" onClick={toggleVoice} title={voiceEnabled ? 'Mute voice' : 'Unmute voice'} aria-label={voiceEnabled ? 'Mute voice' : 'Unmute voice'} className={`p-1 rounded transition-colors ${voiceEnabled ? 'text-brand hover:opacity-80' : 'text-[var(--text-subtle)] hover:text-[var(--text)]'}`}><Icon name={voiceEnabled ? 'volume_up' : 'volume_off'} size={16} /></button>
                        </div>
                    </div>
                    <div ref={chatScrollRef} className="flex-1 min-h-0 overflow-y-auto overscroll-y-contain px-4 py-3 space-y-3 scroll-smooth" role="log" aria-live="polite" aria-relevant="additions">
                        {chatLog.length === 0 && <div className="flex flex-col items-center justify-center h-full text-center gap-2 py-8"><Icon name="sports" size={28} className="text-[var(--border-strong)]" /><p className="text-xs text-[var(--text-muted)]">Start a game to see live coaching commentary here.</p></div>}
                        {chatLog.map(msg => (
                            <div key={msg.id} className="flex gap-2.5 items-start">
                                <span className="text-[10px] text-[var(--text-muted)] font-mono mt-1 flex-shrink-0 w-12">{fmtTime(msg.ts)}</span>
                                {msg.type === 'coach' ? <div className="app-chat-coach flex-1 min-w-0">{msg.text}</div> : msg.type === 'analysis' ? <div className="app-chat-analysis flex-1 min-w-0"><Icon name="sports_tennis" size={11} className="align-middle mr-1 opacity-70" />{msg.text}</div> : <div className="text-[11px] text-[var(--text-muted)] italic flex-1 min-w-0">{msg.text}</div>}
                            </div>
                        ))}
                    </div>
                </aside>
            </div>
        </>
    )
}

function Metric({ label, value, bold = false, className = '' }) {
    return (
        <div>
            <p className="text-[8px] text-neutral-500 uppercase tracking-wider">{label}</p>
            <p className={`${bold ? 'text-xs font-bold text-white' : 'text-[11px] font-medium text-neutral-200'} ${className} truncate`}>{value}</p>
        </div>
    )
}
