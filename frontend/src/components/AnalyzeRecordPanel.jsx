import AnalyzeMobileRecordPanel from './AnalyzeMobileRecordPanel'
import AnalyzeDesktopRecordPanel from './AnalyzeDesktopRecordPanel'

export default function AnalyzeRecordPanel({ deviceType, mobileProps, desktopProps }) {
    return (
        <div className="rounded-md overflow-hidden border border-neutral-800 min-h-[220px] flex flex-col bg-black relative">
            {deviceType === 'mobile' ? (
                <AnalyzeMobileRecordPanel {...mobileProps} />
            ) : (
                <AnalyzeDesktopRecordPanel {...desktopProps} />
            )}
        </div>
    )
}
