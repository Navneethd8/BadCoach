import { useNavigate } from 'react-router-dom'

export default function LegalLayout({ title, children }) {
    const navigate = useNavigate()
    return (
        <div className="min-h-screen bg-neutral-950 text-neutral-100 px-6 py-10">
            <div className="max-w-2xl mx-auto">
                <button
                    type="button"
                    onClick={() => navigate(-1)}
                    className="text-sm text-emerald-400 hover:text-emerald-300 mb-8"
                >
                    ← Back
                </button>
                <h1 className="text-2xl font-bold text-white tracking-tight mb-6">{title}</h1>
                <div className="space-y-4 text-sm text-neutral-300 leading-relaxed">{children}</div>
            </div>
        </div>
    )
}
