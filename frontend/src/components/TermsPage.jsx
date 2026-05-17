import LegalLayout from './LegalLayout'

export default function TermsPage() {
    return (
        <LegalLayout title="Terms of use">
            <p>
                IsoCourt is provided <strong className="text-neutral-200">as-is</strong>. Analysis and tips are automated and for training insight only. They are
                not a substitute for a qualified coach or medical advice.
            </p>
            <p>
                You are responsible for how you use the product (including live camera features) and for any clips you upload. Do not upload content you do not
                have the right to share.
            </p>
            <p>
                We may change, limit, or discontinue the service. Continued use after changes means you accept the updated terms where required by law.
            </p>
            <p className="text-neutral-500 text-xs pt-4">
                This is a plain-language summary. For formal terms in your jurisdiction, consult a lawyer if you need binding legal documents.
            </p>
        </LegalLayout>
    )
}
