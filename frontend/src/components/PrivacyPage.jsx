import LegalLayout from './LegalLayout'

export default function PrivacyPage() {
    return (
        <LegalLayout title="Privacy">
            <p>
                IsoCourt processes the <strong className="text-neutral-200">video clips you upload</strong> so we can run pose and stroke analysis. Clips are
                handled in line with how our backend is configured (retention and storage may change; we aim to keep this page accurate).
            </p>
            <p>
                If you use the <strong className="text-neutral-200">feedback form</strong>, we receive the name, email, and message you submit so we can reply.
            </p>
            <p>
                This site uses <strong className="text-neutral-200">Google Analytics</strong> (see Google&apos;s privacy policy for how they process data). You can
                use browser controls or extensions to limit tracking.
            </p>
            <p className="text-neutral-500 text-xs pt-4">
                This is a summary, not legal advice. For deletion requests or questions, contact us through the feedback form on the home page.
            </p>
        </LegalLayout>
    )
}
