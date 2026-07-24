import { Link } from 'react-router-dom'

export default function FigmaButton({
    children,
    variant = 'primary',
    href,
    onClick,
    className = '',
    disabled = false,
    loading = false,
    type = 'button',
}) {
    const classes = [
        'figma-cta',
        variant === 'primary' ? 'figma-cta--primary' : 'figma-cta--secondary',
        loading ? 'figma-cta--loading' : '',
        className,
    ]
        .filter(Boolean)
        .join(' ')

    const content = loading ? <span className="figma-cta-spinner" aria-hidden /> : children

    if (href && !disabled && !loading) {
        return (
            <Link to={href} className={classes} onClick={onClick}>
                {content}
            </Link>
        )
    }

    return (
        <button type={type} onClick={onClick} disabled={disabled || loading} className={classes}>
            {content}
        </button>
    )
}
