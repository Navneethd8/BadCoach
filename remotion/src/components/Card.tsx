import { Icon } from "./Icon";

export const Card = ({
  children,
  title,
  icon,
}: {
  children: React.ReactNode;
  title?: string;
  icon?: string;
}) => (
  <section
    style={{
      backgroundColor: "#fff",
      border: "1px solid #e5e5e5",
      borderRadius: 8,
      padding: 20,
      boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      marginBottom: 16,
    }}
  >
    {title && (
      <h2
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          margin: "0 0 14px",
          fontFamily: "'Iosevka Charon Mono', monospace",
          fontSize: 13,
          fontWeight: 700,
          letterSpacing: "0.04em",
          textTransform: "lowercase",
          color: "#404040",
        }}
      >
        {icon && <Icon name={icon} size={18} />}
        {title}
      </h2>
    )}
    {children}
  </section>
);
