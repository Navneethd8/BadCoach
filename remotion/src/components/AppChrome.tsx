import { Logo } from "./Logo";

const BRAND = "#6c9c8d";
const HEADER_HEIGHT = 56;

export const AppChrome = ({
  children,
  active = "analyze",
}: {
  children: React.ReactNode;
  active?: "analyze" | "live";
}) => (
  <div
    style={{
      width: "100%",
      height: "100%",
      display: "flex",
      flexDirection: "column",
      backgroundColor: "#fafafa",
      fontFamily: "'Inter', system-ui, sans-serif",
      color: "#0a0a0a",
      overflow: "hidden",
    }}
  >
    <header
      style={{
        flexShrink: 0,
        position: "relative",
        zIndex: 100,
        backgroundColor: BRAND,
        height: HEADER_HEIGHT,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 20px",
        boxShadow: "0 1px 0 rgba(0, 0, 0, 0.08)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <Logo size={24} color="#fafafa" />
        <span
          style={{
            fontFamily: "'Iosevka Charon Mono', monospace",
            fontWeight: 700,
            fontSize: 16,
            color: "#fafafa",
          }}
        >
          IsoCourt
        </span>
      </div>
      <nav style={{ display: "flex", gap: 20 }}>
        {(["analyze", "live"] as const).map((item) => (
          <span
            key={item}
            style={{
              fontFamily: "'Iosevka Charon Mono', monospace",
              fontSize: 11,
              fontWeight: 600,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              color: active === item ? "#fff" : "rgba(250,250,250,0.88)",
              textDecoration: active === item ? "underline" : "none",
              textUnderlineOffset: 4,
            }}
          >
            {item === "analyze" ? "Analyze" : "Live"}
          </span>
        ))}
      </nav>
    </header>
    <main
      style={{
        flex: 1,
        minHeight: 0,
        position: "relative",
        overflow: "hidden",
        padding: "16px 20px 28px",
      }}
    >
      {children}
    </main>
  </div>
);
