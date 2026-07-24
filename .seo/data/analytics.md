# GA4 data — BadCoach property

Range: 2026-01-01 → 2026-07-23  
Sources: `Reports_snapshot.csv`, `Events_Event_name.csv`, `Pages_and_screens_Page_path_and_screen_class.csv`  
Pulled: 2026-07-23

## Headline
| Metric | Value |
|--------|------:|
| Active / new users | 413 / 413 |
| Avg engagement / active user | ~106s |
| Total events | 4,631 |

## Pages by path
| Path | Views | Users | Avg engagement / user | Bounce signal |
|------|------:|------:|----------------------:|---------------|
| `/` | 1,052 | 412 | ~43s | Main entry |
| `/analyze` | 258 | 108 | ~205s | Deep engagement |
| `/live` | 76 | 40 | ~91s | Secondary product |
| `/privacy` | 11 | 6 | ~26s | Thin |
| `/terms` | 7 | 5 | ~3s | Thin |
| `/theme-preview` | 4 | 1 | — | Dev leftover; noindex/remove |

Funnel approx: home users 412 → analyze 108 (~26%) → Stream Started 67 → Stream Complete 46 → Clip Analyzed 26 (legacy path).

## Top events
| Event | Count | Users |
|-------|------:|------:|
| page_view | 1,408 | 413 |
| session_start | 699 | 413 |
| user_engagement | 486 | 141 |
| scroll | 441 | 176 |
| Stream Window Received | 281 | 47 |
| Stream Started | 265 | 67 |
| Analyze_click | 150 | 84 |
| Stream Complete | 106 | 46 |
| Analysis Failed | 97 | 20 |
| Session Start Attempt (live) | 67 | 15 |
| Session Started (live) | 58 | 11 |
| Stream Error | 49 | 27 |
| Live_coaching_click | 48 | 31 |
| Clip Analyzed | 26 | 9 |
| Validation Failed | 14 | 4 |
| Feedback_sent | 9 | 2 |
| Camera Recording Captured | 6 | 4 |
| Capacity Reached | 1 | 1 |

## Acquisition (from snapshot)
| Source | First users | Sessions |
|--------|------------:|---------:|
| Direct | 310 | 449 |
| Google organic | 32 | 76 |
| Reddit | 31 | 48 |
| LinkedIn | 22 | 30 |
| Claude.ai | — | 28 |
| ChatGPT | ~6 | ~16 |
| Bing organic | 3 | 3 |

## Notes
- Event naming is inconsistent (`Analyze_click` vs `analyze_click` in code comments; `Live_coaching_click` vs lowercase). Standardize later.
- Analysis Failed (97) vs Stream Complete (106) is a high failure rate for people who start streams.
- Live: 48 clicks → 15 session attempts → 11 started (drop-off at camera/capacity).
- No `/faq`, `/glossary`, `/compare/*` traffic yet (pre-deploy / new pages).
