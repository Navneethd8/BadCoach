# SEO Foundations — IsoCourt

Canonical site: https://www.isocourt.fit (apex redirects here)

## Product
- What it does: Web app that analyzes badminton video (upload or live camera) and returns pose skeletons, stroke labels, quality scores, and short coaching tips.
- Who it's for: Recreational and club badminton players who already film themselves, plus coaches who want faster clip review.
- The pain it solves: You know something felt off in a rally, but scrubbing phone video and guessing form burns practice time.
- The wedge: Research-backed stroke models plus live court feedback in the browser (Birdzo is the coaching persona inside IsoCourt). Not a generic "AI sports" wrapper.
- Words the site uses: IsoCourt, Birdzo, pose tracing, stroke reads, coaching notes, analyze, live, rally, clip.

## Competitors
| Competitor | How we found them | Their strength | Our advantage |
|------------|------------------|----------------|---------------|
| BadmintonPeak | "AI badminton video analysis" | Clear "one instruction" coaching loop, courses + paid tiers | We ship pose timeline + live session in-browser |
| Kreeda | "AI badminton coaching phone" | Mobile product, dashboards, community | Web-first analyze + live; research model stack |
| AI Sports Trainer | "AI badminton form analysis" | Fast form scores, drill suggestions | Stroke taxonomy + skeleton frames tied to timestamps |
| BadPro+ | "AI badminton biomechanics" | Deep biomechanics / federation positioning | Lighter path: drop a clip, get cues now |
| Bdmntn (iOS) | App Store badminton AI | On-device privacy, match stats | Cross-platform web, live coaching |
| GGAB | "AI badminton performance" | Match tracking / pro comparison narrative | Immediate stroke + pose feedback without beta waitlist framing |

## Search Landscape
| Query | Who ranks | Our position | Opportunity |
|-------|-----------|-------------|-------------|
| AI badminton coach | App/listicle mix | Not ranking (pre-deploy SEO) | FAQ + homepage entity clarity |
| badminton stroke analysis AI | Peak / trainer sites | No page | Explainer + Analyze landing |
| badminton pose estimation coaching | GitHub / research | Docs only offline | Glossary + how-it-works copy |
| BadmintonPeak alternative | Thin SERP | None | Comparison page (next) |
| live AI badminton feedback | Sparse | /live exists but not crawled well | Fix Vercel SPA + unique meta |

## Competitor Content Map (summary)
### BadmintonPeak
- Comparison pages: limited; strong product FAQ
- Blog/courses: courses attached to product
- Explainer pages: stroke-focused landing content
- E-E-A-T: coaching instructional tone

### Kreeda / AI Sports Trainer
- Mobile-app SEO pages, feature FAQs
- Pricing/community CTAs
- Less research-paper authority, more consumer app copy

## Starting Point (priority order)
1. Ship technical SEO (robots, sitemap, www canonical, SPA rewrites) — already built locally; deploy next
2. FAQ + glossary with FAQPage / DefinedTermSet JSON-LD — citation bait for AI Overviews
3. Clarify IsoCourt vs Birdzo on homepage (entity)
4. Comparison pages: IsoCourt vs BadmintonPeak / Kreeda (high intent)
5. Explainer: what is AI badminton stroke analysis
6. Connect GSC + GA for `/seo-briefing` (blocked until deploy + property access)

## Messaging Notes
- H1 leads with Birdzo; title/meta lead with IsoCourt. Fix by treating IsoCourt as product, Birdzo as coach persona (alternateName in schema).
- Landing voice is already strong (court slang, short lines). Keep it; don't corporatize.
- Manual workflow (coach phone review) is a real "competitor." Speak to that pain, not only other apps.
