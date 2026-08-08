/**
 * Post-build static HTML for crawlers / AI bots that weakly execute SPA JS.
 * Injects into #seo-static (sibling of #root) so React never mounts over it.
 * Keep the node hidden until main.jsx removes it after the SPA is ready —
 * never delete it with an inline script (that blanks the page before React paints).
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(__dirname, '..')
const dist = path.join(root, 'dist')
const templatePath = path.join(dist, 'index.html')

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function setMeta(html, attr, key, content) {
  const re = new RegExp(`<meta[^>]+${attr}=["']${key}["'][^>]*>`, 'i')
  const tag = `<meta ${attr}="${key}" content="${esc(content)}" />`
  if (re.test(html)) return html.replace(re, tag)
  return html.replace('</head>', `  ${tag}\n</head>`)
}

function setCanonical(html, url) {
  const re = /<link[^>]+rel=["']canonical["'][^>]*>/i
  const tag = `<link rel="canonical" href="${esc(url)}" />`
  if (re.test(html)) return html.replace(re, tag)
  return html.replace('</head>', `  ${tag}\n</head>`)
}

function wrapMain(inner, heading) {
  return `<main><article><h1>${esc(heading)}</h1>${inner}</article><p><a href="/analyze">Analyze a clip</a> · <a href="/live">Go live</a> · <a href="/faq">FAQ</a></p></main>`
}

async function main() {
  if (!fs.existsSync(templatePath)) {
    console.error('prerender-static: dist/index.html missing; run vite build first')
    process.exit(1)
  }

  const { FAQS } = await import(pathToFileURL(path.join(root, 'src/content/faq.js')).href)
  const { TERMS } = await import(pathToFileURL(path.join(root, 'src/content/glossary.js')).href)
  const { WHAT_IS, WHAT_IS_PATH } = await import(
    pathToFileURL(path.join(root, 'src/content/whatIs.js')).href
  )

  const origin = 'https://www.isocourt.fit'
  const template = fs.readFileSync(templatePath, 'utf8')

  const pages = [
    {
      route: '/',
      title: 'IsoCourt · AI Badminton Stroke Analysis',
      description:
        "Upload a badminton rally or stroke. IsoCourt's Birdzo coach reads footwork, contact, and shot type: pose tracing, scoring, and coaching tips.",
      // Interactive homepage: meta/canonical only — no #seo-static body (avoids layout/paint fights with hero + hover-trace).
      body: null,
    },
    {
      route: '/faq',
      title: 'FAQ · IsoCourt AI Badminton Coach',
      description:
        'Answers about IsoCourt, Birdzo, clip uploads, pose tracing, live coaching, privacy, and how AI badminton stroke analysis works.',
      body: wrapMain(
        FAQS.map(({ q, a }) => `<section><h2>${esc(q)}</h2><p>${esc(a)}</p></section>`).join('\n'),
        'Frequently asked',
      ),
    },
    {
      route: '/glossary',
      title: 'Glossary · IsoCourt Badminton Terms',
      description:
        'Plain-language glossary for IsoCourt: pose tracing, stroke reads, split-step, smash, clear, drop, quality score, Birdzo, and more.',
      body: wrapMain(
        `<dl>${TERMS.map(
          ({ term, def }) => `<div><dt>${esc(term)}</dt><dd>${esc(def)}</dd></div>`,
        ).join('')}</dl>`,
        'Court glossary',
      ),
    },
    {
      route: '/compare',
      title: 'IsoCourt vs BadmintonPeak vs Kreeda · AI Badminton Tools',
      description:
        'Compare IsoCourt, BadmintonPeak, and Kreeda for AI badminton coaching: browser pose analysis and live camera vs courses vs phone match metrics.',
      body: wrapMain(
        `<p>Three AI badminton tools. Different jobs: clip coach, course path, phone match app.</p>
         <p><strong>IsoCourt</strong>: browser pose + stroke timeline and live camera.</p>
         <p><strong>BadmintonPeak</strong>: course drills with step-by-step video correction.</p>
         <p><strong>Kreeda</strong>: phone match metrics, community, coach paths.</p>`,
        'IsoCourt vs alternatives',
      ),
    },
    {
      route: WHAT_IS_PATH,
      title: WHAT_IS.seoTitle,
      description: WHAT_IS.seoDescription,
      body: wrapMain(
        `<p>${esc(WHAT_IS.lead)}</p>` +
          WHAT_IS.sections
            .map(
              (s) =>
                `<section id="${esc(s.id)}"><h2>${esc(s.h)}</h2>${s.p
                  .map((para) => `<p>${esc(para)}</p>`)
                  .join('')}</section>`,
            )
            .join('\n'),
        WHAT_IS.heroTitleText,
      ),
    },
  ]

  for (const page of pages) {
    let html = template
    html = html.replace(/<title>[^<]*<\/title>/i, `<title>${esc(page.title)}</title>`)
    html = setMeta(html, 'name', 'description', page.description)
    html = setMeta(html, 'property', 'og:title', page.title)
    html = setMeta(html, 'property', 'og:description', page.description)
    html = setMeta(html, 'property', 'og:url', `${origin}${page.route === '/' ? '/' : page.route}`)
    html = setMeta(html, 'name', 'twitter:title', page.title)
    html = setMeta(html, 'name', 'twitter:description', page.description)
    html = setCanonical(html, `${origin}${page.route === '/' ? '/' : page.route}`)

    // Keep #root empty for React. Crawler copy lives in a sibling removed on boot.
    if (!html.includes('<div id="root"></div>')) {
      console.error('prerender-static: expected empty <div id="root"></div> in template')
      process.exit(1)
    }
    if (page.body) {
      // hidden + CSS clip: crawlers still see the markup; humans never get a blank flash
      // when the SPA takes over. main.jsx removes the node after first paint.
      html = html.replace(
        '<div id="root"></div>',
        `<div id="root"></div>\n  <div id="seo-static" hidden aria-hidden="true">${page.body}</div>`,
      )
    }

    if (page.route === '/') {
      fs.writeFileSync(path.join(dist, 'index.html'), html)
      console.log('prerender-static: /')
      continue
    }

    const outDir = path.join(dist, page.route.replace(/^\//, ''))
    fs.mkdirSync(outDir, { recursive: true })
    fs.writeFileSync(path.join(outDir, 'index.html'), html)
    console.log(`prerender-static: ${page.route}`)
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
