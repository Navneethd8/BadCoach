import { copyFileSync, mkdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const outDir = join(root, 'out')
const destDir = join(root, '..', 'frontend', 'public', 'demo-videos')

mkdirSync(destDir, { recursive: true })

const files = [
    'full-flow.mp4',
    '01-upload.mp4',
    '02-analyzing.mp4',
    '03-results.mp4',
]

for (const name of files) {
    copyFileSync(join(outDir, name), join(destDir, name))
    console.log(`copied ${name} → frontend/public/demo-videos/`)
}
