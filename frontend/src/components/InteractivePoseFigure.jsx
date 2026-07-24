import { useEffect, useRef } from 'react'

const SOURCE = '/marketing/pose-trace-hero.webp'
const SAMPLE_STEP = 5
const ALPHA_THRESHOLD = 28
const INFLUENCE_RADIUS = 84
const BRAND_RGB = [16, 185, 129]

function mix(from, to, amount) {
    return Math.round(from + (to - from) * amount)
}

function isPoseGreen(r, g, b) {
    return g - r > 5 && g > r * 1.25 && g > b * 1.15
}

function getNeutralNeighbor(pixels, width, height, x, y) {
    for (let radius = 2; radius <= 8; radius += 2) {
        for (let offsetY = -radius; offsetY <= radius; offsetY += 2) {
            for (let offsetX = -radius; offsetX <= radius; offsetX += 2) {
                const sampleX = x + offsetX
                const sampleY = y + offsetY
                if (sampleX < 0 || sampleX >= width || sampleY < 0 || sampleY >= height) continue

                const index = (sampleY * width + sampleX) * 4
                const r = pixels[index]
                const g = pixels[index + 1]
                const b = pixels[index + 2]
                const alpha = pixels[index + 3]
                if (alpha >= ALPHA_THRESHOLD && !isPoseGreen(r, g, b)) {
                    return { r, g, b, a: alpha / 255 }
                }
            }
        }
    }

    return null
}

export default function InteractivePoseFigure() {
    const canvasRef = useRef(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return undefined

        const context = canvas.getContext('2d', { alpha: true })
        const sourceCanvas = document.createElement('canvas')
        const sourceContext = sourceCanvas.getContext('2d', { willReadFrequently: true })
        const image = new Image()
        const pointer = { x: -1000, y: -1000, strength: 0, target: 0 }
        const particles = []
        let frameId = 0
        let resizeObserver
        let ready = false
        let reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

        const draw = () => {
            if (!ready) return

            const rect = canvas.getBoundingClientRect()
            const width = rect.width
            const height = rect.height
            const dpr = Math.min(window.devicePixelRatio || 1, 2)

            if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
                canvas.width = Math.round(width * dpr)
                canvas.height = Math.round(height * dpr)
            }

            context.setTransform(dpr, 0, 0, dpr, 0, 0)
            context.clearRect(0, 0, width, height)

            const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight)
            const offsetX = (width - image.naturalWidth * scale) / 2
            const offsetY = (height - image.naturalHeight * scale) / 2
            const baseSize = Math.max(1.35, SAMPLE_STEP * scale * 0.82)

            for (const particle of particles) {
                const baseX = offsetX + particle.x * scale
                const baseY = offsetY + particle.y * scale
                const dx = baseX - pointer.x
                const dy = baseY - pointer.y
                const distance = Math.hypot(dx, dy)
                const proximity = reducedMotion
                    ? 0
                    : Math.max(0, 1 - distance / INFLUENCE_RADIUS) * pointer.strength
                const direction = Math.atan2(dy, dx)
                const jitter = Math.sin(particle.x * 0.31 + particle.y * 0.17) * proximity
                const displacement = proximity * proximity * (12 + jitter * 4)
                const x = baseX + Math.cos(direction) * displacement
                const y = baseY + Math.sin(direction) * displacement
                const colorMix = Math.min(0.82, proximity * 0.88)
                const size = baseSize * (1 + proximity * 0.7)

                context.fillStyle = `rgba(${mix(particle.r, BRAND_RGB[0], colorMix)}, ${mix(
                    particle.g,
                    BRAND_RGB[1],
                    colorMix,
                )}, ${mix(particle.b, BRAND_RGB[2], colorMix)}, ${particle.a})`
                context.fillRect(x - size / 2, y - size / 2, size, size)
            }

            pointer.strength += (pointer.target - pointer.strength) * 0.14
            if (Math.abs(pointer.target - pointer.strength) > 0.01) {
                frameId = requestAnimationFrame(draw)
            } else {
                pointer.strength = pointer.target
            }
        }

        const queueDraw = () => {
            cancelAnimationFrame(frameId)
            frameId = requestAnimationFrame(draw)
        }

        const updatePointer = (event) => {
            const rect = canvas.getBoundingClientRect()
            pointer.x = event.clientX - rect.left
            pointer.y = event.clientY - rect.top
            pointer.target = 1
            queueDraw()
        }

        const releasePointer = () => {
            pointer.target = 0
            queueDraw()
        }

        const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
        const handleMotionPreference = (event) => {
            reducedMotion = event.matches
            queueDraw()
        }

        image.onload = () => {
            sourceCanvas.width = image.naturalWidth
            sourceCanvas.height = image.naturalHeight
            sourceContext.drawImage(image, 0, 0)
            const pixels = sourceContext.getImageData(
                0,
                0,
                image.naturalWidth,
                image.naturalHeight,
            ).data

            for (let y = 0; y < image.naturalHeight; y += SAMPLE_STEP) {
                for (let x = 0; x < image.naturalWidth; x += SAMPLE_STEP) {
                    const index = (y * image.naturalWidth + x) * 4
                    const alpha = pixels[index + 3]
                    if (alpha < ALPHA_THRESHOLD) continue

                    const sourceR = pixels[index]
                    const sourceG = pixels[index + 1]
                    const sourceB = pixels[index + 2]
                    const poseGreen = isPoseGreen(sourceR, sourceG, sourceB)
                    const neutral = poseGreen
                        ? getNeutralNeighbor(
                              pixels,
                              image.naturalWidth,
                              image.naturalHeight,
                              x,
                              y,
                          )
                        : null

                    if (poseGreen && !neutral) continue

                    particles.push({
                        x,
                        y,
                        r: neutral?.r ?? sourceR,
                        g: neutral?.g ?? sourceG,
                        b: neutral?.b ?? sourceB,
                        a: neutral?.a ?? alpha / 255,
                    })
                }
            }

            ready = true
            resizeObserver = new ResizeObserver(queueDraw)
            resizeObserver.observe(canvas)
            queueDraw()
        }

        image.src = SOURCE
        canvas.addEventListener('pointerenter', updatePointer)
        canvas.addEventListener('pointermove', updatePointer)
        canvas.addEventListener('pointerdown', updatePointer)
        canvas.addEventListener('pointerleave', releasePointer)
        canvas.addEventListener('pointerup', releasePointer)
        canvas.addEventListener('pointercancel', releasePointer)
        motionQuery.addEventListener('change', handleMotionPreference)

        return () => {
            cancelAnimationFrame(frameId)
            resizeObserver?.disconnect()
            canvas.removeEventListener('pointerenter', updatePointer)
            canvas.removeEventListener('pointermove', updatePointer)
            canvas.removeEventListener('pointerdown', updatePointer)
            canvas.removeEventListener('pointerleave', releasePointer)
            canvas.removeEventListener('pointerup', releasePointer)
            canvas.removeEventListener('pointercancel', releasePointer)
            motionQuery.removeEventListener('change', handleMotionPreference)
            image.onload = null
        }
    }, [])

    return (
        <figure className="pixel-pose">
            <canvas
                ref={canvasRef}
                className="pixel-pose__canvas"
                role="img"
                aria-label="Pixelated athlete mid-smash with pose skeleton overlay"
            />
            <figcaption className="pixel-pose__hint" aria-hidden>
                <span className="pixel-pose__hint-dot" />
                hover to trace
            </figcaption>
        </figure>
    )
}
