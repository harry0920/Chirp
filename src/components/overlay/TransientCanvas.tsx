import { Sparkline } from '../shared/Sparkline'

type Mode = 'listening' | 'polishing' | 'error'

interface Props {
  mode: Mode
  amplitudes: number[]
  dismissing: boolean
}

const SAMPLE_COUNT = 48

function shapedAmplitudes(amps: number[]): number[] {
  const len = amps.length
  if (len === 0) return Array(SAMPLE_COUNT).fill(0.05)
  const out: number[] = new Array(SAMPLE_COUNT)
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    const srcIdx = Math.round((i / (SAMPLE_COUNT - 1)) * (len - 1))
    const raw = amps[srcIdx] ?? 0
    out[i] = Math.max(0.04, Math.sqrt(raw))
  }
  return out
}

export function TransientCanvas({ mode, amplitudes, dismissing }: Props) {
  const isListening = mode === 'listening'
  const isError = mode === 'error'

  return (
    <div
      className={`flex items-center justify-center transition-opacity duration-200 ${
        dismissing ? 'opacity-0' : 'opacity-100'
      }`}
      aria-live="polite"
    >
      {isListening ? (
        <Sparkline
          data={shapedAmplitudes(amplitudes)}
          width={200}
          height={28}
          strokeWidth={1.25}
          dotRadius={2.75}
          padding={3}
        />
      ) : (
        <ThreeDots color={isError ? '#EF4444' : '#FFFFFF'} />
      )}
    </div>
  )
}

function ThreeDots({ color }: { color: string }) {
  return (
    <div className="flex items-center gap-[6px]">
      <Dot color={color} delay={0} />
      <Dot color={color} delay={150} />
      <Dot color={color} delay={300} />
    </div>
  )
}

function Dot({ color, delay }: { color: string; delay: number }) {
  return (
    <span
      style={{
        width: 5,
        height: 5,
        borderRadius: 9999,
        backgroundColor: color,
        animation: 'overlay-dot-pulse 900ms ease-in-out infinite',
        animationDelay: `${delay}ms`,
      }}
    />
  )
}
