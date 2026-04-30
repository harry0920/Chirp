type Mode = 'listening' | 'polishing' | 'error'

interface Props {
  mode: Mode
  amplitudes: number[]
  dismissing: boolean
}

const BAR_COUNT = 8
// Three evenly-distributed bar indices that stay as dots when the
// waveform liquid-morphs. Other bars collapse to zero width.
const DOT_INDICES = [1, 4, 7] as const

function ampForBar(amplitudes: number[], i: number): number {
  if (amplitudes.length === 0) return 0.04
  const srcIdx = Math.round((i / (BAR_COUNT - 1)) * (amplitudes.length - 1))
  return amplitudes[srcIdx] ?? 0.04
}

// Listening = follow the voice envelope, not every micro-spike. 90ms
// ease-out interpolates between amplitude frames so the bars track the
// envelope instead of jittering on every tick. Morph = the full liquid
// transition into dots.
const LISTENING_TRANSITION = 'height 90ms ease-out'
const MORPH_TRANSITION =
  'height 420ms cubic-bezier(0.65, 0, 0.35, 1), width 420ms cubic-bezier(0.65, 0, 0.35, 1), opacity 280ms ease-out'

export function TransientCanvas({ mode, amplitudes, dismissing }: Props) {
  const isListening = mode === 'listening'
  const color = mode === 'error' ? '#EF4444' : '#FFFFFF'

  return (
    <div
      className={`flex items-center rounded-full border border-white/10 bg-black/70 px-5 py-3 backdrop-blur-xl transition-opacity duration-200 ${
        dismissing ? 'opacity-0' : 'opacity-100'
      }`}
      style={{ boxShadow: 'inset 0 1px 0 rgba(255, 255, 255, 0.04)' }}
    >
      <div className="flex h-4 items-center gap-[6px]">
        {Array.from({ length: BAR_COUNT }).map((_, i) => {
          const dotIdx = DOT_INDICES.indexOf(i as typeof DOT_INDICES[number])
          const isDotPosition = dotIdx >= 0

          // Gentler response curve (^0.55) plus a slight floor so
          // ambient noise and breath don't drive visible bars — the
          // waveform shapes the user's voice envelope, not every
          // micro-fluctuation. Max bar height capped at 16px so the
          // waveform reads slim against the pill.
          const amp = ampForBar(amplitudes, i)
          const liveHeight = Math.max(2, Math.pow(amp, 0.55) * 16)

          // Listening: bars driven by amplitudes.
          // Polishing/error: 3 dots remain (rounded square 4×4), others collapse.
          const height = isListening ? liveHeight : isDotPosition ? 4 : 0
          const width = isListening ? 2 : isDotPosition ? 4 : 0
          const opacity = isListening ? 1 : isDotPosition ? 1 : 0

          return (
            <span
              key={i}
              style={{
                display: 'block',
                height: `${height}px`,
                width: `${width}px`,
                opacity,
                backgroundColor: color,
                borderRadius: 9999,
                transition: isListening ? LISTENING_TRANSITION : MORPH_TRANSITION,
                animation:
                  !isListening && isDotPosition
                    ? 'overlay-dot-pulse 900ms ease-in-out infinite'
                    : undefined,
                animationDelay:
                  !isListening && isDotPosition ? `${dotIdx * 150}ms` : undefined,
              }}
            />
          )
        })}
      </div>
    </div>
  )
}
