type Mode = 'listening' | 'polishing' | 'error'

interface Props {
  mode: Mode
  amplitudes: number[]
  dismissing: boolean
}

const BAR_COUNT = 12
// Three evenly-distributed bar indices that stay as dots when the
// waveform liquid-morphs. Other bars collapse to zero width.
const DOT_INDICES = [2, 6, 10] as const

function ampForBar(amplitudes: number[], i: number): number {
  if (amplitudes.length === 0) return 0.04
  const srcIdx = Math.round((i / (BAR_COUNT - 1)) * (amplitudes.length - 1))
  return amplitudes[srcIdx] ?? 0.04
}

// Listening = snappy follow-the-amplitude (60ms) so the waveform feels
// alive, not lagged. Morph = the full liquid transition into dots.
const LISTENING_TRANSITION = 'height 60ms linear'
const MORPH_TRANSITION =
  'height 420ms cubic-bezier(0.65, 0, 0.35, 1), width 420ms cubic-bezier(0.65, 0, 0.35, 1), opacity 280ms ease-out'

export function TransientCanvas({ mode, amplitudes, dismissing }: Props) {
  const isListening = mode === 'listening'
  const color = mode === 'error' ? '#EF4444' : '#FFFFFF'

  return (
    <div
      className={`flex h-9 items-center rounded-full border border-white/10 bg-black/70 px-4 backdrop-blur-xl transition-opacity duration-200 ${
        dismissing ? 'opacity-0' : 'opacity-100'
      }`}
      style={{ boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.04)' }}
    >
      <div className="flex h-6 items-center gap-[5px]">
        {Array.from({ length: BAR_COUNT }).map((_, i) => {
          const dotIdx = DOT_INDICES.indexOf(i as typeof DOT_INDICES[number])
          const isDotPosition = dotIdx >= 0

          // Steeper response curve (^0.45 instead of sqrt) so quiet
          // speech still produces visible movement and louder peaks
          // hit ceiling fast — the waveform reads more responsive.
          const amp = ampForBar(amplitudes, i)
          const liveHeight = Math.max(3, Math.pow(amp, 0.45) * 24)

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
