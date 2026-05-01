import { useAppStore } from '../../stores/appStore'
import { useCleanupToggle } from '../../hooks/useCleanupToggle'
import { useCleanupStatusText } from '../../hooks/useCleanupStatusText'
import { TONE_MODES } from '../../lib/constants'
import { Toggle } from '../shared/Toggle'

type ToneId = (typeof TONE_MODES)[number]['id']

export function QuickActions() {
  return (
    <section className="grid grid-cols-1 items-stretch gap-4 lg:grid-cols-2">
      <div className="animate-slide-up stagger-1 flex">
        <SmartCleanupCard />
      </div>
      <div className="animate-slide-up stagger-2 flex">
        <ToneCard />
      </div>
    </section>
  )
}

function SmartCleanupCard() {
  const aiCleanup = useAppStore((s) => s.aiCleanup)
  const { handleCleanupToggle, cleanupStarting } = useCleanupToggle()
  const status = useCleanupStatusText(cleanupStarting)

  return (
    <article className="card-surface flex w-full flex-col p-5">
      <header className="mb-4 flex items-center justify-between">
        <span className="font-geist text-[10px] font-medium uppercase tracking-[0.2em] text-white/45">
          Smart Cleanup
        </span>
      </header>
      <div className="flex flex-1 items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <span className="font-geist text-[15px] font-medium text-white">
            {aiCleanup ? 'On' : 'Off'}
          </span>
          <span className="font-geist text-[12px] text-white/45">
            {aiCleanup
              ? status
              : 'Polish grammar and remove filler words'}
          </span>
        </div>
        <Toggle
          checked={aiCleanup}
          onChange={handleCleanupToggle}
          disabled={cleanupStarting}
        />
      </div>
    </article>
  )
}

function ToneCard() {
  const toneMode = useAppStore((s) => s.toneMode)
  const updateSettings = useAppStore((s) => s.updateSettings)
  const activeTone = TONE_MODES.find((t) => t.id === toneMode) ?? TONE_MODES[0]

  return (
    <article className="card-surface flex w-full flex-col p-5">
      <header className="mb-4 flex items-center justify-between">
        <span className="font-geist text-[10px] font-medium uppercase tracking-[0.2em] text-white/45">
          Tone
        </span>
      </header>
      <div className="flex flex-1 items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <span className="font-geist text-[15px] font-medium text-white">
            {activeTone.label}
          </span>
          <span className="font-geist text-[12px] text-white/45">
            {activeTone.description}
          </span>
        </div>
        <div className="flex items-center rounded-full border border-white/10 bg-white/[0.03] p-0.5">
          {TONE_MODES.map((tone) => {
            const active = tone.id === toneMode
            return (
              <button
                key={tone.id}
                type="button"
                onClick={() => updateSettings({ toneMode: tone.id as ToneId })}
                className={`rounded-full px-3 py-1 font-geist text-[12px] transition-colors ${
                  active
                    ? 'bg-white/[0.08] text-white'
                    : 'text-white/45 hover:text-white/85'
                }`}
              >
                {tone.label}
              </button>
            )
          })}
        </div>
      </div>
    </article>
  )
}
