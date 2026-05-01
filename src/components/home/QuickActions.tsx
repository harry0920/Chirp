import { useAppStore } from '../../stores/appStore'
import { useCleanupToggle } from '../../hooks/useCleanupToggle'
import { useCleanupStatusText } from '../../hooks/useCleanupStatusText'
import { TONE_MODES } from '../../lib/constants'
import type { CleanupProvider } from '../../stores/appStore'

type ToneId = (typeof TONE_MODES)[number]['id']
type CleanupMode = 'off' | 'local' | 'cloud'

/** Cloud provider used when the user picks "Cloud" without a prior
 *  selection — we fall back to the first non-local entry. */
const DEFAULT_CLOUD_PROVIDER: Exclude<CleanupProvider, 'local'> = 'anthropic'

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
  const cleanupProvider = useAppStore((s) => s.cleanupProvider)
  const updateSettings = useAppStore((s) => s.updateSettings)
  const { handleCleanupToggle, cleanupStarting } = useCleanupToggle()
  const status = useCleanupStatusText(cleanupStarting)

  const currentMode: CleanupMode = !aiCleanup
    ? 'off'
    : cleanupProvider === 'local'
      ? 'local'
      : 'cloud'

  const handleSelect = async (mode: CleanupMode) => {
    if (mode === currentMode) return
    if (mode === 'off') {
      await handleCleanupToggle(false)
      return
    }
    if (mode === 'local') {
      // Switch provider first so the toggle handler sees the new
      // value when it decides whether to start the local LLM.
      if (cleanupProvider !== 'local') {
        updateSettings({ cleanupProvider: 'local' })
      }
      await handleCleanupToggle(true)
      return
    }
    // mode === 'cloud'
    const targetProvider: Exclude<CleanupProvider, 'local'> =
      cleanupProvider !== 'local'
        ? (cleanupProvider as Exclude<CleanupProvider, 'local'>)
        : DEFAULT_CLOUD_PROVIDER
    updateSettings({ cleanupProvider: targetProvider })
    await handleCleanupToggle(true)
  }

  const description =
    currentMode === 'off'
      ? 'Polish grammar and remove filler words'
      : status

  return (
    <article className="card-surface flex w-full flex-col p-5">
      <header className="mb-4 flex items-center justify-between">
        <span className="font-geist text-[10px] font-medium uppercase tracking-[0.2em] text-white/45">
          Smart Cleanup
        </span>
      </header>
      <div className="flex flex-1 items-center justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-1">
          <span className="font-geist text-[15px] font-medium text-white">
            {currentMode === 'off' ? 'Off' : currentMode === 'local' ? 'Local' : 'Cloud'}
          </span>
          <span className="truncate font-geist text-[12px] text-white/45">
            {description}
          </span>
        </div>
        <SegmentedControl
          options={[
            { id: 'off', label: 'Off' },
            { id: 'local', label: 'Local' },
            { id: 'cloud', label: 'Cloud' },
          ]}
          value={currentMode}
          onChange={handleSelect}
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
        <div className="flex min-w-0 flex-col gap-1">
          <span className="font-geist text-[15px] font-medium text-white">
            {activeTone.label}
          </span>
          <span className="truncate font-geist text-[12px] text-white/45">
            {activeTone.description}
          </span>
        </div>
        <SegmentedControl
          options={TONE_MODES.map((t) => ({ id: t.id, label: t.label }))}
          value={toneMode}
          onChange={(id) => updateSettings({ toneMode: id as ToneId })}
        />
      </div>
    </article>
  )
}

interface SegmentedControlProps<T extends string> {
  options: { id: T; label: string }[]
  value: T
  onChange: (id: T) => void
  disabled?: boolean
}

function SegmentedControl<T extends string>({ options, value, onChange, disabled }: SegmentedControlProps<T>) {
  return (
    <div
      className={`flex items-center rounded-full border border-white/10 bg-white/[0.03] p-0.5 ${
        disabled ? 'opacity-60' : ''
      }`}
    >
      {options.map((opt) => {
        const active = opt.id === value
        return (
          <button
            key={opt.id}
            type="button"
            onClick={() => onChange(opt.id)}
            disabled={disabled}
            className={`rounded-full px-3 py-1 font-geist text-[12px] transition-all duration-150 active:scale-95 disabled:cursor-not-allowed ${
              active
                ? 'bg-white/[0.1] text-white'
                : 'text-white/45 hover:text-white/85'
            }`}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}
