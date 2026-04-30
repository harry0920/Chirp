import { useAppStore } from '../../stores/appStore'
import { formatHotkey } from '../../lib/utils'
import { KeyBadge } from '../shared/KeyBadge'

export function TestDictationCard() {
  const hotkey = useAppStore((s) => s.hotkey)
  const hotkeyMode = useAppStore((s) => s.hotkeyMode)
  const keys = formatHotkey(hotkey)
  const verb = hotkeyMode === 'tap' ? 'Tap' : 'Hold'

  return (
    <section className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-8">
      <div className="flex flex-col items-start gap-5">
        <span className="font-geist text-[10px] font-medium uppercase tracking-[0.2em] text-white/45">
          Try it
        </span>
        <p className="font-geist text-[18px] leading-snug text-white/85">
          {verb} the hotkey from any app, speak, release.
          <span className="text-white/45"> Your text appears at the cursor.</span>
        </p>
        <div className="flex items-center gap-2">
          {keys.map((k) => (
            <KeyBadge key={k} keyLabel={k} variant="glass" />
          ))}
        </div>
        <p className="font-geist text-[12px] text-white/35">
          Stats and recents appear here after your first dictation.
        </p>
      </div>
    </section>
  )
}
