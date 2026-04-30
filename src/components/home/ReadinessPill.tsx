import { useEffect, useRef, useState } from 'react'
import { useAppStore } from '../../stores/appStore'
import { useCleanupStatusText } from '../../hooks/useCleanupStatusText'
import { formatHotkey } from '../../lib/utils'
import { KeyBadge } from '../shared/KeyBadge'

type Health = 'ready' | 'warning' | 'error'

export function ReadinessPill() {
  const hotkey = useAppStore((s) => s.hotkey)
  const inputDevice = useAppStore((s) => s.inputDevice)
  const model = useAppStore((s) => s.model)
  const modelDownloaded = useAppStore((s) => s.modelDownloaded)
  const aiCleanup = useAppStore((s) => s.aiCleanup)
  const cleanupStatus = useCleanupStatusText(false)

  const speechReady = !!modelDownloaded[model]
  const cleanupReady =
    !aiCleanup ||
    cleanupStatus === 'Active' ||
    cleanupStatus === 'Cloud key set'

  const health: Health = !speechReady ? 'error' : !cleanupReady ? 'warning' : 'ready'

  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onClickAway = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClickAway)
    return () => document.removeEventListener('mousedown', onClickAway)
  }, [open])

  const dotClass =
    health === 'ready'
      ? 'bg-chirp-success shadow-[0_0_8px_rgba(34,197,94,0.55)]'
      : health === 'warning'
        ? 'bg-chirp-yellow shadow-[0_0_8px_rgba(240,183,35,0.55)]'
        : 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.55)]'

  const hotkeyKeys = formatHotkey(hotkey)

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2.5 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 font-geist text-[11px] text-white/85 backdrop-blur-md transition-colors hover:bg-white/[0.06]"
      >
        <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
        <span className="uppercase tracking-[0.14em]">
          {health === 'ready' ? 'Ready' : health === 'warning' ? 'Degraded' : 'Not ready'}
        </span>
        <span className="text-white/20">·</span>
        <span className="flex items-center gap-1">
          {hotkeyKeys.map((k) => (
            <KeyBadge key={k} keyLabel={k} variant="glass" />
          ))}
        </span>
      </button>

      {open && (
        <div className="absolute right-0 top-full z-30 mt-2 w-[280px] rounded-2xl border border-white/10 bg-black/85 p-4 shadow-2xl backdrop-blur-xl animate-popup-in">
          <ul className="flex flex-col gap-2 font-geist text-[12px]">
            <FlyoutRow
              label="Microphone"
              value={inputDevice === 'default' ? 'System default' : inputDevice}
              ok
            />
            <FlyoutRow label="Speech model" value={speechReady ? 'Ready' : 'Not downloaded'} ok={speechReady} />
            <FlyoutRow
              label="Smart Cleanup"
              value={aiCleanup ? cleanupStatus : 'Off'}
              ok={cleanupReady}
            />
          </ul>
        </div>
      )}
    </div>
  )
}

function FlyoutRow({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <li className="flex items-center justify-between gap-3 border-b border-white/[0.06] pb-2 last:border-b-0 last:pb-0">
      <span className="text-white/55">{label}</span>
      <span className="flex items-center gap-2">
        <span
          className={`h-1.5 w-1.5 rounded-full ${
            ok ? 'bg-chirp-success' : 'bg-chirp-yellow'
          }`}
        />
        <span className="text-white/85">{value}</span>
      </span>
    </li>
  )
}
