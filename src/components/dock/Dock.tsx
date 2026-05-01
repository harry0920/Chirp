import { BookText, Home, Settings as SettingsIcon, Zap } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { useAppStore } from '../../stores/appStore'
import { BirdMark } from '../shared/BirdMark'

interface DockItem {
  id: string
  label: string
  icon: LucideIcon
}

const DOCK_ITEMS: DockItem[] = [
  { id: 'home', label: 'Home', icon: Home },
  { id: 'vocabulary', label: 'Vocabulary', icon: BookText },
  { id: 'snippets', label: 'Snippets', icon: Zap },
  { id: 'settings', label: 'Settings', icon: SettingsIcon },
]

export function Dock() {
  const settingsPage = useAppStore((s) => s.settingsPage)
  const setSettingsPage = useAppStore((s) => s.setSettingsPage)

  return (
    <div className="pointer-events-none fixed bottom-6 left-1/2 z-40 -translate-x-1/2">
      <div
        className="pointer-events-auto flex items-center gap-1 rounded-full border border-white/10 bg-black/70 p-1.5 backdrop-blur-xl"
        style={{ boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.04)' }}
      >
        <button
          type="button"
          onClick={() => setSettingsPage('home')}
          aria-label="chirp"
          className="group relative flex h-10 w-10 items-center justify-center rounded-full transition-colors duration-200 active:scale-95 hover:bg-white/[0.04]"
        >
          <BirdMark size={18} color="#F0B723" />
        </button>

        <div className="mx-1 h-5 w-px bg-white/10" aria-hidden />

        {DOCK_ITEMS.map(({ id, label, icon: Icon }) => {
          const active = settingsPage === id
          return (
            <button
              key={id}
              type="button"
              onClick={() => setSettingsPage(id)}
              aria-label={label}
              aria-current={active ? 'page' : undefined}
              title={label}
              className={`group relative flex h-10 w-10 items-center justify-center rounded-full transition-colors duration-200 active:scale-95 ${
                active
                  ? 'halo-active bg-white/[0.06] text-white'
                  : 'text-white/45 hover:bg-white/[0.04] hover:text-white/85'
              }`}
            >
              <Icon
                size={18}
                strokeWidth={1.75}
                className="transition-transform duration-200 group-hover:scale-110"
              />
            </button>
          )
        })}
      </div>
    </div>
  )
}
