import { AlertTriangle } from 'lucide-react'
import type { AttentionItem } from './types'

interface Props {
  items: AttentionItem[]
  onAction: (item: AttentionItem) => void
}

export function AttentionStrip({ items, onAction }: Props) {
  if (items.length === 0) return null

  return (
    <div className="animate-slide-up flex items-center gap-3 rounded-xl border border-chirp-yellow/30 bg-chirp-yellow/[0.05] px-4 py-2.5">
      <AlertTriangle size={14} className="shrink-0 text-chirp-yellow" />
      <ul className="flex flex-1 flex-wrap items-center gap-x-3 gap-y-1 font-geist text-[12px] text-white/85">
        {items.map((item, i) => (
          <li key={item.id} className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => onAction(item)}
              className="text-left transition-colors hover:text-chirp-yellow"
            >
              {item.message}
            </button>
            {i < items.length - 1 && <span className="text-white/20">·</span>}
          </li>
        ))}
      </ul>
    </div>
  )
}
