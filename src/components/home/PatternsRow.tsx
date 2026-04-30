import type { AppPoint } from './types'
import { WhenYouDictate } from './WhenYouDictate'
import { WhereYouDictate } from './WhereYouDictate'
import { WhatYouSay } from './WhatYouSay'

interface Props {
  hourlyGrid: number[][]
  topApps: AppPoint[]
  totalWords: number
  totalSessions: number
  longestDictationWords: number
  totalDurationMs: number
}

export function PatternsRow({
  hourlyGrid,
  topApps,
  totalWords,
  totalSessions,
  longestDictationWords,
  totalDurationMs,
}: Props) {
  return (
    <section className="grid grid-cols-1 items-stretch gap-4 lg:grid-cols-3">
      <div className="animate-slide-up stagger-1 flex">
        <WhenYouDictate hourlyGrid={hourlyGrid} />
      </div>
      <div className="animate-slide-up stagger-2 flex">
        <WhereYouDictate apps={topApps} />
      </div>
      <div className="animate-slide-up stagger-3 flex">
        <WhatYouSay
          totalWords={totalWords}
          totalSessions={totalSessions}
          longestDictationWords={longestDictationWords}
          totalDurationMs={totalDurationMs}
        />
      </div>
    </section>
  )
}
