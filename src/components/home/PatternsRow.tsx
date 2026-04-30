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
    <section className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <WhenYouDictate hourlyGrid={hourlyGrid} />
      <WhereYouDictate apps={topApps} />
      <WhatYouSay
        totalWords={totalWords}
        totalSessions={totalSessions}
        longestDictationWords={longestDictationWords}
        totalDurationMs={totalDurationMs}
      />
    </section>
  )
}
