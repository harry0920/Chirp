import type { AppPoint } from './types'

interface Props {
  apps: AppPoint[]
}

export function WhereYouDictate({ apps }: Props) {
  const top = apps.slice(0, 5)
  const max = Math.max(1, ...top.map((a) => a.count))

  return (
    <article className="rounded-2xl border border-white/[0.06] bg-white/[0.02] p-5">
      <header className="mb-4 flex items-center justify-between">
        <span className="font-geist text-[10px] font-medium uppercase tracking-[0.2em] text-white/45">
          Where you dictate
        </span>
      </header>

      {top.length === 0 ? (
        <div className="font-geist text-[12px] text-white/35">
          No app context yet.
        </div>
      ) : (
        <ul className="flex flex-col gap-3">
          {top.map((app, i) => (
            <li key={app.raw} className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between font-geist text-[12px]">
                <span className="text-white/85">{app.display}</span>
                <span
                  className="font-geist-mono text-[11px] text-white/45"
                  style={{ fontFeatureSettings: '"tnum"' }}
                >
                  {Math.round(app.percent)}%
                </span>
              </div>
              <div className="relative h-[2px] w-full bg-white/[0.06]">
                <div
                  className="absolute inset-y-0 left-0 bg-white"
                  style={{ width: `${(app.count / max) * 100}%` }}
                />
                {i === 0 && (
                  <div
                    className="absolute -top-[1.5px] h-[5px] w-[5px] rounded-full bg-chirp-yellow"
                    style={{
                      left: `calc(${(app.count / max) * 100}% - 2.5px)`,
                    }}
                  />
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </article>
  )
}
