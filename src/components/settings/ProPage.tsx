import { useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { Check, Cloud, Gauge, ListChecks, NotebookTabs, Smartphone, Sparkles, WandSparkles } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { Button } from '../shared/Button'

const PRO_FEATURE_OPTIONS: Array<{
  id: string
  label: string
  description: string
  icon: LucideIcon
}> = [
  {
    id: 'fast-cloud',
    label: 'Faster cloud transcription',
    description: 'Optional low-latency mode for long dictations and lighter hardware.',
    icon: Gauge,
  },
  {
    id: 'mobile',
    label: 'Mobile app',
    description: 'Dictate from your phone and keep your words with you across devices.',
    icon: Smartphone,
  },
  {
    id: 'meeting-notes',
    label: 'Meeting notes',
    description: 'Private transcripts, summaries, action items, and follow-ups.',
    icon: NotebookTabs,
  },
  {
    id: 'rewriter',
    label: 'Selected text rewriting',
    description: 'Highlight text anywhere, speak an edit, and replace it in place.',
    icon: WandSparkles,
  },
  {
    id: 'sync',
    label: 'Synced vocabulary and snippets',
    description: 'Keep personal words, names, and shortcuts consistent everywhere.',
    icon: Cloud,
  },
]

export function ProPage() {
  const [email, setEmail] = useState('')
  const [selected, setSelected] = useState<string[]>(['fast-cloud', 'mobile', 'meeting-notes'])
  const [status, setStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle')
  const [errorMsg, setErrorMsg] = useState('')

  const emailTrimmed = email.trim()
  const invalidEmail = emailTrimmed.length > 0 && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(emailTrimmed)

  const toggleFeature = (id: string) => {
    setSelected((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id]
    )
    if (status !== 'idle') setStatus('idle')
  }

  const handleSubmit = async () => {
    if (invalidEmail) return
    setStatus('sending')
    setErrorMsg('')

    const featureVotes = PRO_FEATURE_OPTIONS
      .filter((option) => selected.includes(option.id))
      .map((option) => option.label)

    const payload = [
      '[Chirp Pro interest]',
      `Email: ${emailTrimmed || 'not provided'}`,
      `Feature votes: ${featureVotes.length > 0 ? featureVotes.join(', ') : 'none selected'}`,
      'Positioning: local-first should remain the default; Pro is for optional speed, sync, meetings, and mobile.',
      'Source: Pro page',
    ].join('\n')

    try {
      await invoke('send_feedback', { text: payload })
      setStatus('sent')
      setEmail('')
    } catch (e) {
      setStatus('error')
      setErrorMsg(String(e))
    }
  }

  return (
    <div className="flex flex-col gap-6 pb-8">
      <section className="animate-slide-up rounded-card border border-card-border bg-card px-6 py-5">
        <div className="flex items-start justify-between gap-6">
          <div className="min-w-0">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-chirp-amber-400/30 bg-chirp-amber-400/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.8px] text-chirp-amber-600">
              <Sparkles size={13} strokeWidth={2} />
              Optional cloud
            </div>
            <h1 className="font-display text-[28px] font-black leading-tight text-dm-primary">
              Chirp Pro
            </h1>
            <p className="mt-2 max-w-[680px] text-[14px] leading-relaxed text-dm-secondary">
              Chirp stays local-first. Pro is an optional layer for speed, sync,
              meeting notes, and mobile workflows that need more than one device.
            </p>
          </div>
          <div className="hidden shrink-0 rounded-xl border border-card-border bg-card-hover px-4 py-3 text-right lg:block">
            <div className="text-[11px] font-semibold uppercase tracking-[0.8px] text-dm-secondary">
              Priority
            </div>
            <div className="mt-1 text-[13px] font-semibold text-dm-primary">
              Faster, still private by default
            </div>
          </div>
        </div>
      </section>

      <section className="animate-slide-up stagger-2">
        <div className="mb-2 flex items-end justify-between gap-4">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.8px] text-dm-secondary">
              What should we build first?
            </div>
            <div className="mt-1 text-[13px] text-dm-secondary">
              Pick the workflows that would make Chirp meaningfully better for you.
            </div>
          </div>
          <div className="hidden items-center gap-1.5 rounded-full border border-card-border bg-card px-3 py-1 text-[12px] text-dm-secondary sm:flex">
            <ListChecks size={14} />
            {selected.length} selected
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {PRO_FEATURE_OPTIONS.map((option) => {
            const active = selected.includes(option.id)
            const Icon = option.icon
            return (
              <button
                key={option.id}
                type="button"
                onClick={() => toggleFeature(option.id)}
                className={`flex min-h-[112px] items-start gap-4 rounded-card border p-4 text-left transition-all ${
                  active
                    ? 'border-chirp-amber-400 bg-chirp-amber-400/10 shadow-[0_8px_22px_rgba(240,183,35,0.12)]'
                    : 'border-card-border bg-card hover:border-chirp-amber-400/45 hover:bg-card-hover'
                }`}
              >
                <span
                  className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border ${
                    active
                      ? 'border-chirp-amber-400 bg-chirp-amber-400 text-chirp-stone-900'
                      : 'border-card-border bg-card-hover text-dm-primary'
                  }`}
                >
                  <Icon size={17} strokeWidth={2} />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center justify-between gap-3">
                    <span className="text-[14px] font-semibold leading-snug text-dm-primary">
                      {option.label}
                    </span>
                    {active && (
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-chirp-amber-400 text-chirp-stone-900">
                        <Check size={13} strokeWidth={2.5} />
                      </span>
                    )}
                  </span>
                  <span className="mt-1 block text-[13px] leading-relaxed text-dm-secondary">
                    {option.description}
                  </span>
                </span>
              </button>
            )
          })}
        </div>
      </section>

      <section className="animate-slide-up stagger-3 rounded-card border border-card-border bg-card p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-[520px]">
            <div className="text-[15px] font-semibold text-dm-primary">
              Join the Pro early access list
            </div>
            <div className="mt-1 text-[13px] leading-relaxed text-dm-secondary">
              Add an email for early access, or leave it blank and only send your feature votes.
            </div>
          </div>
          <div className="w-full lg:w-[420px]">
            <div className="flex items-start gap-2">
              <div className="min-w-0 flex-1">
                <input
                  type="email"
                  value={email}
                  onChange={(e) => {
                    setEmail(e.target.value)
                    if (status !== 'idle') setStatus('idle')
                  }}
                  placeholder="Email for early access, optional"
                  className="h-10 w-full rounded-lg border border-card-border bg-dm-input px-3 font-body text-[13px] text-dm-primary placeholder:text-dm-muted transition-all duration-150 focus:border-chirp-yellow focus:shadow-[0_0_0_3px_rgba(240,183,35,0.16)] focus:outline-none"
                />
                <div className={`mt-1 min-h-[16px] text-[11px] ${status === 'error' || invalidEmail ? 'text-chirp-error' : 'text-dm-secondary'}`}>
                  {invalidEmail
                    ? 'Enter a valid email or leave it blank to only vote.'
                    : status === 'sent'
                      ? 'Thanks. Your interest was sent.'
                      : status === 'error'
                        ? errorMsg
                        : 'No account required. Chirp local remains the default.'}
                </div>
              </div>
              <Button
                onClick={handleSubmit}
                disabled={status === 'sending' || invalidEmail}
                className="h-10 whitespace-nowrap"
              >
                {status === 'sending'
                  ? 'Sending...'
                  : emailTrimmed
                    ? 'Join list'
                    : 'Send votes'}
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
