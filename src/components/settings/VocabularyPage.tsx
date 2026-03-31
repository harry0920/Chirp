import { useState } from 'react'
import { trackEvent } from '@aptabase/tauri'
import { useAppStore } from '../../stores/appStore'
import { Button } from '../shared/Button'
import { Toggle } from '../shared/Toggle'

export function VocabularyPage() {
  const vocabulary = useAppStore((s) => s.vocabulary)
  const addEntry = useAppStore((s) => s.addVocabularyEntry)
  const removeEntry = useAppStore((s) => s.removeVocabularyEntry)
  const beamSearch = useAppStore((s) => s.beamSearch)
  const updateSettings = useAppStore((s) => s.updateSettings)

  const [newWord, setNewWord] = useState('')

  const handleAdd = () => {
    const word = newWord.trim()
    if (!word) return
    addEntry(word)
    setNewWord('')
    trackEvent('feature_used', { feature: 'vocabulary_add' })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleAdd()
  }

  return (
    <div className="flex flex-col gap-5 animate-slide-up">
      <div className="mb-1">
        <h1 className="font-display font-extrabold text-2xl text-[#1a1a1a] tracking-[-0.5px]">
          Vocabulary
        </h1>
        <p className="text-[13px] text-[#aaa] mt-1">
          Names and terms Chirp should recognize. These are passed to Smart Cleanup for correction.
        </p>
      </div>

      {/* Beam search toggle */}
      <div className="flex items-center justify-between rounded-card border border-card-border bg-white px-[18px] py-3.5">
        <div>
          <div className="text-[13px] font-medium text-[#1a1a1a]">Enhanced Recognition</div>
          <div className="text-[11px] text-[#aaa] mt-0.5">
            Uses beam search for better accuracy with accents and noise. Slightly slower.
          </div>
        </div>
        <Toggle
          checked={beamSearch}
          onChange={(v) => updateSettings({ beamSearch: v })}
        />
      </div>

      {vocabulary.length > 0 ? (
        <div className="overflow-hidden rounded-card border border-card-border">
          {/* Header */}
          <div className="flex bg-[#FAFAF8] px-[18px] py-2.5">
            <span className="flex-1 text-[11px] font-semibold uppercase tracking-[0.5px] text-[#aaa]">
              Word / Phrase
            </span>
            <span className="w-10" />
          </div>

          {/* Rows */}
          {vocabulary.map((entry, i) => (
            <div
              key={i}
              className={`flex items-center px-[18px] h-11 border-b border-[#F5F4F0] last:border-b-0 transition-colors hover:bg-[#FAFAF8] group animate-slide-up ${
                i % 2 === 0 ? 'bg-white' : 'bg-[#FAFAF8]/50'
              }`}
              style={{ animationDelay: `${i * 30}ms` }}
            >
              <span className="flex-1 text-[13px] text-[#333]">
                {entry.word}
              </span>
              <button
                onClick={() => removeEntry(i)}
                className="flex h-8 w-10 items-center justify-center text-[#ccc] hover:text-chirp-error transition-colors duration-150 opacity-0 group-hover:opacity-100"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center justify-center rounded-card border border-dashed border-card-border bg-[#FAFAF8] px-6 py-10">
          <p className="text-[13px] text-[#aaa] text-center">
            No entries yet. Add names, companies, and terms Chirp should recognize.
          </p>
        </div>
      )}

      {/* Add row */}
      <div className="flex items-center gap-3">
        <input
          type="text"
          value={newWord}
          onChange={(e) => setNewWord(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Add a word or phrase..."
          className="flex-1 h-10 rounded-lg border border-card-border bg-white px-3 text-[13px] text-[#333] placeholder:text-[#ccc] focus:border-chirp-yellow focus:shadow-[0_0_0_3px_rgba(240,183,35,0.1)] focus:outline-none transition-all duration-150"
        />
        <Button onClick={handleAdd} disabled={!newWord.trim() || vocabulary.length >= 500}>
          Add
        </Button>
      </div>

      {vocabulary.length >= 450 && (
        <p className="text-xs text-chirp-error">
          You're approaching the maximum of 500 entries ({vocabulary.length}/500).
        </p>
      )}
    </div>
  )
}
