import { useState } from 'react'
import { trackEvent } from '@aptabase/tauri'
import { ChevronDown } from 'lucide-react'
import { useAppStore } from '../../stores/appStore'
import { Button } from '../shared/Button'
import { Toggle } from '../shared/Toggle'

export function VocabularyPage() {
  const vocabulary = useAppStore((s) => s.vocabulary)
  const addEntry = useAppStore((s) => s.addVocabularyEntry)
  const removeEntry = useAppStore((s) => s.removeVocabularyEntry)
  const updateBoost = useAppStore((s) => s.updateVocabularyBoost)
  const beamSearch = useAppStore((s) => s.beamSearch)
  const updateSettings = useAppStore((s) => s.updateSettings)

  const [newWord, setNewWord] = useState('')
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)

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

  const boostLabel = (boost: number) => {
    if (boost <= 1.5) return 'Low'
    if (boost <= 3.5) return 'Medium'
    return 'High'
  }

  return (
    <div className="flex flex-col gap-5 animate-slide-up">
      <div className="mb-1">
        <h1 className="font-display font-extrabold text-2xl text-[#1a1a1a] tracking-[-0.5px]">
          Vocabulary
        </h1>
        <p className="text-[13px] text-[#aaa] mt-1">
          Words and names you want Chirp to recognize accurately.
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

      {/* Prerequisite banner */}
      {!beamSearch && vocabulary.length > 0 && (
        <div className="flex items-center justify-between rounded-card border border-chirp-amber-400/30 bg-chirp-amber-400/5 px-[18px] py-3">
          <span className="text-[12px] text-[#999]">
            Enable Enhanced Recognition to use your vocabulary during transcription.
          </span>
          <button
            onClick={() => updateSettings({ beamSearch: true })}
            className="text-[12px] font-medium text-chirp-amber-500 hover:text-chirp-amber-600 transition-colors"
          >
            Enable
          </button>
        </div>
      )}

      {vocabulary.length > 0 ? (
        <div className="overflow-hidden rounded-card border border-card-border">
          {/* Header */}
          <div className="flex bg-[#FAFAF8] px-[18px] py-2.5">
            <span className="flex-1 text-[11px] font-semibold uppercase tracking-[0.5px] text-[#aaa]">
              Word / Phrase
            </span>
            <span className="w-24 text-[11px] font-semibold uppercase tracking-[0.5px] text-[#aaa] text-right mr-10">
              Strength
            </span>
          </div>

          {/* Rows */}
          {vocabulary.map((entry, i) => (
            <div
              key={i}
              className={`border-b border-[#F5F4F0] last:border-b-0 transition-colors hover:bg-[#FAFAF8] group ${
                i % 2 === 0 ? 'bg-white' : 'bg-[#FAFAF8]/50'
              }`}
            >
              <div
                className="flex items-center px-[18px] h-11 animate-slide-up"
                style={{ animationDelay: `${i * 30}ms` }}
              >
                <span className="flex-1 text-[13px] text-[#333]">
                  {entry.word}
                </span>
                <button
                  onClick={() => setExpandedIndex(expandedIndex === i ? null : i)}
                  className="flex items-center gap-1 text-[11px] text-[#aaa] hover:text-[#666] transition-colors mr-2"
                >
                  {boostLabel(entry.boost)}
                  <ChevronDown
                    size={12}
                    className={`transition-transform duration-200 ${expandedIndex === i ? 'rotate-180' : ''}`}
                  />
                </button>
                <button
                  onClick={() => removeEntry(i)}
                  className="flex h-8 w-10 items-center justify-center text-[#ccc] hover:text-chirp-error transition-colors duration-150 opacity-0 group-hover:opacity-100"
                >
                  ✕
                </button>
              </div>

              {/* Expanded boost slider */}
              {expandedIndex === i && (
                <div className="px-[18px] pb-3 pt-1 flex items-center gap-3 animate-slide-up">
                  <span className="text-[11px] text-[#aaa] w-8">Low</span>
                  <input
                    type="range"
                    min="1.0"
                    max="5.0"
                    step="0.5"
                    value={entry.boost}
                    onChange={(e) => updateBoost(i, parseFloat(e.target.value))}
                    className="flex-1 accent-[#1a1a1a] h-1"
                  />
                  <span className="text-[11px] text-[#aaa] w-8 text-right">High</span>
                  <span className="text-[11px] text-[#666] font-medium w-8 text-right">{entry.boost.toFixed(1)}</span>
                </div>
              )}
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
