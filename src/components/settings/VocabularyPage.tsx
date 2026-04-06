import { useState } from 'react'
import { trackEvent } from '@aptabase/tauri'
import { useAppStore } from '../../stores/appStore'
import { Button } from '../shared/Button'

export function VocabularyPage() {
  const vocabulary = useAppStore((s) => s.vocabulary)
  const addWord = useAppStore((s) => s.addVocabularyWord)
  const removeWord = useAppStore((s) => s.removeVocabularyWord)

  const [newWord, setNewWord] = useState('')

  const handleAdd = () => {
    const word = newWord.trim()
    if (!word) return
    addWord(word)
    setNewWord('')
    trackEvent('feature_used', { feature: 'vocabulary_add' })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleAdd()
  }

  return (
    <div className="flex flex-col gap-5 animate-slide-up">
      <div className="mb-1">
        <h1 className="font-display font-extrabold text-2xl text-dm-primary tracking-[-0.5px]">
          Vocabulary
        </h1>
        <p className="text-[13px] text-dm-secondary mt-1">
          Words and names Chirp should know. These are sent to the AI cleanup model so it can correct misheard words.
        </p>
      </div>

      {vocabulary.length > 0 ? (
        <div className="overflow-hidden rounded-card border border-card-border">
          {/* Header */}
          <div className="flex bg-card-hover px-[18px] py-2.5">
            <span className="flex-1 text-[11px] font-semibold uppercase tracking-[0.5px] text-dm-secondary">
              Word / Phrase
            </span>
            <span className="w-10" />
          </div>

          {/* Rows */}
          {vocabulary.map((word, i) => (
            <div
              key={i}
              className={`flex items-center px-[18px] h-11 border-b border-dm-row-sep last:border-b-0 transition-colors hover:bg-card-hover group animate-slide-up ${
                i % 2 === 0 ? 'bg-card' : 'bg-card-hover/50'
              }`}
              style={{ animationDelay: `${i * 30}ms` }}
            >
              <span className="flex-1 text-[13px] text-dm-primary">
                {word}
              </span>
              <button
                onClick={() => removeWord(i)}
                className="flex h-8 w-10 items-center justify-center text-dm-tertiary hover:text-chirp-error transition-colors duration-150 opacity-0 group-hover:opacity-100"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center justify-center rounded-card border border-dashed border-card-border bg-card-hover px-6 py-10">
          <p className="text-[13px] text-dm-secondary text-center">
            No words yet. Add names, jargon, or technical terms that Chirp should recognize.
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
          className="flex-1 h-10 rounded-lg border border-card-border bg-card px-3 text-[13px] text-dm-primary placeholder:text-dm-tertiary focus:border-chirp-yellow focus:shadow-[0_0_0_3px_rgba(240,183,35,0.1)] focus:outline-none transition-all duration-150"
        />
        <Button onClick={handleAdd} disabled={!newWord.trim() || vocabulary.length >= 500}>
          Add
        </Button>
      </div>

      {vocabulary.length >= 450 && (
        <p className="text-xs text-chirp-error">
          You're approaching the maximum of 500 words ({vocabulary.length}/500).
        </p>
      )}
    </div>
  )
}
