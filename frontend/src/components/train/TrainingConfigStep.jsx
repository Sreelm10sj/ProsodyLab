import React, { useState } from 'react'
import { ArrowLeft, ArrowRight, ChevronDown } from 'lucide-react'
import { formatSeparationScore } from './formatters'

export default function TrainingConfigStep({
  datasetValidation,
  selectedAnchor,
  config,
  onChangeConfig,
  onStartTraining,
  onBack,
  isLaunching,
}) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const language = config.language || 'GER'
  const device = config.device || 'auto'
  const seed = config.seed ?? 42

  function updateField(field, value) {
    onChangeConfig({
      ...config,
      [field]: value,
    })
  }

  function handleLanguageChange(lang) {
    const next = {
      ...config,
      language: lang,
    }
    if (lang === 'GER') {
      next.top_k_rep = 1
      next.batch_words = 32
    } else if (lang === 'ITA') {
      next.top_k_rep = 10
      next.batch_words = 48
    }
    onChangeConfig(next)
  }

  // Derived protocol hints
  let protocolInfo = {
    topK: 1,
    batchWords: 32,
    note: 'German benchmark protocol (direct anchor prototype representation)',
  }
  if (language === 'ITA') {
    protocolInfo = {
      topK: 10,
      batchWords: 48,
      note: 'Italian benchmark protocol (top-10 soft clustering cluster averages)',
    }
  } else if (language === 'CUSTOM') {
    protocolInfo = {
      topK: config.top_k_rep ?? 1,
      batchWords: config.batch_words ?? 32,
      note: 'Custom dataset protocol with generic defaults',
    }
  }

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 md:p-8 space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900 tracking-tight">
          Stage 03 — Training Configuration
        </h2>
        <p className="text-xs text-slate-500 mt-1 leading-relaxed">
          Configure hyperparameters and execution hardware. Collaborator protocols are automatically resolved by language.
        </p>
      </div>

      {/* Primary Configuration Controls */}
      <div className="rounded-xl border border-slate-200 bg-slate-50/30 p-6 space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          {/* Language Selection */}
          <div>
            <label className="block text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
              Language / Protocol
            </label>
            <div className="inline-flex rounded-xl border border-slate-200 p-1 bg-slate-50/70 shadow-2xs w-full gap-1">
              {['GER', 'ITA', 'CUSTOM'].map((lang) => (
                <button
                  key={lang}
                  type="button"
                  onClick={() => handleLanguageChange(lang)}
                  className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
                    language === lang
                      ? 'bg-slate-900 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/60'
                  }`}
                >
                  {lang}
                </button>
              ))}
            </div>
          </div>

          {/* Execution Device */}
          <div>
            <label className="block text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
              Compute Device
            </label>
            <div className="inline-flex rounded-xl border border-slate-200 p-1 bg-slate-50/70 shadow-2xs w-full gap-1">
              {[
                { id: 'auto', label: 'Auto' },
                { id: 'cpu', label: 'CPU' },
                { id: 'cuda', label: 'CUDA' },
              ].map((dev) => (
                <button
                  key={dev.id}
                  type="button"
                  onClick={() => updateField('device', dev.id)}
                  className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all ${
                    device === dev.id
                      ? 'bg-slate-900 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/60'
                  }`}
                >
                  {dev.label}
                </button>
              ))}
            </div>
          </div>

          {/* Random Seed */}
          <div>
            <label className="block text-xs font-bold text-slate-900 uppercase tracking-wider mb-2">
              Random Seed
            </label>
            <input
              type="number"
              value={seed}
              onChange={(e) => updateField('seed', parseInt(e.target.value, 10) || 42)}
              className="w-full py-2 px-3.5 border border-slate-200 rounded-xl font-mono text-sm text-slate-900 bg-white focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500 shadow-2xs"
            />
          </div>
        </div>

        {/* Protocol Transparency Banner */}
        <div className="p-4 bg-white rounded-xl border border-slate-200 font-mono text-xs space-y-1.5 shadow-2xs">
          <div className="font-bold text-slate-900 flex items-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full bg-[#FDB813]" />
            <span>Active Protocol: {language}</span>
          </div>
          <div className="text-slate-600 text-[11px] leading-relaxed">
            • Representative Selection: <strong className="text-slate-900 font-bold">Top {protocolInfo.topK}</strong> ({language === 'GER' ? '1-sample prototype' : '10-sample cluster mean'})<br />
            • Batching Strategy: <strong className="text-slate-900 font-bold">{protocolInfo.batchWords} words</strong> per contrastive batch<br />
            • Context: {protocolInfo.note}
          </div>
        </div>

        {/* Advanced Hyperparameters Collapsible */}
        <div className="pt-2 border-t border-slate-200/80">
          <button
            type="button"
            onClick={() => setShowAdvanced((p) => !p)}
            className="text-xs font-semibold text-slate-600 hover:text-slate-900 flex items-center gap-1.5 cursor-pointer"
          >
            <ChevronDown
              size={14}
              className={`transition-transform duration-150 ${showAdvanced ? 'rotate-180' : ''}`}
            />
            <span>{showAdvanced ? 'Hide Advanced Hyperparameters' : 'Show Advanced Hyperparameters'}</span>
          </button>

          {showAdvanced && (
            <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 bg-white rounded-xl border border-slate-200 font-mono text-xs shadow-2xs">
              <div>
                <label className="block text-[10px] text-slate-500 uppercase font-semibold mb-1">Warm-up Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={config.warmup_epochs ?? 5}
                  onChange={(e) => updateField('warmup_epochs', parseInt(e.target.value, 10) || 5)}
                  className="w-full p-2 border border-slate-200 rounded-lg bg-white text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500"
                />
              </div>

              <div>
                <label className="block text-[10px] text-slate-500 uppercase font-semibold mb-1">IDEC Pretrain</label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={config.idec_pretrain_epochs ?? 10}
                  onChange={(e) => updateField('idec_pretrain_epochs', parseInt(e.target.value, 10) || 10)}
                  className="w-full p-2 border border-slate-200 rounded-lg bg-white text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500"
                />
              </div>

              <div>
                <label className="block text-[10px] text-slate-500 uppercase font-semibold mb-1">Outer Loops</label>
                <input
                  type="number"
                  min="1"
                  max="30"
                  value={config.outer_loops ?? 13}
                  onChange={(e) => updateField('outer_loops', parseInt(e.target.value, 10) || 13)}
                  className="w-full p-2 border border-slate-200 rounded-lg bg-white text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500"
                />
              </div>

              <div>
                <label className="block text-[10px] text-slate-500 uppercase font-semibold mb-1">IDEC Steps / Loop</label>
                <input
                  type="number"
                  min="10"
                  max="500"
                  value={config.idec_steps_per_loop ?? 100}
                  onChange={(e) => updateField('idec_steps_per_loop', parseInt(e.target.value, 10) || 100)}
                  className="w-full p-2 border border-slate-200 rounded-lg bg-white text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500"
                />
              </div>

              <div>
                <label className="block text-[10px] text-slate-500 uppercase font-semibold mb-1">Refinement Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={config.refinement_epochs ?? 3}
                  onChange={(e) => updateField('refinement_epochs', parseInt(e.target.value, 10) || 3)}
                  className="w-full p-2 border border-slate-200 rounded-lg bg-white text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500"
                />
              </div>

              <div>
                <label className="block text-[10px] text-slate-500 uppercase font-semibold mb-1">SupCon Temp (τ)</label>
                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  max="1.0"
                  value={config.supcon_temperature ?? 0.08}
                  onChange={(e) => updateField('supcon_temperature', parseFloat(e.target.value) || 0.08)}
                  className="w-full p-2 border border-slate-200 rounded-lg bg-white text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500"
                />
              </div>

              <div>
                <label className="block text-[10px] text-slate-500 uppercase font-semibold mb-1">Learning Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  min="0.00001"
                  max="0.1"
                  value={config.learning_rate ?? 0.001}
                  onChange={(e) => updateField('learning_rate', parseFloat(e.target.value) || 0.001)}
                  className="w-full p-2 border border-slate-200 rounded-lg bg-white text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500"
                />
              </div>

              <div>
                <label className="block text-[10px] text-slate-500 uppercase font-semibold mb-1">Top-K Override</label>
                <input
                  type="number"
                  min="1"
                  placeholder={`Default (${protocolInfo.topK})`}
                  value={config.top_k_rep ?? ''}
                  onChange={(e) => updateField('top_k_rep', e.target.value ? parseInt(e.target.value, 10) : undefined)}
                  className="w-full p-2 border border-slate-200 rounded-lg bg-white text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500"
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Pre-launch Summary */}
      <div className="rounded-xl border border-slate-200 bg-slate-50/50 p-4 font-mono text-xs flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-0.5">
          <span className="text-slate-500 text-[10px] uppercase block font-medium">Launch Summary</span>
          <span className="text-slate-700">
            Dataset: <strong className="text-slate-900 font-bold">{datasetValidation?.syllable_count?.toLocaleString()} syllables</strong> · Anchor: <strong className="text-slate-900 font-bold">Word {selectedAnchor?.word_id}</strong> (S_train: {formatSeparationScore(selectedAnchor?.separation_score)}) · Device: <strong className="text-slate-900 font-bold">{device.toUpperCase()}</strong>
          </span>
        </div>
        <div className="font-bold text-slate-900 text-sm">
          {config.outer_loops ?? 13} Loops
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between pt-2">
        <button
          type="button"
          onClick={onBack}
          disabled={isLaunching}
          className="btn-white-action px-4 py-2 text-xs font-semibold disabled:opacity-50"
        >
          <ArrowLeft size={14} />
          <span>Back to Anchors</span>
        </button>

        <button
          type="button"
          onClick={onStartTraining}
          disabled={isLaunching}
          className="btn-gold-action px-8 py-3.5 text-base font-bold shadow-gold disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {isLaunching ? (
            <>
              <span className="w-4 h-4 border-2 border-[#0F172A] border-t-transparent rounded-full animate-spin" />
              <span>Initializing Job...</span>
            </>
          ) : (
            <>
              <span>Start Training Job</span>
              <ArrowRight size={18} className="arrow-shift" />
            </>
          )}
        </button>
      </div>
    </div>
  )
}
