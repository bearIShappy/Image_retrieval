import { Settings } from 'lucide-react';
import { type RetrievalParams } from '../api';

interface RetrievalControlsProps {
  isTextOnly?: boolean;
  params: RetrievalParams;
  onChange: (params: Partial<RetrievalParams>) => void;
}

export function RetrievalControls({ params, onChange, isTextOnly }: RetrievalControlsProps) {
  return (
    <div className="flex flex-col gap-6 w-full glass-card p-6 rounded-2xl">
      <div className="flex items-center gap-2">
        <Settings className="w-5 h-5 text-accent" />
        <h2 className="text-xl font-semibold">Retrieval Controls</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {/* Top K */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-text-secondary">Top-K Results</label>
          <input
            type="number"
            min={1}
            max={100}
            value={params.topK}
            onChange={(e) => onChange({ topK: parseInt(e.target.value) || 10 })}
            className="bg-surface border border-border rounded-xl px-4 py-2.5 text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500/50"
          />
        </div>

        {/* Retrieval Mode */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-text-secondary">Retrieval Mode</label>
          <div className="relative">
            <select
              value={params.mode}
              onChange={(e) => onChange({ mode: e.target.value as RetrievalParams['mode'] })}
              className="w-full appearance-none bg-surface border border-border rounded-xl px-4 py-2.5 text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500/50"
            >
              <option value="global">Global</option>
              <option value="multi-query">Multi-query</option>
              <option value="region-aware">Region-aware</option>
              <option value="prototype">Prototype</option>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-text-tertiary">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
              </svg>
            </div>
          </div>
        </div>

        {/* Aggregation */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-text-secondary">Aggregation</label>
          <div className="relative">
            <select
              value={params.aggregation}
              onChange={(e) => onChange({ aggregation: e.target.value as RetrievalParams['aggregation'] })}
              className="w-full appearance-none bg-surface border border-border rounded-xl px-4 py-2.5 text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500/50"
            >
              <option value="max">Max</option>
              <option value="mean">Mean</option>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-text-tertiary">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
              </svg>
            </div>
          </div>
        </div>

        {/* Class Filter
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-text-secondary">
            Class Filter
          </label>
          <div className="relative">
            <select
              value={params.forcedClass || ''}
              onChange={(e) =>
                onChange({ forcedClass: e.target.value || null })
              }
              className="w-full appearance-none bg-surface border border-border rounded-xl px-4 py-2.5 text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500/50"
            >
              <option value="">— All Classes —</option>
              <option value="heavy_drop">Heavy Drop</option>
              <option value="cff">CFF</option>
              <option value="static_line_jump">Static Line Jump</option>
              <option value="paramotor">Paramotor</option>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-4 text-text-tertiary">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
              </svg>
            </div>
          </div>
        </div> */}

        {/* Threshold */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-center text-sm font-medium text-text-secondary">
            <label>Similarity Threshold</label>
            <span className="text-primary-400 font-mono">{params.threshold.toFixed(2)}</span>
          </div>
          <div className="flex items-center h-[42px]">
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={params.threshold}
              onChange={(e) => onChange({ threshold: parseFloat(e.target.value) })}
              className="w-full h-2 bg-border rounded-lg appearance-none cursor-pointer accent-primary-500"
            />
          </div>
        </div>

        {/* Date Filters */}
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-text-secondary">From Date</label>
          <input
            type="date"
            value={params.fromDate || ''}
            onChange={(e) => onChange({ fromDate: e.target.value })}
            className="bg-surface border border-border rounded-xl px-4 py-2.5 text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500/50"
          />
        </div>
        <div className="flex flex-col gap-2">
          <label className="text-sm font-medium text-text-secondary">To Date</label>
          <input
            type="date"
            value={params.toDate || ''}
            onChange={(e) => onChange({ toDate: e.target.value })}
            className="bg-surface border border-border rounded-xl px-4 py-2.5 text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500/50"
          />
        </div>

        {/* Checkboxes */}
        <div className="flex flex-col justify-center gap-3 xl:col-span-4 lg:col-span-3 md:col-span-2">
          <div className="flex flex-wrap gap-6">
            <label className={`flex items-center gap-3 cursor-pointer group relative ${isTextOnly ? 'opacity-50 cursor-not-allowed' : ''}`} title="Focuses on important image regions instead of the full image.">
              <div className="relative flex items-center">
                <input
                  type="checkbox"
                  checked={isTextOnly ? false : params.useRegions}
                  disabled={isTextOnly}
                  onChange={(e) => onChange({ useRegions: e.target.checked })}
                  className="peer sr-only"
                />
                <div className="w-5 h-5 rounded border-2 border-border group-hover:border-primary-400 peer-checked:bg-primary-500 peer-checked:border-primary-500 transition-all flex items-center justify-center">
                  <svg className="w-3 h-3 text-white opacity-0 peer-checked:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              </div>
              <span className="text-sm font-medium text-text-secondary group-hover:text-text-primary transition-colors">Use Region-aware Features</span>
            </label>

            <label className="flex items-center gap-3 cursor-pointer group relative" title="Uses support examples for improved class-specific retrieval.">
              <div className="relative flex items-center">
                <input
                  type="checkbox"
                  checked={params.useFinetuned}
                  onChange={(e) => onChange({ useFinetuned: e.target.checked })}
                  className="peer sr-only"
                />
                <div className="w-5 h-5 rounded border-2 border-border group-hover:border-primary-400 peer-checked:bg-primary-500 peer-checked:border-primary-500 transition-all flex items-center justify-center">
                  <svg className="w-3 h-3 text-white opacity-0 peer-checked:opacity-100 transition-opacity" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              </div>
              <span className="text-sm font-medium text-text-secondary group-hover:text-text-primary transition-colors">Use Fine-tuned Model</span>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}
