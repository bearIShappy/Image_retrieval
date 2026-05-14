import { useState, useEffect, useCallback } from 'react';
import { Database, BarChart2, Calendar, RefreshCw, Wand2, Download, AlertCircle, HardDrive } from 'lucide-react';
import { api } from '../api';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const BASE = import.meta.env.VITE_API_BASE_URL || '';

export function DatasetOverview() {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const loadStats = useCallback(async () => {
    setLoading(true);
    try {
      // Instead of complex DB routes, we'll hit the /api/dataset or /api/stats endpoint
      const res = await fetch(`${BASE}/api/stats`);
      if (!res.ok) throw new Error('Failed to load stats');
      const data = await res.json();
      setStats(data);
      setError(null);
    } catch (err) {
      setError("Failed to load dataset overview");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadStats(); }, [loadStats]);

  const handleRebuild = async () => {
    if (!window.confirm("Are you sure you want to rebuild the entire search index? This may take a while.")) return;
    setActionLoading('rebuild');
    try {
      await api.rebuildIndex();
      await loadStats();
    } catch (err) {
      alert("Failed to rebuild index");
    } finally {
      setActionLoading(null);
    }
  };

  const handleTrain = async () => {
    setActionLoading('train');
    try {
      await api.finetuneModel();
      await loadStats();
    } catch (err) {
      alert("Failed to train model");
    } finally {
      setActionLoading(null);
    }
  };

  const handleExport = () => {
    // In a real app, this would trigger a download of the metadata JSON
    alert("Exporting metadata...");
  };

  if (loading && !stats) return (
    <div className="flex items-center justify-center py-12">
      <RefreshCw className="w-8 h-8 text-primary-400 animate-spin" />
    </div>
  );

  if (error) return (
    <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/30 text-red-400 rounded-xl">
      <AlertCircle className="w-5 h-5" />
      <span>{error}</span>
    </div>
  );

  const sqlite = stats?.sqlite || {};
  
  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-500">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <HardDrive className="w-8 h-8 text-primary-400" />
          <div>
            <h2 className="text-3xl font-bold text-white tracking-tight">Dataset Overview</h2>
            <p className="text-text-tertiary">High-level insights and management controls</p>
          </div>
        </div>
        <button
          onClick={loadStats}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-surface border border-border hover:bg-surface-hover transition-colors text-sm"
        >
          <RefreshCw className={clsx("w-4 h-4", loading && "animate-spin")} />
          Refresh Stats
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card p-6 rounded-2xl flex flex-col gap-2 border border-border/50">
          <span className="text-text-tertiary text-sm font-medium uppercase tracking-wider">Total Images</span>
          <span className="text-4xl font-bold text-white">{sqlite.total_images || 0}</span>
        </div>
        <div className="glass-card p-6 rounded-2xl flex flex-col gap-2 border border-border/50">
          <span className="text-text-tertiary text-sm font-medium uppercase tracking-wider">Classes</span>
          <span className="text-4xl font-bold text-primary-400">{sqlite.prototypes || 0}</span>
        </div>
        <div className="glass-card p-6 rounded-2xl flex flex-col gap-2 border border-border/50">
          <span className="text-text-tertiary text-sm font-medium uppercase tracking-wider">Support Examples</span>
          <span className="text-4xl font-bold text-accent">{sqlite.support_images || 0}</span>
        </div>
        <div className="glass-card p-6 rounded-2xl flex flex-col gap-2 border border-border/50">
          <span className="text-text-tertiary text-sm font-medium uppercase tracking-wider">Recently Added</span>
          <span className="text-4xl font-bold text-green-400">{sqlite.pending_uploads || 0}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-card p-6 rounded-2xl border border-border/50 flex flex-col gap-4">
          <div className="flex items-center gap-2 text-text-secondary pb-3 border-b border-border">
            <BarChart2 className="w-5 h-5" />
            <h3 className="font-semibold text-lg">Class Distribution</h3>
          </div>
          <div className="flex items-center justify-center h-48 text-text-tertiary italic">
            Chart data unavailable
          </div>
        </div>
        <div className="glass-card p-6 rounded-2xl border border-border/50 flex flex-col gap-4">
          <div className="flex items-center gap-2 text-text-secondary pb-3 border-b border-border">
            <Calendar className="w-5 h-5" />
            <h3 className="font-semibold text-lg">Upload Activity Timeline</h3>
          </div>
          <div className="flex items-center justify-center h-48 text-text-tertiary italic">
            Timeline data unavailable
          </div>
        </div>
      </div>

      <div className="glass-card p-6 rounded-2xl border border-border/50 flex flex-col gap-6">
        <h3 className="font-semibold text-lg border-b border-border pb-3">Platform Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={handleRebuild}
            disabled={actionLoading !== null}
            className="flex flex-col items-center justify-center gap-3 p-6 rounded-xl bg-surface border border-border hover:border-primary-500 hover:bg-primary-500/10 transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Database className="w-8 h-8 text-text-secondary group-hover:text-primary-400 transition-colors" />
            <span className="font-medium text-text-primary">Rebuild Search Index</span>
            <span className="text-xs text-text-tertiary text-center">Fully reconstruct the vector database from source images</span>
          </button>
          
          <button
            onClick={handleTrain}
            disabled={actionLoading !== null}
            className="flex flex-col items-center justify-center gap-3 p-6 rounded-xl bg-surface border border-border hover:border-accent hover:bg-accent/10 transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Wand2 className="w-8 h-8 text-text-secondary group-hover:text-accent transition-colors" />
            <span className="font-medium text-text-primary">Train Support Model</span>
            <span className="text-xs text-text-tertiary text-center">Process pending uploads and update class prototypes</span>
          </button>

          <button
            onClick={handleExport}
            disabled={actionLoading !== null}
            className="flex flex-col items-center justify-center gap-3 p-6 rounded-xl bg-surface border border-border hover:border-green-500 hover:bg-green-500/10 transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download className="w-8 h-8 text-text-secondary group-hover:text-green-400 transition-colors" />
            <span className="font-medium text-text-primary">Export Metadata</span>
            <span className="text-xs text-text-tertiary text-center">Download current dataset mappings as JSON</span>
          </button>
        </div>
      </div>
    </div>
  );
}
