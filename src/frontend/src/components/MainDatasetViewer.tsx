import { useState, useEffect, useRef } from 'react';
import { Folder, Image as ImageIcon, Loader2, Database, CheckCircle2, XCircle, RefreshCw, Lock, Trash2, Upload } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

// ── Types ──────────────────────────────────────────────────────────────────
interface ImageDetail {
  url: string;
  filename: string;
  path: string;
  indexed: boolean;
}

interface DatasetClassDetail {
  name: string;
  images: string[];
  images_detail: ImageDetail[];
}

interface DatasetStats {
  support_classes: number;
  support_images: number;
  support_images_pending: number;
  test_images: number;
  qdrant_support: number;
  qdrant_support_active: number;
  qdrant_test: number;
  qdrant_total: number;
}

interface DatasetResponse {
  support: DatasetClassDetail[];
  test_detail: ImageDetail[];
  stats: DatasetStats;
}

const BASE = import.meta.env.VITE_API_BASE_URL || '';

// ── Main Component ──────────────────────────────────────────────────────────
export function MainDatasetViewer() {
  const [dataset, setDataset]           = useState<DatasetResponse | null>(null);
  const [isLoading, setIsLoading]       = useState(true);
  const [error, setError]               = useState<string | null>(null);
  const [activeTab, setActiveTab]       = useState<'main' | 'support'>('main');
  const [selectedClass, setSelectedClass] = useState<string | null>(null);
  const [isDeleting, setIsDeleting]     = useState<string | null>(null);
  const [isUploading, setIsUploading]   = useState(false);
  const [deleteError, setDeleteError]   = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDataset(true);
    const interval = setInterval(() => loadDataset(false), 10_000);
    return () => clearInterval(interval);
  }, []);

  const loadDataset = async (showSpinner = false) => {
    try {
      if (showSpinner) setIsLoading(true);
      setError(null);
      const res = await fetch(`${BASE}/api/dataset`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json: DatasetResponse = await res.json();
      setDataset(json);
      setSelectedClass(prev => prev ?? (json.support?.[0]?.name ?? null));
    } catch (err) {
      if (showSpinner) setError('Failed to load dataset. Is the backend running?');
      console.error('[MainDatasetViewer]', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Delete — only works for TEST images (backend enforces SUPPORT block too)
  const handleDelete = async (path: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteError(null);
    if (!confirm('Permanently delete this image from disk and index?')) return;
    setIsDeleting(path);
    try {
      const res = await fetch(`${BASE}/api/delete-image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      });
      const json = await res.json();
      if (json.status === 'error') {
        setDeleteError(json.message);
      } else {
        await loadDataset(false);
      }
    } catch (err) {
      setDeleteError('Network error while deleting.');
    } finally {
      setIsDeleting(null);
    }
  };

  // Upload — goes to test_dataset/, indexed as TEST
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setIsUploading(true);
    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) formData.append('images', files[i]);
      const res = await fetch(`${BASE}/api/upload-main`, { method: 'POST', body: formData });
      if (!res.ok) throw new Error('Upload failed');
      await loadDataset(false);
    } catch (err) {
      alert('Failed to upload: ' + err);
    } finally {
      setIsUploading(false);
      e.target.value = '';
    }
  };

  // ── Loading / Error ─────────────────────────────────────────────────────
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
        <p className="text-text-secondary">Loading dataset...</p>
      </div>
    );
  }
  if (error || !dataset) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <p className="text-red-400">{error || 'Dataset empty'}</p>
        <button onClick={() => loadDataset(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-surface border border-border hover:border-primary-400 text-text-secondary hover:text-white transition-all">
          <RefreshCw className="w-4 h-4" /> Retry
        </button>
      </div>
    );
  }

  const { stats } = dataset;
  const selectedClassData = dataset.support.find(c => c.name === selectedClass);
  // Combined count: support active + test images
  const mainTotal = stats.support_images + stats.test_images;

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-3">
            <Database className="w-6 h-6 text-primary-400" />
            <h2 className="text-2xl font-semibold">Main Dataset Viewer</h2>
          </div>
          <p className="text-xs text-text-tertiary">
            Main Dataset = Support (read-only) + Test (upload / delete)
          </p>
        </div>
        <button onClick={() => loadDataset(true)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface border border-border hover:border-primary-400 text-text-secondary hover:text-white transition-all text-sm">
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {/* Delete error banner */}
      {deleteError && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm flex items-center gap-2">
          <XCircle className="w-4 h-4 flex-shrink-0" />
          <span>{deleteError}</span>
          <button onClick={() => setDeleteError(null)} className="ml-auto text-red-400 hover:text-red-300">✕</button>
        </div>
      )}

      {/* Pending uploads banner */}
      {stats.support_images_pending > 0 && (
        <div className="p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-sm flex items-center gap-2">
          <span>⏳</span>
          <span>{stats.support_images_pending} support image{stats.support_images_pending !== 1 ? 's' : ''} pending training — go to Few-shot Management to activate them.</span>
        </div>
      )}

      {/* Stats bar */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        {[
          { label: 'Main Total',    value: mainTotal,                   color: 'text-blue-400' },
          { label: 'Support (Active)', value: stats.support_images,     color: 'text-primary-400' },
          { label: 'Support Pending',  value: stats.support_images_pending, color: stats.support_images_pending > 0 ? 'text-yellow-400' : 'text-text-tertiary' },
          { label: 'Test Images',   value: stats.test_images,           color: 'text-accent' },
          { label: 'Qdrant Total',  value: stats.qdrant_total,          color: 'text-green-400' },
        ].map(s => (
          <div key={s.label} className="glass-card p-3 rounded-xl flex flex-col gap-1">
            <span className="text-xs text-text-tertiary uppercase tracking-wider">{s.label}</span>
            <span className={`text-2xl font-bold ${s.color}`}>{s.value}</span>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="flex gap-4 border-b border-border pb-2">
        <button
          onClick={() => setActiveTab('main')}
          className={twMerge(clsx(
            'px-4 py-2 text-lg font-medium transition-all rounded-t-lg',
            activeTab === 'main'
              ? 'text-blue-400 border-b-2 border-blue-400 bg-blue-500/10'
              : 'text-text-tertiary hover:text-text-secondary hover:bg-surface'
          ))}
        >
          Main Dataset ({mainTotal})
        </button>
        <button
          onClick={() => setActiveTab('support')}
          className={twMerge(clsx(
            'px-4 py-2 text-lg font-medium transition-all rounded-t-lg',
            activeTab === 'support'
              ? 'text-primary-400 border-b-2 border-primary-400 bg-primary-500/10'
              : 'text-text-tertiary hover:text-text-secondary hover:bg-surface'
          ))}
        >
          Support Classes ({stats.support_classes})
        </button>
      </div>

      {/* ── MAIN DATASET TAB: Support (locked) + Test (deletable) ─────────── */}
      {activeTab === 'main' && (
        <div className="flex flex-col gap-6">

          {/* Section: Support images — READ-ONLY */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2 pb-2 border-b border-border">
              <Lock className="w-4 h-4 text-text-tertiary" />
              <h3 className="font-semibold text-text-secondary">Support Images <span className="text-xs text-text-tertiary">(read-only — managed via Few-Shot Management)</span></h3>
              <QdrantBadge detail={dataset.support.flatMap(c => c.images_detail)} />
            </div>
            {dataset.support.map(cls => (
              <div key={cls.name} className="flex flex-col gap-2">
                <div className="flex items-center gap-2">
                  <Folder className="w-4 h-4 text-primary-400" />
                  <span className="text-sm font-medium text-text-secondary">{cls.name}</span>
                  <span className="text-xs text-text-tertiary">({cls.images.length})</span>
                </div>
                <div className="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-8 xl:grid-cols-10 gap-2">
                  {(cls.images_detail ?? []).map((img, idx) => (
                    <div key={idx} className="relative aspect-square rounded-lg overflow-hidden border border-border bg-background group">
                      <img src={img.url} alt={img.filename} className="w-full h-full object-cover" loading="lazy" />
                      {/* Lock badge — no delete button */}
                      <div className="absolute top-1 right-1">
                        <Lock className="w-3 h-3 text-white/60 drop-shadow" />
                      </div>
                      <div className="absolute inset-x-0 bottom-0 bg-black/60 text-white text-[9px] px-1 py-0.5 truncate opacity-0 group-hover:opacity-100 transition-opacity">
                        {img.filename}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Section: Test images — UPLOADABLE + DELETABLE */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between pb-2 border-b border-border">
              <div className="flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-accent" />
                <h3 className="font-semibold text-text-secondary">Test Images <span className="text-xs text-text-tertiary">(you can upload &amp; delete these)</span></h3>
                <QdrantBadge detail={dataset.test_detail} />
              </div>
              {/* Upload button */}
              <div className="relative">
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept="image/png,image/jpeg,image/webp"
                  onChange={handleUpload}
                  disabled={isUploading}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                />
                <button
                  disabled={isUploading}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-accent hover:bg-accent/80 text-white font-medium transition-colors disabled:opacity-50 text-sm"
                >
                  {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <><Upload className="w-4 h-4" /> Upload Test Images</>}
                </button>
              </div>
            </div>

            {dataset.test_detail && dataset.test_detail.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6 gap-4">
                {dataset.test_detail.map((img, idx) => (
                  <div key={idx} className="group relative aspect-square rounded-xl overflow-hidden border border-border bg-background">
                    <img src={img.url} alt={img.filename} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" loading="lazy" />
                    {/* Index badge */}
                    <div className="absolute top-1.5 right-1.5" title={img.indexed ? 'Indexed' : 'Not indexed'}>
                      {img.indexed
                        ? <CheckCircle2 className="w-4 h-4 text-green-400 drop-shadow" />
                        : <XCircle className="w-4 h-4 text-red-400/70 drop-shadow" />}
                    </div>
                    {/* Delete button */}
                    <button
                      onClick={(e) => handleDelete(img.path, e)}
                      disabled={isDeleting === img.path}
                      className="absolute top-2 left-2 p-1.5 rounded-full bg-red-500/80 hover:bg-red-500 text-white opacity-0 group-hover:opacity-100 transition-opacity"
                      title="Delete test image"
                    >
                      {isDeleting === img.path
                        ? <Loader2 className="w-3 h-3 animate-spin" />
                        : <Trash2 className="w-3 h-3" />}
                    </button>
                    <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-2 pt-6 pointer-events-none">
                      <p className="text-white text-[10px] truncate drop-shadow-md">{img.filename}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-32 flex flex-col items-center justify-center gap-3 text-text-tertiary border border-dashed border-border rounded-xl">
                <ImageIcon className="w-8 h-8 opacity-40" />
                <p className="text-sm">No test images yet. Click "Upload Test Images" to add some.</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── SUPPORT TAB: Class browser, read-only ─────────────────────────── */}
      {activeTab === 'support' && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Class list */}
          <div className="md:col-span-1 flex flex-col gap-2">
            <div className="flex items-center gap-2 mb-2">
              <Lock className="w-3.5 h-3.5 text-text-tertiary" />
              <h3 className="text-sm font-medium text-text-tertiary uppercase tracking-wider">Classes</h3>
            </div>
            {dataset.support.map(cls => (
              <button
                key={cls.name}
                onClick={() => setSelectedClass(cls.name)}
                className={twMerge(clsx(
                  'flex items-center justify-between p-3 rounded-xl transition-all text-left',
                  selectedClass === cls.name
                    ? 'bg-primary-600 text-white shadow-lg shadow-primary-500/20'
                    : 'bg-surface border border-border hover:border-primary-400/50 hover:bg-surface-hover text-text-secondary'
                ))}
              >
                <div className="flex items-center gap-2 overflow-hidden">
                  <Folder className={clsx('w-4 h-4 flex-shrink-0', selectedClass === cls.name ? 'text-white' : 'text-primary-400')} />
                  <span className="truncate font-medium">{cls.name}</span>
                </div>
                <span className={clsx('text-xs px-2 py-1 rounded-full', selectedClass === cls.name ? 'bg-white/20' : 'bg-background')}>
                  {cls.images.length}
                </span>
              </button>
            ))}
          </div>

          {/* Image grid — no delete */}
          <div className="md:col-span-3 glass-card p-6 rounded-2xl">
            {selectedClassData ? (
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between pb-4 border-b border-border">
                  <div className="flex items-center gap-2">
                    <ImageIcon className="w-5 h-5 text-text-secondary" />
                    <h3 className="text-lg font-medium">{selectedClassData.name}</h3>
                    <span className="text-sm text-text-tertiary">({selectedClassData.images.length} images)</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <QdrantBadge detail={selectedClassData.images_detail} />
                    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs text-text-tertiary border border-border">
                      <Lock className="w-3 h-3" /> Read-only
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                  {(selectedClassData.images_detail ?? []).map((img, idx) => (
                    <div key={idx} className="group relative aspect-square rounded-xl overflow-hidden border border-border bg-background">
                      <img src={img.url} alt={img.filename} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" loading="lazy" />
                      <div className="absolute top-1.5 right-1.5" title={img.indexed ? 'Indexed' : 'Not indexed'}>
                        {img.indexed
                          ? <CheckCircle2 className="w-4 h-4 text-green-400 drop-shadow" />
                          : <XCircle className="w-4 h-4 text-red-400/70 drop-shadow" />}
                      </div>
                      <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white text-[10px] px-1.5 py-1 truncate opacity-0 group-hover:opacity-100 transition-opacity">
                        {img.filename}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-text-tertiary">
                Select a class to view images
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── QdrantBadge ─────────────────────────────────────────────────────────────
function QdrantBadge({ detail }: { detail?: ImageDetail[] }) {
  if (!detail || detail.length === 0) return null;
  const indexed = detail.filter(d => d.indexed).length;
  const allIndexed = indexed === detail.length;
  return (
    <div className={clsx(
      'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border',
      allIndexed
        ? 'bg-green-500/10 text-green-400 border-green-500/30'
        : 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30'
    )}>
      {allIndexed ? <CheckCircle2 className="w-3.5 h-3.5" /> : <Database className="w-3.5 h-3.5" />}
      {indexed}/{detail.length} in Qdrant
    </div>
  );
}