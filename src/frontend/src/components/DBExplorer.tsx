import { useState, useEffect, useCallback } from 'react';
import {
  Database, Layers, Image as ImageIcon, RefreshCw, Search,
  ChevronLeft, ChevronRight, Tag, Box, Clock, Cpu, Filter,
  CheckCircle2, XCircle, FolderOpen, Braces, Activity, Trash2, AlertTriangle
} from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const BASE = import.meta.env.VITE_API_BASE_URL || '';

// ── Types ─────────────────────────────────────────────────────────────────

interface Collection {
  name: string;
  points: number;
}

interface CollectionsData {
  collections: Collection[];
  total: number;
  collection_name: string;
  vector_size: number;
}

interface DBImage {
  path: string;
  class: string;
  source: string;
  width: number | null;
  height: number | null;
  n_regions: number;
  indexed_at: number;
  url: string | null;
  filename: string;
}

interface ClassRow {
  class: string;
  source: string;
  count: number;
}

interface Prototype {
  class_name: string;
  n_images: number;
  updated_at: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────

const SOURCE_COLORS: Record<string, string> = {
  support: 'bg-primary-500/20 text-primary-300 border-primary-500/30',
  dataset: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  test:    'bg-accent/20 text-accent border-accent/30',
};

function Badge({ label, color }: { label: string; color?: string }) {
  return (
    <span className={twMerge(
      'px-2 py-0.5 rounded-full text-xs font-medium border',
      color ?? 'bg-surface text-text-tertiary border-border'
    )}>
      {label}
    </span>
  );
}

function StatCard({ icon, label, value, sub, accent }: {
  icon: React.ReactNode; label: string; value: string | number;
  sub?: string; accent?: string;
}) {
  return (
    <div className="glass-card p-4 rounded-xl flex flex-col gap-2 border border-border/50">
      <div className="flex items-center gap-2 text-text-tertiary text-xs uppercase tracking-wider">
        {icon}
        {label}
      </div>
      <div className={twMerge('text-3xl font-bold tabular-nums', accent ?? 'text-white')}>
        {value.toLocaleString()}
      </div>
      {sub && <div className="text-xs text-text-tertiary">{sub}</div>}
    </div>
  );
}

function SectionHeader({ title, icon }: { title: string; icon: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 pb-3 border-b border-border mb-4">
      <span className="text-primary-400">{icon}</span>
      <h3 className="text-sm font-semibold uppercase tracking-widest text-text-secondary">{title}</h3>
    </div>
  );
}

// ── Sub-panels ────────────────────────────────────────────────────────────

function QdrantPanel({ data, loading }: { data: CollectionsData | null; loading: boolean }) {
  if (loading) return <Spinner />;
  if (!data) return <Empty text="No Qdrant data" />;
  return (
    <div className="flex flex-col gap-6">
      <SectionHeader title="Qdrant Vector Store" icon={<Box className="w-4 h-4" />} />

      {/* Collection meta */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard icon={<Database className="w-3.5 h-3.5" />} label="Collection" value={data.collection_name} accent="text-primary-300" />
        <StatCard icon={<Layers className="w-3.5 h-3.5" />} label="Total Points" value={data.total} accent="text-green-400" />
        <StatCard icon={<Cpu className="w-3.5 h-3.5" />} label="Vector Size" value={data.vector_size} accent="text-accent" />
        <StatCard icon={<Activity className="w-3.5 h-3.5" />} label="Sources" value={data.collections.length} />
      </div>

      {/* Per-source breakdown */}
      <div className="glass-card rounded-xl p-5 border border-border/50">
        <SectionHeader title="Points by Source" icon={<Filter className="w-4 h-4" />} />
        <div className="flex flex-col gap-3">
          {data.collections.map(col => {
            const pct = data.total > 0 ? (col.points / data.total) * 100 : 0;
            return (
              <div key={col.name} className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge label={col.name} color={SOURCE_COLORS[col.name]} />
                  </div>
                  <span className="text-sm font-mono text-white font-semibold">
                    {col.points.toLocaleString()}
                    <span className="text-text-tertiary font-normal text-xs ml-1.5">
                      ({pct.toFixed(1)}%)
                    </span>
                  </span>
                </div>
                <div className="h-2 bg-surface rounded-full overflow-hidden">
                  <div
                    className={twMerge(
                      'h-full rounded-full transition-all duration-700',
                      col.name === 'support' ? 'bg-primary-500' :
                      col.name === 'dataset' ? 'bg-blue-500' : 'bg-accent'
                    )}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function ImageBrowser({
  classes,
}: {
  classes: ClassRow[];
}) {
  const [source, setSource] = useState<string>('');
  const [cls, setCls]       = useState<string>('');
  const [search, setSearch] = useState('');
  const [page, setPage]     = useState(0);
  const [images, setImages] = useState<DBImage[]>([]);
  const [total, setTotal]   = useState(0);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<DBImage | null>(null);
  const LIMIT = 30;

  const uniqueSources = [...new Set(classes.map(c => c.source))];
  const uniqueClasses = [...new Set(
    classes.filter(c => !source || c.source === source).map(c => c.class)
  )].sort();

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (source) params.set('source', source);
      if (cls) params.set('cls', cls);
      params.set('limit', String(LIMIT));
      params.set('offset', String(page * LIMIT));
      const res = await fetch(`${BASE}/api/db/images?${params}`);
      const json = await res.json();
      setImages(json.images ?? []);
      setTotal(json.total ?? 0);
    } finally {
      setLoading(false);
    }
  }, [source, cls, page]);

  useEffect(() => { setPage(0); }, [source, cls]);
  useEffect(() => { load(); }, [load]);

  const filtered = search
    ? images.filter(i => i.filename.toLowerCase().includes(search.toLowerCase()) ||
                         i.class.toLowerCase().includes(search.toLowerCase()))
    : images;

  const totalPages = Math.ceil(total / LIMIT);

  return (
    <div className="flex flex-col gap-4">
      <SectionHeader title="SQLite Image Records" icon={<ImageIcon className="w-4 h-4" />} />

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <div className="flex items-center gap-2 bg-surface border border-border rounded-xl px-3 py-2">
          <Search className="w-4 h-4 text-text-tertiary" />
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search filename or class..."
            className="bg-transparent text-sm text-white placeholder:text-text-tertiary outline-none w-44"
          />
        </div>

        <select
          value={source}
          onChange={e => { setSource(e.target.value); setCls(''); }}
          className="bg-surface border border-border rounded-xl px-3 py-2 text-sm text-white outline-none"
        >
          <option value="">All sources</option>
          {uniqueSources.map(s => <option key={s} value={s}>{s}</option>)}
        </select>

        <select
          value={cls}
          onChange={e => setCls(e.target.value)}
          className="bg-surface border border-border rounded-xl px-3 py-2 text-sm text-white outline-none"
        >
          <option value="">All classes</option>
          {uniqueClasses.map(c => <option key={c} value={c}>{c}</option>)}
        </select>

        <span className="text-text-tertiary text-sm ml-auto">
          {total.toLocaleString()} records
        </span>
      </div>

      {/* Table */}
      <div className="glass-card rounded-xl border border-border/50 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface/80">
                <th className="text-left px-4 py-3 text-text-tertiary font-medium uppercase text-xs tracking-wider w-16">Preview</th>
                <th className="text-left px-4 py-3 text-text-tertiary font-medium uppercase text-xs tracking-wider">Filename</th>
                <th className="text-left px-4 py-3 text-text-tertiary font-medium uppercase text-xs tracking-wider">Class</th>
                <th className="text-left px-4 py-3 text-text-tertiary font-medium uppercase text-xs tracking-wider">Source</th>
                <th className="text-left px-4 py-3 text-text-tertiary font-medium uppercase text-xs tracking-wider">Size</th>
                <th className="text-left px-4 py-3 text-text-tertiary font-medium uppercase text-xs tracking-wider">Regions</th>
                <th className="text-left px-4 py-3 text-text-tertiary font-medium uppercase text-xs tracking-wider">Indexed</th>
                <th className="text-left px-4 py-3 text-text-tertiary font-medium uppercase text-xs tracking-wider">Path</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={8} className="text-center py-12"><Spinner /></td></tr>
              ) : filtered.length === 0 ? (
                <tr><td colSpan={8} className="text-center py-12 text-text-tertiary">No records found</td></tr>
              ) : filtered.map((img, idx) => (
                <tr
                  key={idx}
                  onClick={() => img.url && setPreview(img)}
                  className={twMerge(
                    'border-b border-border/40 transition-colors',
                    img.url ? 'cursor-pointer hover:bg-surface-hover' : 'opacity-60'
                  )}
                >
                  {/* Thumbnail */}
                  <td className="px-4 py-2">
                    {img.url ? (
                      <div className="w-10 h-10 rounded-lg overflow-hidden bg-background border border-border flex-shrink-0">
                        <img src={img.url} alt={img.filename} className="w-full h-full object-cover" loading="lazy" />
                      </div>
                    ) : (
                      <div className="w-10 h-10 rounded-lg bg-surface border border-dashed border-border flex items-center justify-center">
                        <XCircle className="w-4 h-4 text-text-tertiary" />
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-2 font-mono text-xs text-white max-w-[180px] truncate" title={img.filename}>
                    {img.filename}
                  </td>
                  <td className="px-4 py-2">
                    <Badge label={img.class} color="bg-surface text-text-secondary border-border" />
                  </td>
                  <td className="px-4 py-2">
                    <Badge label={img.source} color={SOURCE_COLORS[img.source]} />
                  </td>
                  <td className="px-4 py-2 text-text-tertiary text-xs font-mono">
                    {img.width && img.height ? `${img.width}×${img.height}` : '—'}
                  </td>
                  <td className="px-4 py-2 text-center">
                    <span className={clsx(
                      'text-xs font-mono font-semibold',
                      img.n_regions > 0 ? 'text-green-400' : 'text-text-tertiary'
                    )}>
                      {img.n_regions}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-text-tertiary text-xs">
                    {img.indexed_at ? new Date(img.indexed_at * 1000).toLocaleDateString() : '—'}
                  </td>
                  <td className="px-4 py-2 font-mono text-[10px] text-text-tertiary max-w-[200px] truncate" title={img.path}>
                    {img.path}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-border bg-surface/50">
            <span className="text-xs text-text-tertiary">
              Page {page + 1} of {totalPages} · {total} total
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="p-1.5 rounded-lg bg-surface border border-border hover:border-primary-400 disabled:opacity-40 transition-all"
              >
                <ChevronLeft className="w-4 h-4 text-text-secondary" />
              </button>
              <button
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="p-1.5 rounded-lg bg-surface border border-border hover:border-primary-400 disabled:opacity-40 transition-all"
              >
                <ChevronRight className="w-4 h-4 text-text-secondary" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Image preview modal */}
      {preview && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
          onClick={() => setPreview(null)}
        >
          <div
            className="glass-card rounded-2xl border border-border p-6 max-w-2xl w-full mx-4 flex flex-col gap-4"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-sm text-white">{preview.filename}</span>
              <button onClick={() => setPreview(null)} className="text-text-tertiary hover:text-white transition-colors">✕</button>
            </div>
            <img src={preview.url!} alt={preview.filename} className="w-full max-h-96 object-contain rounded-xl bg-background" />
            {/* Metadata grid */}
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              {[
                ['Class', preview.class],
                ['Source', preview.source],
                ['Size', preview.width ? `${preview.width}×${preview.height}` : '—'],
                ['Regions', String(preview.n_regions)],
                ['Indexed', preview.indexed_at ? new Date(preview.indexed_at * 1000).toLocaleString() : '—'],
                ['Path', preview.path],
              ].map(([k, v]) => (
                <div key={k} className="flex flex-col gap-0.5">
                  <span className="text-text-tertiary uppercase tracking-wider text-[10px]">{k}</span>
                  <span className="text-text-secondary break-all">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ClassesPanel({ classes, loading }: { classes: ClassRow[]; loading: boolean }) {
  if (loading) return <Spinner />;
  if (!classes.length) return <Empty text="No class data" />;

  const bySource: Record<string, ClassRow[]> = {};
  classes.forEach(c => {
    if (!bySource[c.source]) bySource[c.source] = [];
    bySource[c.source].push(c);
  });

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader title="Classes by Source" icon={<Tag className="w-4 h-4" />} />
      {Object.entries(bySource).map(([src, rows]) => (
        <div key={src} className="glass-card rounded-xl p-4 border border-border/50">
          <div className="flex items-center gap-2 mb-3">
            <Badge label={src} color={SOURCE_COLORS[src]} />
            <span className="text-text-tertiary text-xs">{rows.length} classes · {rows.reduce((a, r) => a + r.count, 0)} images</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
            {rows.map(r => (
              <div key={r.class} className="flex items-center justify-between p-2.5 rounded-lg bg-background border border-border/40">
                <div className="flex items-center gap-2 overflow-hidden">
                  <FolderOpen className="w-3.5 h-3.5 text-primary-400 flex-shrink-0" />
                  <span className="text-xs text-white truncate" title={r.class}>{r.class}</span>
                </div>
                <span className="text-xs font-mono text-text-tertiary flex-shrink-0 ml-2">{r.count}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function PrototypesPanel({ data, loading }: { data: Prototype[]; loading: boolean }) {
  if (loading) return <Spinner />;
  if (!data.length) return <Empty text="No prototypes built yet. Run 'Train & Build Index' first." />;
  return (
    <div className="flex flex-col gap-4">
      <SectionHeader title="Class Prototypes" icon={<Braces className="w-4 h-4" />} />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {data.map(p => (
          <div key={p.class_name} className="glass-card rounded-xl p-4 border border-border/50 flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-primary-600/20 border border-primary-500/30 flex items-center justify-center flex-shrink-0">
              <CheckCircle2 className="w-5 h-5 text-primary-400" />
            </div>
            <div className="flex flex-col gap-0.5 overflow-hidden">
              <span className="text-white font-medium truncate">{p.class_name}</span>
              <span className="text-text-tertiary text-xs">
                {p.n_images} images · updated {new Date(p.updated_at * 1000).toLocaleString()}
              </span>
            </div>
            <div className="ml-auto">
              <span className="text-xs font-mono text-green-400 bg-green-500/10 border border-green-500/20 px-2 py-1 rounded-full">
                512-dim
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Shared utils ──────────────────────────────────────────────────────────

function Spinner() {
  return (
    <div className="flex items-center justify-center py-12">
      <RefreshCw className="w-6 h-6 text-primary-400 animate-spin" />
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="flex items-center justify-center py-12 text-text-tertiary text-sm">{text}</div>
  );
}


const FIXED_CLASSES = ['CFF with and without load', 'heavy drop', 'para motor', 'static line jump'];

function CleanupPanel({ classes, onRefresh }: { classes: ClassRow[]; onRefresh: () => void }) {
  const BASE = import.meta.env.VITE_API_BASE_URL || '';
  const [removing, setRemoving] = useState<string | null>(null);
  const [message, setMessage]   = useState<string | null>(null);

  // Find non-allowed classes
  const rogue = classes.filter(c => c.source === 'support' && !FIXED_CLASSES.includes(c.class));

  const handleRemove = async (className: string) => {
    if (!window.confirm(`Archive all images in "${className}" and remove from system?`)) return;
    setRemoving(className);
    setMessage(null);
    try {
      const res  = await fetch(`${BASE}/api/db/cleanup-class/${encodeURIComponent(className)}`, { method: 'DELETE' });
      const json = await res.json();
      setMessage(json.message);
      onRefresh();
    } catch {
      setMessage('Request failed.');
    } finally {
      setRemoving(null);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <SectionHeader title="Database Cleanup" icon={<Trash2 className="w-4 h-4" />} />

      <div className="glass-card rounded-xl p-4 border border-border/50">
        <SectionHeader title="Fixed Allowed Classes" icon={<CheckCircle2 className="w-4 h-4" />} />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {FIXED_CLASSES.map(cls => (
            <div key={cls} className="flex items-center gap-2 p-2.5 rounded-lg bg-green-500/10 border border-green-500/20">
              <CheckCircle2 className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
              <span className="text-xs text-green-300 truncate">{cls}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-card rounded-xl p-4 border border-border/50">
        <SectionHeader title="Non-allowed Classes (to remove)" icon={<AlertTriangle className="w-4 h-4" />} />
        {rogue.length === 0 ? (
          <div className="py-8 text-center text-green-400 text-sm">✓ No rogue classes found — database is clean.</div>
        ) : (
          <div className="flex flex-col gap-3">
            {rogue.map(r => (
              <div key={r.class} className="flex items-center justify-between p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                  <div>
                    <span className="text-white font-medium">{r.class}</span>
                    <span className="text-text-tertiary text-xs ml-2">({r.count} images · {r.source})</span>
                  </div>
                </div>
                <button
                  onClick={() => handleRemove(r.class)}
                  disabled={removing === r.class}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-600/20 border border-red-500/30 hover:bg-red-600/40 text-red-400 text-xs font-medium transition-all disabled:opacity-50"
                >
                  {removing === r.class ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                  Archive & Remove
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {message && (
        <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/30 text-blue-400 text-sm">
          {message}
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────

type Panel = 'qdrant' | 'images' | 'classes' | 'prototypes' | 'cleanup';

const PANELS: { id: Panel; label: string; icon: React.ReactNode }[] = [
  { id: 'qdrant',     label: 'Qdrant Store',   icon: <Box className="w-4 h-4" /> },
  { id: 'images',     label: 'Image Records',  icon: <ImageIcon className="w-4 h-4" /> },
  { id: 'classes',    label: 'Classes',         icon: <Tag className="w-4 h-4" /> },
  { id: 'prototypes', label: 'Prototypes',      icon: <Braces className="w-4 h-4" /> },
  { id: 'cleanup',    label: 'Cleanup',          icon: <Trash2 className="w-4 h-4" /> },
];

export function DBExplorer() {
  const [panel, setPanel]           = useState<Panel>('qdrant');
  const [collections, setCollections] = useState<CollectionsData | null>(null);
  const [classes, setClasses]       = useState<ClassRow[]>([]);
  const [prototypes, setPrototypes] = useState<Prototype[]>([]);
  const [loading, setLoading]       = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  const loadAll = useCallback(async () => {
    setLoading(true);
    try {
      const [col, cls, proto] = await Promise.all([
        fetch(`${BASE}/api/db/collections`).then(r => r.json()),
        fetch(`${BASE}/api/db/classes`).then(r => r.json()),
        fetch(`${BASE}/api/db/prototypes`).then(r => r.json()),
      ]);
      setCollections(col);
      setClasses(cls.classes ?? []);
      setPrototypes(proto.prototypes ?? []);
      setLastRefresh(new Date());
    } catch (err) {
      console.error('[DBExplorer]', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Database className="w-6 h-6 text-primary-400" />
          <div>
            <h2 className="text-2xl font-semibold text-white">DB Explorer</h2>
            <p className="text-text-tertiary text-sm">Qdrant vector store · SQLite metadata</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-text-tertiary text-xs flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5" />
            {lastRefresh.toLocaleTimeString()}
          </span>
          <button
            onClick={loadAll}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface border border-border hover:border-primary-400 text-text-secondary hover:text-white transition-all text-sm"
          >
            <RefreshCw className={clsx('w-3.5 h-3.5', loading && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      {/* Top-level stats strip */}
      {collections && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard icon={<Layers className="w-3.5 h-3.5" />} label="Total Vectors" value={collections.total} accent="text-green-400" sub="in Qdrant" />
          <StatCard icon={<Tag className="w-3.5 h-3.5" />} label="Classes" value={[...new Set(classes.map(c => c.class))].length} accent="text-primary-400" />
          <StatCard icon={<Braces className="w-3.5 h-3.5" />} label="Prototypes" value={prototypes.length} accent="text-accent" />
          <StatCard icon={<ImageIcon className="w-3.5 h-3.5" />} label="SQLite Records" value={classes.reduce((a, c) => a + c.count, 0)} accent="text-yellow-400" />
        </div>
      )}

      {/* Panel tabs */}
      <div className="flex bg-surface p-1 rounded-xl border border-border gap-1 w-fit">
        {PANELS.map(p => (
          <button
            key={p.id}
            onClick={() => setPanel(p.id)}
            className={twMerge(clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
              panel === p.id
                ? 'bg-primary-600 text-white shadow-lg'
                : 'text-text-secondary hover:text-white hover:bg-surface-hover'
            ))}
          >
            {p.icon}
            {p.label}
          </button>
        ))}
      </div>

      {/* Panel content */}
      <div className="glass-card rounded-2xl border border-border/50 p-6">
        {panel === 'qdrant'     && <QdrantPanel data={collections} loading={loading} />}
        {panel === 'images'     && <ImageBrowser classes={classes} />}
        {panel === 'classes'    && <ClassesPanel classes={classes} loading={loading} />}
        {panel === 'prototypes' && <PrototypesPanel data={prototypes} loading={loading} />}
        {panel === 'cleanup'    && <CleanupPanel classes={classes} onRefresh={loadAll} />}
      </div>
    </div>
  );
}