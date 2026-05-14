import { useState, useRef, useEffect } from 'react';
import {
  Database, Upload, Wand2, Terminal, CheckCircle2,
  RefreshCw, AlertCircle, X, Tag, Clock, Image as ImageIcon
} from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const BASE = import.meta.env.VITE_API_BASE_URL || '';

const FIXED_CLASSES = [
  'CFF with and without load',
  'heavy drop',
  'para motor',
  'static line jump',
];

interface PendingFile {
  file: File;
  preview: string;
  assignedClass: string;
}

interface DatasetClass {
  name: string;
  images: string[];
}

interface PendingUpload {
  id: number;
  filename: string;
  class_name: string;
  status: string;
  staged_at: number;
}

type TrainStatus = 'idle' | 'training' | 'success' | 'error';

export function FewShotManagementPage() {
  const [classCounts, setClassCounts] = useState<Record<string, number>>({});
  const [pendingUploads, setPendingUploads] = useState<PendingUpload[]>([]);
  const [staged, setStaged] = useState<PendingFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [trainStatus, setTrainStatus] = useState<TrainStatus>('idle');
  const [trainProgress, setTrainProgress] = useState('');
  const [trainMessage, setTrainMessage] = useState('');
  const [activeModel, setActiveModel] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { loadData(); }, []);

  const loadData = async () => {
    try {
      const [dsRes, pendRes, modelsRes] = await Promise.all([
        fetch(`${BASE}/api/dataset`),
        fetch(`${BASE}/api/get-pending-uploads`),
        fetch(`${BASE}/api/models`)
      ]);
      if (dsRes.ok) {
        const ds = await dsRes.json();
        const counts: Record<string, number> = {};
        (ds.support as any[]).forEach(c => {
          if (FIXED_CLASSES.includes(c.name)) {
            // Count only images that are actually indexed (ACTIVE)
            const activeCount = c.images_detail ? c.images_detail.filter((i: any) => i.indexed).length : c.images.length;
            counts[c.name] = activeCount;
          }
        });
        setClassCounts(counts);
      }
      if (pendRes.ok) {
        const pend = await pendRes.json();
        const uploads: PendingUpload[] = pend.pending
          ?? (pend.pending_by_class
            ? (Object.values(pend.pending_by_class) as PendingUpload[][]).flat()
            : []);
        setPendingUploads(uploads);
      }
      if (modelsRes.ok) {
        const m = await modelsRes.json();
        setActiveModel(m.active_model);
      }
    } catch (err) { console.error('[FewShot] loadData failed', err); }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const newFiles: PendingFile[] = Array.from(e.target.files).map(file => ({
      file, preview: URL.createObjectURL(file), assignedClass: '',
    }));
    setStaged(prev => [...prev, ...newFiles]);
    setUploadError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const assignClass = (idx: number, cls: string) =>
    setStaged(prev => prev.map((f, i) => i === idx ? { ...f, assignedClass: cls } : f));
  const assignAll = (cls: string) =>
    setStaged(prev => prev.map(f => ({ ...f, assignedClass: cls })));
  const removeStaged = (idx: number) => {
    setStaged(prev => { URL.revokeObjectURL(prev[idx].preview); return prev.filter((_, i) => i !== idx); });
  };
  const clearStaged = () => {
    staged.forEach(f => URL.revokeObjectURL(f.preview));
    setStaged([]); setUploadError(null);
  };

  const handleUpload = async () => {
    const unassigned = staged.filter(f => !f.assignedClass);
    if (unassigned.length > 0) {
      setUploadError(`${unassigned.length} image${unassigned.length > 1 ? 's' : ''} still need a class assigned.`);
      return;
    }
    if (staged.length === 0) return;
    setIsUploading(true); setUploadError(null);
    try {
      const formData = new FormData();
      staged.forEach(f => formData.append('files', f.file));
      staged.forEach(f => formData.append('classes', f.assignedClass));
      const res = await fetch(`${BASE}/api/upload-support`, { method: 'POST', body: formData });
      const json = await res.json();
      if (!res.ok || json.status === 'error') {
        setUploadError(json.message ?? 'Upload failed');
        return;
      }

      clearStaged(); // This clears staged files AND resets uploadError to null
      await loadData();

      if (json.status === 'partial') {
        // Set it AFTER clearStaged so it doesn't get erased
        setUploadError(`Some files failed: ${json.failed_details.map((f: any) => f.filename + ' (' + f.reason + ')').join(', ')}`);
      }
    } catch {
      setUploadError('Network error during upload.');
    }
    finally {
      setIsUploading(false);
    }
  };

  const handleTrain = async () => {
    setTrainStatus('training'); setTrainProgress('Starting...'); setTrainMessage('');
    try {
      const res = await fetch(`${BASE}/api/finetune`, { method: 'POST' });
      const json = await res.json();
      if (json.status === 'error') { setTrainStatus('error'); setTrainMessage(json.message); return; }
      const poll = async () => {
        try {
          const sr = await fetch(`${BASE}/api/finetune/status`);
          const s = await sr.json();
          setTrainProgress(s.progress ?? '');
          if (s.status === 'running') { setTimeout(poll, 2000); }
          else if (s.status === 'success') {
            setTrainStatus('success'); setTrainMessage(s.message ?? 'Training complete.');
            await loadData();
            setTimeout(() => { setTrainStatus('idle'); setTrainMessage(''); }, 7000);
          } else { setTrainStatus('error'); setTrainMessage(s.message ?? 'Training failed.'); }
        } catch { setTimeout(poll, 3000); }
      };
      setTimeout(poll, 2000);
    } catch { setTrainStatus('error'); setTrainMessage('Failed to start training.'); }
  };

  const handleCancel = async () => {
    if (!confirm('Are you sure you want to cancel the training?')) return;
    try {
      await fetch(`${BASE}/api/finetune/cancel`, { method: 'POST' });
    } catch (err) {
      console.error('Failed to cancel', err);
    }
  };

  const allAssigned = staged.length > 0 && staged.every(f => f.assignedClass);
  const unassignedCount = staged.filter(f => !f.assignedClass).length;
  const pendingByClass: Record<string, number> = {};
  pendingUploads.forEach(p => { pendingByClass[p.class_name] = (pendingByClass[p.class_name] ?? 0) + 1; });

  return (
    <div className="flex flex-col gap-8 animate-in fade-in duration-500 max-w-5xl mx-auto w-full">
      <div className="flex flex-col gap-1">
        <h2 className="text-3xl font-bold text-white flex items-center gap-3">
          <Database className="w-8 h-8 text-purple-400" /> Few-shot Management
        </h2>
        <p className="text-text-secondary">Upload images to the 4 fixed support classes, then train to activate them.</p>
      </div>

      {/* Section A: Active dataset */}
      <div className="glass-card p-6 rounded-2xl border-t-4 border-t-purple-500 flex flex-col gap-4">
        <h3 className="text-xl font-semibold text-white">Section A: Active Support Dataset</h3>
        <p className="text-text-secondary text-sm">Currently active, trained images per class.</p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {FIXED_CLASSES.map(cls => (
            <div key={cls} className="flex flex-col items-center justify-center p-4 rounded-xl bg-background border border-border gap-1">
              <span className="font-medium text-white text-center text-sm leading-tight">{cls}</span>
              <span className="text-2xl font-bold text-primary-400">{classCounts[cls] ?? 0}</span>
              <span className="text-xs text-text-tertiary">active images</span>
              {pendingByClass[cls] > 0 && (
                <span className="text-xs text-yellow-400 bg-yellow-500/10 border border-yellow-500/20 px-2 py-0.5 rounded-full mt-1">
                  +{pendingByClass[cls]} pending
                </span>
              )}
            </div>
          ))}
        </div>
        {pendingUploads.length > 0 && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-sm">
            <Clock className="w-4 h-4 flex-shrink-0" />
            <span><strong>{pendingUploads.length}</strong> image{pendingUploads.length !== 1 ? 's' : ''} pending training. Click "Train & Build Index" below.</span>
          </div>
        )}
      </div>

      {/* Section B: Upload */}
      <div className="glass-card p-6 rounded-2xl border-t-4 border-t-blue-500 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-semibold text-white">Section B: Upload New Images</h3>
            <p className="text-text-secondary text-sm mt-1">Each image must be assigned to one of the 4 fixed classes.</p>
          </div>
          <button onClick={() => fileInputRef.current?.click()} disabled={isUploading}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-blue-500/30 bg-blue-600/20 hover:bg-blue-600/30 transition-all text-white font-medium disabled:opacity-50">
            <Upload className="w-4 h-4" /> Select Images
          </button>
          <input type="file" multiple accept="image/*" ref={fileInputRef} className="hidden" onChange={handleFileSelect} />
        </div>

        {staged.length > 0 ? (
          <div className="flex flex-col gap-4">
            {/* Bulk assign bar */}
            <div className="flex items-center gap-3 p-3 rounded-xl bg-surface border border-border flex-wrap">
              <Tag className="w-4 h-4 text-text-tertiary flex-shrink-0" />
              <span className="text-sm text-text-secondary">Assign all to:</span>
              {FIXED_CLASSES.map(cls => (
                <button key={cls} onClick={() => assignAll(cls)}
                  className="px-3 py-1 rounded-lg text-xs font-medium border border-border bg-background hover:border-primary-400 hover:text-white text-text-secondary transition-all">
                  {cls}
                </button>
              ))}
              <button onClick={clearStaged} className="ml-auto text-text-tertiary hover:text-red-400 transition-colors text-xs">Clear all</button>
            </div>

            {/* Image grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 max-h-[480px] overflow-y-auto custom-scrollbar pr-1">
              {staged.map((f, idx) => (
                <div key={idx} className={twMerge(clsx(
                  "relative flex flex-col rounded-xl border overflow-hidden transition-all",
                  f.assignedClass ? "border-primary-500/50 bg-primary-500/5" : "border-red-500/30 bg-surface"
                ))}>
                  <div className="aspect-video bg-background overflow-hidden">
                    <img src={f.preview} alt={f.file.name} className="w-full h-full object-cover" />
                  </div>
                  <div className="flex items-center justify-between px-3 pt-2">
                    <span className="text-xs text-text-tertiary truncate">{f.file.name}</span>
                    <button onClick={() => removeStaged(idx)} className="text-text-tertiary hover:text-red-400 transition-colors ml-2 flex-shrink-0">
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <div className="p-2">
                    <select value={f.assignedClass} onChange={e => assignClass(idx, e.target.value)}
                      className={twMerge(clsx(
                        "w-full bg-background border rounded-lg px-2 py-1.5 text-xs outline-none transition-all",
                        f.assignedClass ? "border-primary-500/50 text-white" : "border-red-500/40 text-text-tertiary"
                      ))}>
                      <option value="">— assign class —</option>
                      {FIXED_CLASSES.map(cls => <option key={cls} value={cls}>{cls}</option>)}
                    </select>
                  </div>
                  {f.assignedClass && (
                    <div className="absolute top-2 right-2 bg-primary-600/90 text-white text-[10px] px-1.5 py-0.5 rounded-full">✓</div>
                  )}
                </div>
              ))}
            </div>

            {uploadError && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
                <AlertCircle className="w-4 h-4 flex-shrink-0" /> {uploadError}
              </div>
            )}

            <div className="flex items-center justify-between pt-2 border-t border-border">
              <div className="text-sm text-text-secondary">
                <span className="text-white font-medium">{staged.length}</span> staged
                {unassignedCount > 0 && <span className="text-red-400 ml-2">· {unassignedCount} unassigned</span>}
                {allAssigned && <span className="text-green-400 ml-2">· all assigned ✓</span>}
              </div>
              <button onClick={handleUpload} disabled={isUploading || !allAssigned}
                className={twMerge(clsx(
                  "flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-all",
                  allAssigned && !isUploading
                    ? "bg-blue-600 hover:bg-blue-500 text-white shadow-lg"
                    : "bg-surface border border-border text-text-tertiary cursor-not-allowed"
                ))}>
                {isUploading
                  ? <><RefreshCw className="w-4 h-4 animate-spin" /> Uploading...</>
                  : <><Upload className="w-4 h-4" /> Upload {staged.length} Image{staged.length !== 1 ? 's' : ''}</>
                }
              </button>
            </div>
          </div>
        ) : (
          <div onClick={() => fileInputRef.current?.click()}
            className="flex flex-col items-center justify-center h-32 rounded-xl border-2 border-dashed border-border hover:border-blue-400 transition-all cursor-pointer text-text-tertiary hover:text-text-secondary">
            <ImageIcon className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-sm">Click to select images</p>
          </div>
        )}
      </div>

      {/* Section C: Training */}
      <div className="glass-card p-6 rounded-2xl border-t-4 border-t-orange-500 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-semibold text-white">Section C: Training Panel</h3>
            <p className="text-text-secondary text-sm mt-1">Generates embeddings and activates pending images in Qdrant.</p>
          </div>
          <div className={twMerge(clsx(
            "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium border transition-all",
            trainStatus === 'idle' && "bg-surface text-text-tertiary border-border",
            trainStatus === 'training' && "bg-orange-500/10 text-orange-400 border-orange-500/30",
            trainStatus === 'success' && "bg-green-500/10 text-green-400 border-green-500/30",
            trainStatus === 'error' && "bg-red-500/10 text-red-400 border-red-500/30",
          ))}>
            {trainStatus === 'idle' && "Status: Idle"}
            {trainStatus === 'training' && <><RefreshCw className="w-4 h-4 animate-spin" /> Training...</>}
            {trainStatus === 'success' && <><CheckCircle2 className="w-4 h-4" /> Done ✓</>}
            {trainStatus === 'error' && <><AlertCircle className="w-4 h-4" /> Error</>}
          </div>
        </div>

        {pendingUploads.length < 20 && (
          <div className="text-yellow-400 bg-yellow-500/10 p-3 rounded-lg border border-yellow-500/30 text-sm flex items-center gap-2 mt-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>Add at least 20 new support images to enable fine-tuning. Currently have {pendingUploads.length}.</span>
          </div>
        )}
        <div className="flex items-center gap-3 mt-2">
          <button onClick={handleTrain} disabled={trainStatus === 'training' || pendingUploads.length < 20}
            className="flex items-center justify-center gap-2 px-8 py-4 rounded-xl border border-orange-500/30 bg-orange-600/20 hover:bg-orange-600/30 transition-all disabled:opacity-50">
            {trainStatus === 'training'
              ? <RefreshCw className="w-5 h-5 text-orange-400 animate-spin" />
              : <Wand2 className="w-5 h-5 text-orange-400" />}
            <span className="font-semibold text-white text-lg">Train & Build Index</span>
          </button>
          
          <button onClick={handleCancel} disabled={trainStatus !== 'training'}
            className={twMerge(clsx(
              "flex items-center justify-center gap-2 px-6 py-4 rounded-xl border transition-all",
              trainStatus === 'training'
                ? "border-red-500/30 bg-red-600/20 hover:bg-red-600/30 text-red-400"
                : "border-border bg-surface text-text-tertiary cursor-not-allowed opacity-50"
            ))}>
            <X className="w-5 h-5" />
            <span className="font-semibold text-lg">Cancel</span>
          </button>
        </div>

        {activeModel && (
          <div className="mt-2 text-sm text-text-tertiary">
            <span className="text-white font-medium">Active Model:</span> {activeModel}
          </div>
        )}

        <div className="p-4 rounded-xl bg-[#0d1117] border border-[#30363d] font-mono text-sm">
          <div className="flex items-center gap-2 text-text-tertiary mb-3 pb-2 border-b border-[#30363d]">
            <Terminal className="w-4 h-4" /><span>Training Log</span>
          </div>
          <code className={clsx("text-green-400 break-all block", trainStatus !== 'training' && "opacity-50")}>
            $ python app.py --finetune --rebuild-index --source cleaned_dataset
          </code>
          {trainStatus === 'training' && trainProgress && <div className="text-blue-400 mt-2 animate-pulse">&gt; {trainProgress}</div>}
          {trainStatus === 'training' && !trainProgress && (
            <div className="text-text-tertiary mt-2 space-y-1 animate-pulse">
              <div>&gt; Loading dataset...</div><div>&gt; Initializing visual backbone...</div>
            </div>
          )}
          {trainStatus === 'success' && <div className="text-green-400 mt-2">&gt; ✓ {trainMessage}</div>}
          {trainStatus === 'error' && <div className="text-red-400 mt-2">&gt; ✗ {trainMessage}</div>}
        </div>
      </div>
    </div>
  );
}