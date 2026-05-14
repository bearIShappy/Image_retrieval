import { useState, useRef, useEffect } from 'react';
import { Layers, Upload, RefreshCw, Image as ImageIcon } from 'lucide-react';
import { api, type MainDataset } from '../api';

export function TestDatasetManager() {
  const [isUploading, setIsUploading] = useState(false);
  const [dataset, setDataset] = useState<MainDataset | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDataset();
  }, []);

  const loadDataset = async () => {
    try {
      const data = await api.getDataset();
      setDataset(data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setIsUploading(true);
      try {
        await api.uploadTestImages(e.target.files);
        await loadDataset();
        // The backend automatically triggers index rebuild for test dataset on upload
      } catch (err) {
        alert("Failed to upload test images.");
      } finally {
        setIsUploading(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      }
    }
  };

  const testImages = dataset?.test || [];

  return (
    <div className="flex flex-col gap-4 w-full glass-card p-6 rounded-2xl border-l-4 border-l-accent">
      <div className="flex items-center gap-2">
        <Layers className="w-5 h-5 text-accent" />
        <h2 className="text-xl font-semibold">Test Dataset</h2>
      </div>
      <p className="text-sm text-text-tertiary">
        Upload raw test images to evaluate the retrieval system. These are added to the main dataset.
      </p>

      <div className="mt-2">
        <button 
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          className="flex w-full items-center justify-center gap-2 p-4 rounded-xl border border-border bg-surface hover:bg-surface-hover hover:border-accent transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isUploading ? (
            <RefreshCw className="w-6 h-6 text-accent animate-spin" />
          ) : (
            <Upload className="w-6 h-6 text-text-tertiary group-hover:text-accent transition-colors" />
          )}
          <span className="font-medium">Upload Test Images</span>
        </button>
        <input 
          type="file" 
          multiple 
          accept="image/*" 
          ref={fileInputRef}
          className="hidden" 
          onChange={handleUpload}
        />
      </div>

      {/* Uploaded images thumbnail strip */}
      {testImages.length > 0 && (
        <div className="mt-2 flex flex-col gap-2">
          <div className="flex justify-between items-center text-xs text-text-tertiary uppercase font-medium">
            <span>Uploaded Images</span>
            <span>{testImages.length} count</span>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-2 custom-scrollbar">
            {testImages.map((img, idx) => (
              <div key={idx} className="flex-shrink-0 w-16 h-16 rounded-lg overflow-hidden border border-border bg-background">
                <img src={img} alt={`Test ${idx}`} className="w-full h-full object-cover" loading="lazy" />
              </div>
            ))}
            {testImages.length === 0 && (
              <div className="w-16 h-16 flex items-center justify-center rounded-lg border border-dashed border-border text-text-tertiary">
                <ImageIcon className="w-6 h-6 opacity-50" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
