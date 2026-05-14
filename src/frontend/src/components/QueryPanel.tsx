import { useCallback, useState } from 'react';
import { Upload, X, Search } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface QueryPanelProps {
  onImageChange: (file: File | undefined) => void;
  onTextChange: (text: string) => void;
  onClear: () => void;
  imageFile: File | undefined;
  queryText: string;
}

export function QueryPanel({ onImageChange, onTextChange, onClear, imageFile, queryText }: QueryPanelProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onImageChange(e.dataTransfer.files[0]);
    }
  }, [onImageChange]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onImageChange(e.target.files[0]);
    }
  };

  const previewUrl = imageFile ? URL.createObjectURL(imageFile) : null;

  return (
    <div className="flex flex-col gap-6 w-full glass-card p-6 rounded-2xl transition-all duration-300">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold flex items-center gap-2">
          <Search className="w-5 h-5 text-accent" />
          Query Configuration
        </h2>
        <button
          onClick={onClear}
          className="text-sm px-3 py-1.5 rounded-lg border border-border hover:bg-surface-hover text-text-secondary hover:text-text-primary transition-colors flex items-center gap-1"
        >
          <X className="w-4 h-4" /> Clear All
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Text Search */}
        <div className="flex flex-col gap-3">
          <label className="text-sm font-medium text-text-secondary">Text Query</label>
          <div className="relative">
            <select
              value={queryText}
              onChange={(e) => onTextChange(e.target.value)}
              className="w-full bg-surface border border-border rounded-xl px-4 py-3 pl-10 text-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500 transition-all appearance-none cursor-pointer"
            >
              <option value="" disabled className="text-text-tertiary">Select a class...</option>
              <option value="heavy_drop">Heavy Drop</option>
              <option value="cff">CFF</option>
              <option value="static_line_jump">Static Line Jump</option>
              <option value="paramotor">Paramotor</option>
            </select>
            <Search className="absolute left-3 top-3.5 w-5 h-5 text-text-tertiary" />
            <div className="absolute inset-y-0 right-0 flex items-center pr-4 pointer-events-none">
              <svg className="w-4 h-4 text-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </div>

        {/* Image Upload */}
        <div className="flex flex-col gap-3">
          <label className="text-sm font-medium text-text-secondary">Image Query</label>
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={twMerge(
              clsx(
                "relative border-2 border-dashed rounded-xl h-32 flex flex-col items-center justify-center cursor-pointer overflow-hidden transition-all duration-200",
                isDragging ? "border-primary-400 bg-primary-500/10" : "border-border hover:border-text-tertiary bg-surface",
                previewUrl && "border-none"
              )
            )}
          >
            <input
              type="file"
              accept="image/*"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              onChange={handleFileChange}
            />
            {previewUrl ? (
              <>
                <img src={previewUrl} alt="Preview" className="absolute inset-0 w-full h-full object-cover opacity-60" />
                <div className="absolute inset-0 bg-background/40 backdrop-blur-sm z-0"></div>
                <img src={previewUrl} alt="Preview" className="h-full z-0 object-contain drop-shadow-2xl" />
                <div className="absolute top-2 right-2 z-20">
                  <button
                    onClick={(e) => { e.preventDefault(); e.stopPropagation(); onImageChange(undefined); }}
                    className="p-1 bg-surface/80 rounded-full hover:bg-surface border border-border"
                  >
                    <X className="w-4 h-4 text-text-secondary" />
                  </button>
                </div>
              </>
            ) : (
              <div className="flex flex-col items-center gap-2 text-text-tertiary">
                <Upload className={twMerge(clsx("w-6 h-6", isDragging && "text-primary-400 animate-bounce"))} />
                <p className="text-sm"><span className="text-primary-400 font-medium">Click to upload</span> or drag and drop</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}