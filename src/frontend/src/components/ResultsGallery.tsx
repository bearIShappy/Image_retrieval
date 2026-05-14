import { useState } from 'react';
import { type RetrievalResult } from '../api';
import { Download, Maximize2, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface ResultsGalleryProps {
  results: RetrievalResult[];
  onExport: () => void;
  isExporting: boolean;
}

export function ResultsGallery({ results, onExport, isExporting }: ResultsGalleryProps) {
  const [selectedImage, setSelectedImage] = useState<RetrievalResult | null>(null);

  if (results.length === 0) return null;

  return (
    <div className="flex flex-col gap-6 w-full mt-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold flex items-center gap-2">
          Retrieval Results
          <span className="text-sm font-normal text-text-tertiary bg-surface px-3 py-1 rounded-full border border-border">
            {results.length} images
          </span>
        </h2>
        <button
          onClick={onExport}
          disabled={isExporting}
          className={twMerge(
            clsx(
              "flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all shadow-lg",
              isExporting 
                ? "bg-surface text-text-tertiary cursor-not-allowed border border-border" 
                : "bg-surface border border-border hover:border-primary-500 hover:text-primary-400 text-text-secondary"
            )
          )}
        >
          {isExporting ? (
            <div className="w-4 h-4 border-2 border-text-tertiary border-t-transparent rounded-full animate-spin" />
          ) : (
            <Download className="w-4 h-4" />
          )}
          {isExporting ? 'Exporting...' : 'Export Results'}
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        {results.map((result, idx) => (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: idx * 0.05 }}
            key={result.id}
            className="group relative rounded-2xl overflow-hidden glass-card aspect-square border border-border/50 hover:border-primary-500/50 transition-all cursor-pointer"
            onClick={() => setSelectedImage(result)}
          >
            <img 
              src={result.image_path} 
              alt={result.matched_class || 'Result'} 
              className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
              loading="lazy"
            />
            
            {/* BBox Overlay */}
            {result.bbox && (
              <div 
                className="absolute border-2 border-primary-500 bg-primary-500/10 transition-opacity"
                style={{
                  left: `${result.bbox[0]}%`,
                  top: `${result.bbox[1]}%`,
                  width: `${result.bbox[2] - result.bbox[0]}%`,
                  height: `${result.bbox[3] - result.bbox[1]}%`,
                }}
              />
            )}

            {/* Hover overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-background via-background/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col justify-end p-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium px-2 py-1 bg-primary-500/20 text-primary-300 rounded-md backdrop-blur-md border border-primary-500/30">
                  {(result.similarity * 100).toFixed(1)}%
                </span>
                <button className="p-1.5 bg-surface/50 hover:bg-surface text-text-secondary hover:text-text-primary rounded-lg backdrop-blur-md transition-colors">
                  <Maximize2 className="w-4 h-4" />
                </button>
              </div>
              {result.matched_class && (
                <p className="text-sm font-medium text-text-primary mt-2 truncate">
                  {result.matched_class}
                </p>
              )}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Modal */}
      <AnimatePresence>
        {selectedImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-xl"
            onClick={() => setSelectedImage(null)}
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="relative max-w-5xl w-full max-h-[90vh] glass-card rounded-2xl overflow-hidden shadow-2xl border border-border flex flex-col md:flex-row"
            >
              <button 
                onClick={() => setSelectedImage(null)}
                className="absolute top-4 right-4 z-10 p-2 bg-surface/50 hover:bg-surface text-text-secondary hover:text-text-primary rounded-full backdrop-blur-md transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
              
              <div className="w-full md:w-2/3 bg-black flex items-center justify-center relative p-4">
                <img 
                  src={selectedImage.image_path} 
                  alt="Enlarged result" 
                  className="max-w-full max-h-[80vh] object-contain rounded-lg"
                />
                {selectedImage.bbox && (
                  <div 
                    className="absolute border-2 border-primary-500 bg-primary-500/10"
                    style={{
                      left: `calc(1rem + ${selectedImage.bbox[0]}% * calc(100% - 2rem))`,
                      top: `calc(1rem + ${selectedImage.bbox[1]}% * calc(100% - 2rem))`,
                      width: `calc(${selectedImage.bbox[2] - selectedImage.bbox[0]}% * calc(100% - 2rem))`,
                      height: `calc(${selectedImage.bbox[3] - selectedImage.bbox[1]}% * calc(100% - 2rem))`,
                    }}
                  />
                )}
              </div>
              
              <div className="w-full md:w-1/3 p-6 flex flex-col gap-6 bg-surface/50">
                <h3 className="text-2xl font-bold text-text-primary">Image Details</h3>
                
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-1">
                    <span className="text-sm text-text-tertiary">Similarity Score</span>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-primary-600 to-primary-400" 
                          style={{ width: `${selectedImage.similarity * 100}%` }}
                        />
                      </div>
                      <span className="font-mono font-medium text-primary-400">
                        {(selectedImage.similarity * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                  
                  {selectedImage.matched_class && (
                    <div className="flex flex-col gap-1">
                      <span className="text-sm text-text-tertiary">Matched Class</span>
                      <span className="text-lg font-medium text-text-primary">
                        {selectedImage.matched_class}
                      </span>
                    </div>
                  )}

                  <div className="flex flex-col gap-1">
                    <span className="text-sm text-text-tertiary">Image ID</span>
                    <span className="font-mono text-sm text-text-secondary bg-surface px-3 py-2 rounded-lg border border-border">
                      {selectedImage.id}
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
