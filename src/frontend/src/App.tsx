import { useState } from 'react';
import { QueryPanel } from './components/QueryPanel';
import { RetrievalControls } from './components/RetrievalControls';
import { ResultsGallery } from './components/ResultsGallery';
import { AnalyticsPanel } from './components/AnalyticsPanel';
import { TestDatasetManager } from './components/TestDatasetManager';
import { MainDatasetViewer } from './components/MainDatasetViewer';
import { FewShotManagementPage } from './components/FewShotManagementPage';
import { DatasetOverview } from './components/DatasetOverview';
import { api, type RetrievalParams, type RetrievalResult, type Analytics } from './api';
import { Search as SearchIcon, AlertCircle, Database, Wand2, HardDrive } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function App() {
  const [activeTab, setActiveTab] = useState<'search' | 'dataset' | 'finetune' | 'dbexplorer'>('search');

  const [queryText, setQueryText] = useState('');
  const [queryImage, setQueryImage] = useState<File | undefined>();

  const [params, setParams] = useState<RetrievalParams>({
    topK: 20,
    mode: 'global',
    aggregation: 'max',
    threshold: 0.5,
    useRegions: true,
    useFinetuned: false,
    forcedClass: null,
  });

  const [isSearching, setIsSearching] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleClear = () => {
    setQueryText('');
    setQueryImage(undefined);
    setResults([]);
    setAnalytics(null);
    setError(null);
  };

  const handleSearch = async () => {
    if (!queryText && !queryImage && !params.forcedClass) {
      setError("Please provide an image query, a text query, or select a class filter.");
      return;
    }

    setIsSearching(true);
    setError(null);

    try {
      const response = await api.search({
        ...params,
        queryText: params.forcedClass ? undefined : queryText,  // ← bypass text if class forced
        queryImage
      });
      setResults(response.results);
      setAnalytics(response.analytics);
    } catch (err) {
      setError("An error occurred while fetching retrieval results.");
      console.error(err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleExport = async () => {
    if (results.length === 0) return;
    setIsExporting(true);
    try {
      await api.exportResults(results.map(r => r.id));
    } catch (err) {
      console.error("Failed to export results", err);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-text-primary custom-scrollbar selection:bg-primary-500/30 selection:text-white">
      {/* Background decoration */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary-600/10 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-accent/5 rounded-full blur-3xl" />
      </div>

      <main className="relative z-10 container mx-auto px-4 py-8 flex flex-col gap-8 max-w-7xl">
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-border">
          <div className="flex flex-col gap-1">
            <h1 className="text-4xl font-bold text-gradient inline-block tracking-tight">
              CLIP Vision System
            </h1>
            <p className="text-text-secondary text-lg">
              Few-Shot Image Retrieval & Fine-tuning Platform
            </p>
          </div>

          <div className="flex bg-surface p-1 rounded-xl border border-border">
            <button
              onClick={() => setActiveTab('search')}
              className={twMerge(
                clsx(
                  "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all",
                  activeTab === 'search' ? "bg-primary-600 text-white shadow-lg" : "text-text-secondary hover:text-text-primary hover:bg-surface-hover"
                )
              )}
            >
              <SearchIcon className="w-4 h-4" /> Retrieval
            </button>
            <button
              onClick={() => setActiveTab('dataset')}
              className={twMerge(
                clsx(
                  "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all",
                  activeTab === 'dataset' ? "bg-primary-600 text-white shadow-lg" : "text-text-secondary hover:text-text-primary hover:bg-surface-hover"
                )
              )}
            >
              <Database className="w-4 h-4" /> Main Dataset
            </button>
            <button
              onClick={() => setActiveTab('finetune')}
              className={twMerge(
                clsx(
                  "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all",
                  activeTab === 'finetune' ? "bg-primary-600 text-white shadow-lg" : "text-text-secondary hover:text-text-primary hover:bg-surface-hover"
                )
              )}
            >
              <Wand2 className="w-4 h-4" /> Few-shot Management
            </button>
            <button
              onClick={() => setActiveTab('dbexplorer')}
              className={twMerge(
                clsx(
                  "flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all",
                  activeTab === 'dbexplorer' ? "bg-primary-600 text-white shadow-lg" : "text-text-secondary hover:text-text-primary hover:bg-surface-hover"
                )
              )}
            >
              <HardDrive className="w-4 h-4" /> Dataset Overview
            </button>
          </div>
        </header>

        {error && (
          <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/30 text-red-400 rounded-xl animate-pulse">
            <AlertCircle className="w-5 h-5" />
            <span className="font-medium">{error}</span>
          </div>
        )}

        {activeTab === 'search' && (
          <>
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
              <div className="xl:col-span-2 flex flex-col gap-8">
                <QueryPanel
                  onImageChange={setQueryImage}
                  onTextChange={setQueryText}
                  onClear={handleClear}
                  imageFile={queryImage}
                  queryText={queryText}
                  selectedClass={params.forcedClass}
                />
                <RetrievalControls
                  params={params}
                  onChange={(newParams) => setParams(prev => ({ ...prev, ...newParams }))}
                  isTextOnly={!!queryText && !queryImage}
                />
              </div>

              <div className="xl:col-span-1 flex flex-col gap-4">
                <TestDatasetManager />
                <button
                  onClick={() => setActiveTab('dataset')}
                  className="flex items-center justify-center gap-2 p-4 rounded-xl border border-border bg-surface hover:bg-surface-hover hover:border-primary-400 transition-all text-text-secondary hover:text-white group"
                >
                  <Database className="w-5 h-5 group-hover:text-primary-400 transition-colors" />
                  <span className="font-medium">View Training Dataset</span>
                </button>
              </div>
            </div>

            {/* Action Bar */}
            <div className="sticky bottom-4 z-40 glass-panel p-4 rounded-2xl flex justify-center mt-4">
              <button
                onClick={handleSearch}
                disabled={isSearching}
                className={twMerge(
                  clsx(
                    "relative flex items-center justify-center gap-3 w-full max-w-md px-8 py-4 rounded-xl text-lg font-semibold text-white shadow-xl transition-all duration-300 overflow-hidden group",
                    isSearching
                      ? "bg-surface border border-border cursor-not-allowed"
                      : "bg-primary-600 hover:bg-primary-500 hover:shadow-primary-500/25 border border-primary-400/30"
                  )
                )}
              >
                {isSearching ? (
                  <>
                    <div className="w-6 h-6 border-3 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
                    <SearchIcon className="w-6 h-6" />
                    <span>Execute Search</span>
                  </>
                )}
              </button>
            </div>

            {results.length > 0 && !isSearching && (
              <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <AnalyticsPanel analytics={analytics} />
                <ResultsGallery
                  results={results}
                  onExport={handleExport}
                  isExporting={isExporting}
                />
              </div>
            )}
          </>
        )}

        {activeTab === 'dataset' && (
          <MainDatasetViewer />
        )}

        {activeTab === 'finetune' && (
          <FewShotManagementPage />
        )}

        {activeTab === 'dbexplorer' && (
          <DatasetOverview />
        )}
      </main>
    </div>
  );
}

export default App;