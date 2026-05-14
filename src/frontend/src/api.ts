// Use relative URLs in dev — Vite proxy forwards to localhost:5000
// In production, set VITE_API_BASE_URL to your backend's address
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export interface RetrievalParams {
  queryText?: string;
  queryImage?: File;
  topK: number;
  mode: 'global' | 'multi-query' | 'region-aware' | 'prototype';
  aggregation: 'max' | 'mean';
  threshold: number;
  useRegions: boolean;
  useFinetuned: boolean;
  fromDate?: string;
  toDate?: string;
  forcedClass?: string | null;
}

export interface RetrievalResult {
  id: string;
  image_path: string;
  similarity: number;
  matched_class?: string;
  bbox?: [number, number, number, number];
}

export interface Analytics {
  time_ms: number;
  mode: string;
  top_score: number;
  total_retrieved: number;
}

export interface DatasetClass {
  name: string;
  images: string[];
}

export interface MainDataset {
  support: DatasetClass[];
  test: string[];
}

export const api = {
  async getDataset(): Promise<MainDataset> {
    const response = await fetch(`${API_BASE_URL}/api/dataset`);
    if (!response.ok) throw new Error('Failed to fetch dataset');
    return response.json();
  },

  async search(params: RetrievalParams): Promise<{ results: RetrievalResult[], analytics: Analytics }> {
    // Fast path: text-only search directly against dataset folders (no ML needed)
    // Only when region-aware is OFF AND no forced class (forced class needs ML backend)
    if (params.queryText && !params.queryImage && !params.useRegions && !params.forcedClass) {
      const t_start = Date.now();
      const dataset = await this.getDataset();
      const query = params.queryText.toLowerCase();

      const results: RetrievalResult[] = [];
      dataset.support.forEach(cls => {
        if (cls.name.toLowerCase().includes(query) || query.includes(cls.name.toLowerCase())) {
          cls.images.forEach((img, idx) => {
            results.push({
              id: `${cls.name}-${idx}`,
              image_path: img,
              similarity: 1.0,
              matched_class: cls.name
            });
          });
        }
      });

      return {
        results: results.slice(0, params.topK),
        analytics: {
          time_ms: Date.now() - t_start,
          mode: params.mode,
          top_score: results.length > 0 ? 100.0 : 0,
          total_retrieved: Math.min(results.length, params.topK)
        }
      };
    }

    // Normal ML retrieval via backend
    const formData = new FormData();
    if (params.queryText) formData.append('text', params.queryText);
    if (params.queryImage) formData.append('image', params.queryImage);
    formData.append('top_k', params.topK.toString());
    formData.append('mode', params.mode);
    formData.append('aggregation', params.aggregation);
    formData.append('threshold', params.threshold.toString());
    formData.append('use_regions', params.useRegions.toString());
    formData.append('use_finetuned', params.useFinetuned.toString());
    if (params.fromDate) formData.append('from_date', new Date(params.fromDate).getTime() / 1000 + '');
    if (params.toDate) formData.append('to_date', new Date(params.toDate).getTime() / 1000 + '');
    if (params.forcedClass) formData.append('forced_class', params.forcedClass);

    const response = await fetch(`${API_BASE_URL}/api/search`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    return response.json();
  },

  async exportResults(resultsIds: string[]): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids: resultsIds })
    });
    if (!response.ok) throw new Error('Export failed');
  },

  async uploadSupportImages(files: FileList, classes: string[]): Promise<{ status: string; uploaded: number; failed: number; message: string }> {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
      formData.append('classes', classes[i]);
    }
    const response = await fetch(`${API_BASE_URL}/api/upload-support`, {
      method: 'POST',
      body: formData
    });
    if (!response.ok) throw new Error('Failed to upload support images');
    return response.json();
  },

  async getAllowedClasses(): Promise<{ allowed_classes: string[]; count: number }> {
    const response = await fetch(`${API_BASE_URL}/api/get-allowed-classes`);
    if (!response.ok) throw new Error('Failed to get allowed classes');
    return response.json();
  },

  async getPendingUploads(): Promise<{ total_pending: number; pending_by_class: Record<string, any[]> }> {
    const response = await fetch(`${API_BASE_URL}/api/get-pending-uploads`);
    if (!response.ok) throw new Error('Failed to get pending uploads');
    return response.json();
  },

  async finetuneModel(): Promise<{ status: string; message: string; newly_activated: number; qdrant_support: number; qdrant_total: number; prototypes: string[] }> {
    const response = await fetch(`${API_BASE_URL}/api/finetune`, {
      method: 'POST'
    });
    if (!response.ok) throw new Error('Fine-tuning failed');
    return response.json();
  },

  async rebuildIndex(): Promise<{ status: string; points: number; images: number }> {
    const response = await fetch(`${API_BASE_URL}/api/rebuild-index`, { method: 'POST' });
    if (!response.ok) throw new Error('Failed to rebuild index');
    return response.json();
  },

  async uploadTestImages(files: FileList): Promise<void> {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('images', files[i]);
    }
    const response = await fetch(`${API_BASE_URL}/api/upload-test`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw new Error('Failed to upload test images');
  },

  async deleteImage(path: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/api/delete-image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path })
    });
    if (!response.ok) throw new Error('Failed to delete image');
  },

  async getModels(): Promise<{ models: string[]; active_model: string; count: number }> {
    const response = await fetch(`${API_BASE_URL}/api/models`);
    if (!response.ok) throw new Error('Failed to fetch models');
    return response.json();
  }
};