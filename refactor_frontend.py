import os

# Update api.ts
api_file = "src/frontend/src/api.ts"
with open(api_file, "r", encoding="utf-8") as f:
    api_content = f.read()

# Add fromDate and toDate to RetrievalParams
api_content = api_content.replace(
    "useFinetuned: boolean;",
    "useFinetuned: boolean;\n  fromDate?: string;\n  toDate?: string;"
)

# Append to formData
form_data_orig = """    formData.append('threshold', params.threshold.toString());
    formData.append('use_regions', params.useRegions.toString());
    formData.append('use_finetuned', params.useFinetuned.toString());"""

form_data_new = """    formData.append('threshold', params.threshold.toString());
    formData.append('use_regions', params.useRegions.toString());
    formData.append('use_finetuned', params.useFinetuned.toString());
    if (params.fromDate) formData.append('from_date', new Date(params.fromDate).getTime() / 1000 + '');
    if (params.toDate) formData.append('to_date', new Date(params.toDate).getTime() / 1000 + '');"""
api_content = api_content.replace(form_data_orig, form_data_new)

with open(api_file, "w", encoding="utf-8") as f:
    f.write(api_content)

# Update RetrievalControls.tsx
rc_file = "src/frontend/src/components/RetrievalControls.tsx"
with open(rc_file, "r", encoding="utf-8") as f:
    rc_content = f.read()

# Add isTextOnly to props
rc_content = rc_content.replace(
    "interface RetrievalControlsProps {",
    "interface RetrievalControlsProps {\n  isTextOnly?: boolean;"
)

# Add isTextOnly to function signature
rc_content = rc_content.replace(
    "export function RetrievalControls({ params, onChange }: RetrievalControlsProps) {",
    "export function RetrievalControls({ params, onChange, isTextOnly }: RetrievalControlsProps) {"
)

# Update checkboxes to handle isTextOnly and tooltips
checkbox_orig = """        {/* Checkboxes */}
        <div className="flex flex-col justify-center gap-3 xl:col-span-4 lg:col-span-3 md:col-span-2">
          <div className="flex flex-wrap gap-6">
            <label className="flex items-center gap-3 cursor-pointer group">
              <div className="relative flex items-center">
                <input
                  type="checkbox"
                  checked={params.useRegions}
                  onChange={(e) => onChange({ useRegions: e.target.checked })}
                  className="peer sr-only"
                />"""

checkbox_new = """        {/* Date Filters */}
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
                />"""

rc_content = rc_content.replace(checkbox_orig, checkbox_new)

checkbox2_orig = """            <label className="flex items-center gap-3 cursor-pointer group">
              <div className="relative flex items-center">
                <input
                  type="checkbox"
                  checked={params.useFinetuned}
                  onChange={(e) => onChange({ useFinetuned: e.target.checked })}
                  className="peer sr-only"
                />"""

checkbox2_new = """            <label className="flex items-center gap-3 cursor-pointer group relative" title="Uses support examples for improved class-specific retrieval.">
              <div className="relative flex items-center">
                <input
                  type="checkbox"
                  checked={params.useFinetuned}
                  onChange={(e) => onChange({ useFinetuned: e.target.checked })}
                  className="peer sr-only"
                />"""

rc_content = rc_content.replace(checkbox2_orig, checkbox2_new)

with open(rc_file, "w", encoding="utf-8") as f:
    f.write(rc_content)

print("Updated api.ts and RetrievalControls.tsx")
