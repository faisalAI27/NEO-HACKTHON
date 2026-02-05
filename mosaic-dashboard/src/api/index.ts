import axios from 'axios'
import { PredictionRequest, PredictionResult, HealthStatus } from '../types'

// API Base URL - proxied through Vite in development
const API_BASE = '/api'
const WSI_BASE = '/wsi'

/**
 * Cancer detection result type
 */
export interface CancerDetectionResult {
  slide_id: string;
  is_cancerous: boolean;
  confidence: number;
  cancer_type?: string;
  tumor_regions?: number;
  analysis_time_seconds: number;
}

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
})

/**
 * Health check endpoint
 */
export async function getHealthStatus(): Promise<HealthStatus> {
  const response = await api.get<HealthStatus>('/health')
  return response.data
}

/**
 * Submit prediction request
 */
export async function predict(request: PredictionRequest): Promise<PredictionResult> {
  const response = await api.post<PredictionResult>('/predict', request)
  return response.data
}

/**
 * Analyze WSI for cancer detection (Stage 1)
 */
export async function detectCancer(slideId: string): Promise<CancerDetectionResult> {
  const response = await api.post<CancerDetectionResult>('/analyze/cancer-detection', { slide_id: slideId })
  return response.data
}

/**
 * Upload response type
 */
export interface UploadResponse {
  status: string;
  slide_id: string;
  filename: string;
  size_bytes: number;
  size_mb: number;
  path: string;
}

/**
 * Upload a WSI file with progress tracking
 */
export async function uploadWSIFile(
  file: File,
  onProgress?: (progress: number) => void
): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await axios.post<UploadResponse>(`${API_BASE}/upload/wsi`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && onProgress) {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        onProgress(percentCompleted)
      }
    },
  })

  return response.data
}

/**
 * Check upload status for a slide
 */
export async function getUploadStatus(slideId: string): Promise<{ exists: boolean; slide_id: string; size_mb?: number }> {
  const response = await api.get(`/upload/status/${slideId}`)
  return response.data
}

/**
 * List uploaded files
 */
export async function listUploadedFiles(): Promise<{ files: Array<{ slide_id: string; filename: string; size_mb: number }> }> {
  const response = await api.get('/upload/list')
  return response.data
}

/**
 * Get list of available WSI slides
 */
export async function getSlideList(): Promise<string[]> {
  const response = await axios.get<{ slides: string[] }>(`${WSI_BASE}/list`)
  return response.data.slides
}

/**
 * Get DZI metadata URL for a slide
 */
export function getDziUrl(slideId: string): string {
  return `${WSI_BASE}/${slideId}/dzi.xml`
}

/**
 * Get thumbnail URL for a slide
 */
export function getThumbnailUrl(slideId: string, maxWidth = 256, maxHeight = 256): string {
  return `${WSI_BASE}/${slideId}/thumbnail?max_width=${maxWidth}&max_height=${maxHeight}`
}

/**
 * Get tile URL pattern for OpenSeadragon
 */
export function getTileUrlTemplate(slideId: string): string {
  return `${WSI_BASE}/${slideId}/deepzoom/{level}/{col}_{row}.jpg`
}

export default api
