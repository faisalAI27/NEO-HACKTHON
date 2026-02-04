import axios from 'axios'
import { PredictionRequest, PredictionResult, HealthStatus } from '../types'

// API Base URL - proxied through Vite in development
const API_BASE = '/api'
const WSI_BASE = '/wsi'

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
