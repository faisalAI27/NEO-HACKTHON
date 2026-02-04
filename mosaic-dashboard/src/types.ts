// Type definitions for MOSAIC Dashboard

export interface ClinicalData {
  age?: number;
  gender?: 'male' | 'female' | 'unknown';
  tumor_stage?: string;
  hpv_status?: boolean;
  smoking_history?: number;
  alcohol_history?: boolean;
  tumor_site?: string;
}

export interface OmicsData {
  gene_expression?: number[];
  gene_names?: string[];
  methylation?: number[];
  probe_ids?: string[];
  mutated_genes?: string[];
  driver_mutations?: string[];
}

export interface WSIData {
  slide_id?: string;
  features?: number[][];
}

export interface PredictionRequest {
  patient_id?: string;
  clinical?: ClinicalData;
  omics?: OmicsData;
  wsi?: WSIData;
  time_points?: number[];
  return_attention?: boolean;
}

export interface SurvivalProbabilities {
  [time: string]: number;
}

export interface AttentionMaps {
  cross_modal?: number[][] | Record<string, number[][]>;
  wsi?: number[][];
  rna?: number[];
  clinical?: number[];
  modality_importance?: Record<string, number>;
}

export interface GeneImportance {
  gene: string;
  importance: number;
}

export interface PredictionResult {
  patient_id: string;
  risk_score: number;
  risk_group?: 'low' | 'medium' | 'high';
  survival_probability: SurvivalProbabilities;
  survival_probabilities?: SurvivalProbabilities;
  confidence_interval?: { lower: number; upper: number };
  attention_weights?: Record<string, number>;
  gene_importance?: GeneImportance[];
  attention_maps?: AttentionMaps;
  modalities_used?: string[];
  model_version?: string;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  available_modalities: string[];
  wsi_server_available: boolean;
  version: string;
}

export interface SurvivalDataPoint {
  time: number;
  probability: number;
  lower?: number;
  upper?: number;
}
