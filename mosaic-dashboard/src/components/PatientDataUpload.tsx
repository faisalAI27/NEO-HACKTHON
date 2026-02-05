import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Box,
  Button,
  Grid,
  Typography,
  Paper,
  Alert,
  CircularProgress,
  Chip,
  LinearProgress,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableRow,
  Divider,
} from '@mui/material'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import SendIcon from '@mui/icons-material/Send'
import DescriptionIcon from '@mui/icons-material/Description'
import BiotechIcon from '@mui/icons-material/Biotech'
import LocalHospitalIcon from '@mui/icons-material/LocalHospital'
import ScienceIcon from '@mui/icons-material/Science'
import { PredictionResult } from '../types'

interface PatientDataUploadProps {
  onPredictionComplete: (result: PredictionResult) => void
  onSlideSelected: (slideId: string | null) => void
}

interface ParsedClinicalData {
  patient_id: string
  age: number | null
  gender: string | null
  vital_status: string | null
  tumor_stage: string | null
  tumor_site: string | null
  days_to_follow_up: number | null
  hpv_status?: string | null
  smoking_pack_years?: number | null
  treatments: string[]
}

interface ParsedMutationData {
  patient_id: string
  total_mutations: number
  mutated_genes: string[]
  driver_mutations: string[]
  variant_types: Record<string, number>
}

interface FileUploadState<T = unknown> {
  file: File | null
  status: 'idle' | 'uploading' | 'parsed' | 'error'
  error: string | null
  data: T | null
}

interface OmicsFileData {
  genes: number
  samples?: number
}

interface WSIFileData {
  slide_id: string
}

const DRIVER_GENES = [
  'TP53', 'CDKN2A', 'PIK3CA', 'NOTCH1', 'FAT1', 'CASP8', 'NSD1', 'HRAS',
  'EGFR', 'PTEN', 'NFE2L2', 'KEAP1', 'RB1', 'FBXW7', 'EP300', 'CREBBP'
]

export default function PatientDataUpload({
  onPredictionComplete,
  onSlideSelected,
}: PatientDataUploadProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // File states
  const [clinicalFile, setClinicalFile] = useState<FileUploadState<ParsedClinicalData>>({
    file: null, status: 'idle', error: null, data: null
  })
  const [mutationFile, setMutationFile] = useState<FileUploadState<ParsedMutationData>>({
    file: null, status: 'idle', error: null, data: null
  })
  const [rnaFile, setRnaFile] = useState<FileUploadState<OmicsFileData>>({
    file: null, status: 'idle', error: null, data: null
  })
  const [methylationFile, setMethylationFile] = useState<FileUploadState<OmicsFileData>>({
    file: null, status: 'idle', error: null, data: null
  })
  const [wsiFile, setWsiFile] = useState<FileUploadState<WSIFileData>>({
    file: null, status: 'idle', error: null, data: null
  })

  // Parse TSV file
  const parseTSV = (content: string): { headers: string[], rows: string[][] } => {
    const lines = content.trim().split('\n')
    const headers = lines[0].split('\t')
    const rows = lines.slice(1).map(line => line.split('\t'))
    return { headers, rows }
  }

  // Parse clinical data from TSV
  const parseClinicalData = (content: string): ParsedClinicalData => {
    const { headers, rows } = parseTSV(content)
    
    // Get first patient row (or aggregate)
    const row = rows[0] || []
    const getValue = (colName: string): string | null => {
      const idx = headers.findIndex(h => h.toLowerCase().includes(colName.toLowerCase()))
      if (idx === -1) return null
      const val = row[idx]
      return val && val !== "'--" && val !== '--' ? val : null
    }

    // Extract key fields
    const patientId = getValue('submitter_id') || getValue('case_id') || 'Unknown'
    const age = getValue('age_at_index') || getValue('age_at_diagnosis')
    const gender = getValue('gender')
    const vitalStatus = getValue('vital_status')
    const tumorStage = getValue('ajcc_pathologic_stage') || getValue('ajcc_clinical_stage')
    const tumorSite = getValue('primary_site') || getValue('tissue_or_organ_of_origin')
    const daysToFollowUp = getValue('days_to_last_follow_up') || getValue('days_to_death')
    const treatments = [
      getValue('treatment_type'),
      getValue('therapeutic_agents')
    ].filter(Boolean) as string[]

    return {
      patient_id: patientId,
      age: age ? parseInt(age) : null,
      gender: gender,
      vital_status: vitalStatus,
      tumor_stage: tumorStage,
      tumor_site: tumorSite,
      days_to_follow_up: daysToFollowUp ? parseFloat(daysToFollowUp) : null,
      treatments: treatments,
    }
  }

  // Parse mutation data from MAF/TSV
  const parseMutationData = (content: string): ParsedMutationData => {
    const { headers, rows } = parseTSV(content)
    
    const hugoIdx = headers.findIndex(h => h === 'Hugo_Symbol')
    const variantIdx = headers.findIndex(h => h === 'Variant_Classification')
    const sampleIdx = headers.findIndex(h => h.includes('Tumor_Sample_Barcode'))
    
    const mutatedGenes = new Set<string>()
    const variantTypes: Record<string, number> = {}
    const driverMutations: string[] = []
    
    let patientId = 'Unknown'
    
    rows.forEach(row => {
      const gene = row[hugoIdx]
      const variant = row[variantIdx]
      const sample = row[sampleIdx]
      
      if (sample && patientId === 'Unknown') {
        // Extract patient ID from barcode (e.g., TCGA-BA-4074-01A...)
        const parts = sample.split('-')
        if (parts.length >= 3) {
          patientId = `${parts[0]}-${parts[1]}-${parts[2]}`
        }
      }
      
      if (gene) {
        mutatedGenes.add(gene)
        if (DRIVER_GENES.includes(gene)) {
          if (!driverMutations.includes(gene)) {
            driverMutations.push(gene)
          }
        }
      }
      
      if (variant) {
        variantTypes[variant] = (variantTypes[variant] || 0) + 1
      }
    })

    return {
      patient_id: patientId,
      total_mutations: rows.length,
      mutated_genes: Array.from(mutatedGenes),
      driver_mutations: driverMutations,
      variant_types: variantTypes,
    }
  }

  // Handle file upload - generic version
  const handleFileUpload = async <T,>(
    file: File,
    setter: React.Dispatch<React.SetStateAction<FileUploadState<T>>>,
    parser: (content: string) => T
  ) => {
    setter({ file, status: 'uploading', error: null, data: null })
    
    try {
      const content = await file.text()
      const data = parser(content)
      setter({ file, status: 'parsed', error: null, data })
    } catch (err) {
      setter({ 
        file, 
        status: 'error', 
        error: err instanceof Error ? err.message : 'Failed to parse file',
        data: null 
      })
    }
  }

  // Dropzone for clinical data
  const onDropClinical = useCallback((files: File[]) => {
    if (files.length > 0) {
      handleFileUpload<ParsedClinicalData>(files[0], setClinicalFile, parseClinicalData)
    }
  }, [])

  const onDropMutation = useCallback((files: File[]) => {
    if (files.length > 0) {
      handleFileUpload<ParsedMutationData>(files[0], setMutationFile, parseMutationData)
    }
  }, [])

  const onDropRNA = useCallback((files: File[]) => {
    if (files.length > 0) {
      handleFileUpload<OmicsFileData>(files[0], setRnaFile, (content) => {
        const { headers, rows } = parseTSV(content)
        return { genes: rows.length, samples: headers.length - 1 }
      })
    }
  }, [])

  const onDropMethylation = useCallback((files: File[]) => {
    if (files.length > 0) {
      handleFileUpload<OmicsFileData>(files[0], setMethylationFile, (content) => {
        const { headers, rows } = parseTSV(content)
        return { genes: rows.length, samples: headers.length - 1 }
      })
    }
  }, [])

  const onDropWSI = useCallback(async (files: File[]) => {
    if (files.length > 0) {
      const file = files[0]
      setWsiFile({ file, status: 'uploading', error: null, data: null })
      
      try {
        // Upload to server
        const formData = new FormData()
        formData.append('file', file)
        
        const response = await fetch('/api/upload/wsi', {
          method: 'POST',
          body: formData,
        })
        
        if (!response.ok) throw new Error('Upload failed')
        
        const result = await response.json()
        setWsiFile({ file, status: 'parsed', error: null, data: result })
        onSlideSelected(result.slide_id)
      } catch (err) {
        setWsiFile({
          file,
          status: 'error',
          error: err instanceof Error ? err.message : 'Upload failed',
          data: null
        })
      }
    }
  }, [onSlideSelected])

  const clinicalDropzone = useDropzone({
    onDrop: onDropClinical,
    accept: { 'text/plain': ['.txt', '.tsv', '.csv'] },
    maxFiles: 1,
  })

  const mutationDropzone = useDropzone({
    onDrop: onDropMutation,
    accept: { 'text/plain': ['.txt', '.tsv', '.maf'] },
    maxFiles: 1,
  })

  const rnaDropzone = useDropzone({
    onDrop: onDropRNA,
    accept: { 'text/plain': ['.txt', '.tsv', '.csv'] },
    maxFiles: 1,
  })

  const methylationDropzone = useDropzone({
    onDrop: onDropMethylation,
    accept: { 'text/plain': ['.txt', '.tsv', '.csv'] },
    maxFiles: 1,
  })

  const wsiDropzone = useDropzone({
    onDrop: onDropWSI,
    accept: { 'image/tiff': ['.svs', '.tif', '.tiff'] },
    maxFiles: 1,
  })

  // Submit prediction
  const handleSubmit = async () => {
    setLoading(true)
    setError(null)

    try {
      const clinicalData = clinicalFile.data as ParsedClinicalData | null
      const mutationData = mutationFile.data as ParsedMutationData | null
      const wsiData = wsiFile.data as { slide_id: string } | null

      const request = {
        patient_id: clinicalData?.patient_id || mutationData?.patient_id || `patient_${Date.now()}`,
        clinical: clinicalData ? {
          age: clinicalData.age,
          gender: clinicalData.gender,
          tumor_stage: clinicalData.tumor_stage?.toLowerCase(),
          tumor_site: clinicalData.tumor_site,
        } : undefined,
        omics: mutationData ? {
          mutated_genes: mutationData.mutated_genes.slice(0, 100), // Limit for API
          driver_mutations: mutationData.driver_mutations,
        } : undefined,
        wsi: wsiData ? { slide_id: wsiData.slide_id } : undefined,
        time_points: [365, 730, 1095, 1825, 3650],
        return_attention: true,
      }

      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      })

      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.statusText}`)
      }

      const result = await response.json()
      onPredictionComplete(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  // Check if we have any data to predict
  const canPredict = clinicalFile.status === 'parsed' || 
                     mutationFile.status === 'parsed' ||
                     rnaFile.status === 'parsed' ||
                     methylationFile.status === 'parsed' ||
                     wsiFile.status === 'parsed'

  // Render file upload box
  const renderUploadBox = (
    dropzone: ReturnType<typeof useDropzone>,
    state: FileUploadState,
    icon: React.ReactNode,
    title: string,
    description: string,
    fileTypes: string,
    required?: boolean
  ) => (
    <Paper
      {...dropzone.getRootProps()}
      sx={{
        p: 3,
        border: '2px dashed',
        borderColor: state.status === 'parsed' ? 'success.main' :
                     state.status === 'error' ? 'error.main' :
                     dropzone.isDragActive ? 'primary.main' : 'grey.400',
        bgcolor: state.status === 'parsed' ? 'success.50' :
                 dropzone.isDragActive ? 'action.hover' : 'background.paper',
        cursor: 'pointer',
        textAlign: 'center',
        transition: 'all 0.3s ease',
        '&:hover': {
          borderColor: 'primary.main',
          bgcolor: 'action.hover',
        },
      }}
    >
      <input {...dropzone.getInputProps()} />
      
      {state.status === 'uploading' ? (
        <CircularProgress size={40} />
      ) : state.status === 'parsed' ? (
        <>
          <CheckCircleIcon sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
          <Typography variant="subtitle1" color="success.main" fontWeight={600}>
            ✓ {state.file?.name}
          </Typography>
        </>
      ) : (
        <>
          <Box sx={{ mb: 1 }}>{icon}</Box>
          <Typography variant="subtitle1" fontWeight={500}>
            {title} {required && <Chip label="Required" size="small" color="primary" sx={{ ml: 1 }} />}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {description}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {fileTypes}
          </Typography>
        </>
      )}
      
      {state.status === 'error' && (
        <Alert severity="error" sx={{ mt: 1 }}>{state.error}</Alert>
      )}
    </Paper>
  )

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h5" gutterBottom fontWeight={600}>
          Upload Patient Data for Survival Prediction
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Upload your patient's data files in TSV format (same format as TCGA training data).
          The model will analyze the data and predict survival outcomes.
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* File Upload Grid */}
      <Grid container spacing={3}>
        {/* Clinical Data */}
        <Grid item xs={12} md={6}>
          <Box>
            {renderUploadBox(
              clinicalDropzone,
              clinicalFile,
              <LocalHospitalIcon sx={{ fontSize: 40, color: 'primary.main' }} />,
              'Clinical Data',
              'Patient demographics, staging, treatment info',
              '.txt, .tsv (GDC format)',
              true
            )}
            
            {clinicalFile.status === 'parsed' && clinicalFile.data && (() => {
              const data = clinicalFile.data
              return (
                <Card sx={{ mt: 2 }}>
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom>Extracted Clinical Features:</Typography>
                    <Table size="small">
                      <TableBody>
                        {Object.entries(data)
                          .filter(([_, v]) => v !== null && !Array.isArray(v))
                          .slice(0, 6)
                          .map(([key, value]) => (
                            <TableRow key={key}>
                              <TableCell sx={{ py: 0.5 }}>{key.replace(/_/g, ' ')}</TableCell>
                              <TableCell sx={{ py: 0.5 }}><strong>{String(value)}</strong></TableCell>
                            </TableRow>
                          ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              )
            })()}
          </Box>
        </Grid>

        {/* Mutation Data */}
        <Grid item xs={12} md={6}>
          <Box>
            {renderUploadBox(
              mutationDropzone,
              mutationFile,
              <BiotechIcon sx={{ fontSize: 40, color: 'secondary.main' }} />,
              'Mutation Data',
              'Somatic mutations in MAF format',
              '.txt, .tsv, .maf'
            )}
            
            {mutationFile.status === 'parsed' && mutationFile.data && (() => {
              const data = mutationFile.data
              return (
                <Card sx={{ mt: 2 }}>
                  <CardContent>
                    <Typography variant="subtitle2" gutterBottom>Mutation Summary:</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1 }}>
                      <Chip 
                        label={`${data.total_mutations} total mutations`} 
                        size="small" 
                      />
                      <Chip 
                        label={`${data.mutated_genes.length} genes`} 
                        size="small" 
                      />
                    </Box>
                    {data.driver_mutations.length > 0 && (
                      <>
                        <Typography variant="caption" color="error.main" fontWeight={600}>
                          Driver Mutations:
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                          {data.driver_mutations.map(gene => (
                            <Chip key={gene} label={gene} size="small" color="error" variant="outlined" />
                          ))}
                        </Box>
                      </>
                    )}
                  </CardContent>
                </Card>
              )
            })()}
          </Box>
        </Grid>

        {/* RNA Expression */}
        <Grid item xs={12} md={4}>
          <Box>
            {renderUploadBox(
              rnaDropzone,
              rnaFile,
              <ScienceIcon sx={{ fontSize: 40, color: 'info.main' }} />,
              'RNA Expression',
              'Gene expression matrix',
              '.txt, .tsv'
            )}
            
            {rnaFile.status === 'parsed' && rnaFile.data && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                {rnaFile.data.genes} genes loaded
              </Typography>
            )}
          </Box>
        </Grid>

        {/* Methylation */}
        <Grid item xs={12} md={4}>
          <Box>
            {renderUploadBox(
              methylationDropzone,
              methylationFile,
              <DescriptionIcon sx={{ fontSize: 40, color: 'warning.main' }} />,
              'Methylation Data',
              'DNA methylation beta values',
              '.txt, .tsv'
            )}
            
            {methylationFile.status === 'parsed' && methylationFile.data && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                {methylationFile.data.genes} genes loaded
              </Typography>
            )}
          </Box>
        </Grid>

        {/* WSI Image */}
        <Grid item xs={12} md={4}>
          <Box>
            {renderUploadBox(
              wsiDropzone,
              wsiFile,
              <CloudUploadIcon sx={{ fontSize: 40, color: 'grey.500' }} />,
              'Pathology Image',
              'Whole slide image (optional)',
              '.svs, .tif, .tiff'
            )}
          </Box>
        </Grid>
      </Grid>

      <Divider sx={{ my: 4 }} />

      {/* How it works */}
      <Accordion sx={{ mb: 3 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography fontWeight={500}>How the prediction works</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography variant="body2" paragraph>
            The MOSAIC model analyzes your uploaded data to predict patient survival:
          </Typography>
          <ol style={{ margin: 0, paddingLeft: 20 }}>
            <li><strong>Clinical features</strong>: Age, gender, tumor stage, and treatment are encoded</li>
            <li><strong>Mutations</strong>: Driver gene mutations (TP53, CDKN2A, etc.) are identified</li>
            <li><strong>RNA expression</strong>: Gene expression patterns are analyzed via Transformer</li>
            <li><strong>Methylation</strong>: Epigenetic markers are processed</li>
            <li><strong>Pathology</strong>: Tumor morphology features are extracted from WSI</li>
            <li><strong>Fusion</strong>: All modalities are combined via cross-modal attention</li>
            <li><strong>Prediction</strong>: A risk score and survival curve are generated</li>
          </ol>
        </AccordionDetails>
      </Accordion>

      {/* Submit Button */}
      <Box sx={{ display: 'flex', justifyContent: 'center' }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleSubmit}
          disabled={!canPredict || loading}
          startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
          sx={{ px: 6, py: 1.5 }}
        >
          {loading ? 'Analyzing Data...' : 'Run Survival Prediction'}
        </Button>
      </Box>

      {loading && (
        <Box sx={{ mt: 2 }}>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 1 }}>
            Processing data through MOSAIC neural network...
          </Typography>
        </Box>
      )}
    </Box>
  )
}
