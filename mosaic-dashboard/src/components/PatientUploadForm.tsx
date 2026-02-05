import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  Box,
  Button,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Alert,
  CircularProgress,
  FormControlLabel,
  Switch,
  Chip,
  LinearProgress,
  Card,
  CardContent,
} from '@mui/material'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import CancelIcon from '@mui/icons-material/Cancel'
import BiotechIcon from '@mui/icons-material/Biotech'
import SendIcon from '@mui/icons-material/Send'
import { predict, uploadWSIFile, detectCancer, CancerDetectionResult } from '../api'
import { ClinicalData, OmicsData, PredictionRequest, PredictionResult } from '../types'

interface PatientUploadFormProps {
  onPredictionComplete: (result: PredictionResult) => void
  onSlideSelected: (slideId: string | null) => void
}

// New workflow steps - WSI first, then cancer detection, then clinical/genomic data
const STEPS = ['Upload Pathology Image', 'Cancer Detection', 'Clinical Data', 'Genomic Data', 'Review & Submit']

// Common cancer driver genes for mutation selection
const COMMON_DRIVER_GENES = [
  'TP53', 'CDKN2A', 'PIK3CA', 'NOTCH1', 'FAT1', 'CASP8', 'NSD1', 'HRAS',
  'EGFR', 'PTEN', 'NFE2L2', 'KEAP1', 'RB1', 'FBXW7', 'EP300', 'CREBBP',
  'KMT2D', 'HLA-A', 'HLA-B', 'TGFBR2', 'AJUBA', 'RAC1', 'CUL3', 'EPHA2'
]

const TUMOR_STAGES = [
  'stage i',
  'stage ii',
  'stage iii',
  'stage iva',
  'stage ivb',
  'stage ivc',
]

const TUMOR_SITES = [
  'oral cavity',
  'oropharynx',
  'hypopharynx',
  'larynx',
  'nasal cavity',
  'other',
]

export default function PatientUploadForm({
  onPredictionComplete,
  onSlideSelected,
}: PatientUploadFormProps) {
  const [activeStep, setActiveStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Clinical data state
  const [clinicalData, setClinicalData] = useState<ClinicalData>({
    age: undefined,
    gender: undefined,
    tumor_stage: undefined,
    hpv_status: undefined,
    smoking_history: undefined,
    alcohol_history: undefined,
    tumor_site: undefined,
  })

  // Omics/Genomic data state
  const [omicsData, setOmicsData] = useState<OmicsData>({
    mutated_genes: [],
    driver_mutations: [],
  })
  const [rnaFile, setRnaFile] = useState<File | null>(null)
  const [methylationFile, setMethylationFile] = useState<File | null>(null)
  const [selectedMutations, setSelectedMutations] = useState<string[]>([])

  // File upload state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [slideId, setSlideId] = useState<string>('')
  const [uploadProgress, setUploadProgress] = useState<number>(0)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle')
  const [uploadError, setUploadError] = useState<string | null>(null)

  // Cancer detection state (Stage 1)
  const [cancerDetectionResult, setCancerDetectionResult] = useState<CancerDetectionResult | null>(null)
  const [detectingCancer, setDetectingCancer] = useState(false)
  const [detectionError, setDetectionError] = useState<string | null>(null)

  // Request attention maps
  const [returnAttention, setReturnAttention] = useState(true)

  // Format file size for display
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // Handle file drop - start actual upload
  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0]
        setUploadedFile(file)
        setUploadProgress(0)
        setUploadStatus('uploading')
        setUploadError(null)

        try {
          const result = await uploadWSIFile(file, (progress) => {
            setUploadProgress(progress)
          })
          
          setSlideId(result.slide_id)
          onSlideSelected(result.slide_id)
          setUploadStatus('success')
        } catch (err) {
          console.error('Upload error:', err)
          setUploadStatus('error')
          setUploadError(err instanceof Error ? err.message : 'Upload failed')
        }
      }
    },
    [onSlideSelected]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/tiff': ['.svs', '.tif', '.tiff'],
      'application/octet-stream': ['.svs', '.ndpi'],
    },
    maxFiles: 1,
    maxSize: 5 * 1024 * 1024 * 1024, // 5GB max
  })

  // Handle clinical data changes
  const handleClinicalChange = (field: keyof ClinicalData, value: unknown) => {
    setClinicalData((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  // Handle step navigation
  const handleNext = () => {
    setActiveStep((prev) => prev + 1)
  }

  const handleBack = () => {
    setActiveStep((prev) => prev - 1)
  }

  // Run cancer detection (Stage 1)
  const handleCancerDetection = async () => {
    if (!slideId) return
    
    setDetectingCancer(true)
    setDetectionError(null)
    
    try {
      const result = await detectCancer(slideId)
      setCancerDetectionResult(result)
      
      // If cancerous, automatically advance to next step
      if (result.is_cancerous) {
        setActiveStep(2) // Clinical Data step
      }
    } catch (err) {
      console.error('Cancer detection error:', err)
      setDetectionError(err instanceof Error ? err.message : 'Detection failed')
    } finally {
      setDetectingCancer(false)
    }
  }

  // Submit prediction request
  const handleSubmit = async () => {
    setLoading(true)
    setError(null)

    try {
      // Build omics data with selected mutations
      const omicsPayload: OmicsData = {
        ...omicsData,
        mutated_genes: selectedMutations,
        driver_mutations: selectedMutations.filter(g => 
          ['TP53', 'PIK3CA', 'CDKN2A', 'NOTCH1', 'HRAS', 'EGFR'].includes(g)
        ),
      }

      const request: PredictionRequest = {
        patient_id: `patient_${Date.now()}`,
        clinical: clinicalData,
        omics: (selectedMutations.length > 0 || rnaFile || methylationFile) ? omicsPayload : undefined,
        wsi: slideId ? { slide_id: slideId } : undefined,
        time_points: [365, 730, 1095, 1825, 3650],
        return_attention: returnAttention,
      }

      const result = await predict(request)
      onPredictionComplete(result)
    } catch (err: unknown) {
      console.error('Prediction error:', err)
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(`Prediction failed: ${errorMessage}`)
    } finally {
      setLoading(false)
    }
  }

  // Toggle mutation selection
  const handleMutationToggle = (gene: string) => {
    setSelectedMutations(prev => 
      prev.includes(gene) 
        ? prev.filter(g => g !== gene)
        : [...prev, gene]
    )
  }

  // Check if step is complete
  const isStepComplete = (step: number) => {
    switch (step) {
      case 0: // Upload WSI
        return uploadStatus === 'success' && slideId !== ''
      case 1: // Cancer Detection
        return cancerDetectionResult !== null
      case 2: // Clinical Data
        return clinicalData.age !== undefined || clinicalData.tumor_stage !== undefined
      case 3: // Genomic Data
        return true // Optional
      case 4: // Review
        return true
      default:
        return false
    }
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* Stepper */}
      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Step Content */}
      <Box sx={{ minHeight: 400 }}>
        {/* Step 0: Upload Pathology Image (WSI) */}
        {activeStep === 0 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Upload Pathology Image
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Start by uploading a whole slide image (WSI). The system will first analyze 
                this image to detect if the tissue shows signs of cancer.
              </Typography>
            </Grid>

            {uploadError && (
              <Grid item xs={12}>
                <Alert severity="error" onClose={() => setUploadError(null)}>
                  {uploadError}
                </Alert>
              </Grid>
            )}

            <Grid item xs={12}>
              <Paper
                {...getRootProps()}
                sx={{
                  p: 4,
                  border: '2px dashed',
                  borderColor: uploadStatus === 'success' ? 'success.main' : 
                               uploadStatus === 'error' ? 'error.main' :
                               uploadStatus === 'uploading' ? 'primary.main' :
                               isDragActive ? 'primary.main' : 'grey.400',
                  backgroundColor: uploadStatus === 'success' ? 'success.50' : 
                                   uploadStatus === 'uploading' ? 'primary.50' :
                                   isDragActive ? 'action.hover' : 'background.paper',
                  cursor: uploadStatus === 'uploading' ? 'default' : 'pointer',
                  textAlign: 'center',
                  minHeight: 250,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.3s ease',
                  pointerEvents: uploadStatus === 'uploading' ? 'none' : 'auto',
                }}
              >
                <input {...getInputProps()} />
                
                {uploadStatus === 'uploading' ? (
                  <>
                    <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                    <Typography variant="h6" color="primary">
                      Uploading {uploadedFile?.name}...
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {formatFileSize(uploadedFile?.size || 0)}
                    </Typography>
                    <Box sx={{ width: '80%', maxWidth: 400 }}>
                      <LinearProgress 
                        variant="determinate" 
                        value={uploadProgress} 
                        sx={{ height: 10, borderRadius: 5 }}
                      />
                      <Typography variant="body2" color="primary" sx={{ mt: 1 }}>
                        {uploadProgress}% complete
                      </Typography>
                    </Box>
                  </>
                ) : uploadStatus === 'success' ? (
                  <>
                    <Box 
                      sx={{ 
                        width: 64, 
                        height: 64, 
                        borderRadius: '50%', 
                        bgcolor: 'success.main', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        mb: 2
                      }}
                    >
                      <CheckCircleIcon sx={{ fontSize: 40, color: 'white' }} />
                    </Box>
                    <Typography variant="h6" color="success.main" fontWeight={600}>
                      ✓ Upload Complete
                    </Typography>
                    <Typography variant="subtitle1" color="text.primary" sx={{ mt: 1 }}>
                      {uploadedFile?.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Size: {formatFileSize(uploadedFile?.size || 0)}
                    </Typography>
                    <Typography variant="body2" color="primary" sx={{ mt: 0.5 }}>
                      Slide ID: <strong>{slideId}</strong>
                    </Typography>
                  </>
                ) : (
                  <>
                    <CloudUploadIcon sx={{ fontSize: 64, color: 'grey.400', mb: 2 }} />
                    <Typography variant="h5" color="text.secondary">
                      {isDragActive ? 'Drop the file here...' : 'Drag & drop a WSI file here'}
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
                      or click to select a file
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 2 }}>
                      Supported formats: .svs, .tif, .tiff, .ndpi (up to 5GB)
                    </Typography>
                  </>
                )}
              </Paper>
            </Grid>

            {/* Or use existing slide */}
            <Grid item xs={12}>
              <Typography variant="body2" color="text.secondary" align="center" sx={{ my: 2 }}>
                — OR —
              </Typography>
              <TextField
                fullWidth
                label="Use Existing Slide ID"
                value={slideId}
                onChange={(e) => {
                  setSlideId(e.target.value)
                  onSlideSelected(e.target.value || null)
                  if (e.target.value) {
                    setUploadStatus('success')
                  }
                }}
                helperText="Enter the ID of a slide already on the server"
              />
            </Grid>
          </Grid>
        )}

        {/* Step 1: Cancer Detection */}
        {activeStep === 1 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Cancer Detection Analysis
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                The system will analyze the pathology image to determine if the tissue shows 
                signs of cancer. This is the first step before survival prediction.
              </Typography>
            </Grid>

            {detectionError && (
              <Grid item xs={12}>
                <Alert severity="error" onClose={() => setDetectionError(null)}>
                  {detectionError}
                </Alert>
              </Grid>
            )}

            <Grid item xs={12}>
              <Card sx={{ maxWidth: 600, mx: 'auto' }}>
                <CardContent sx={{ textAlign: 'center', py: 4 }}>
                  {!cancerDetectionResult && !detectingCancer && (
                    <>
                      <BiotechIcon sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
                      <Typography variant="h5" gutterBottom>
                        Ready to Analyze
                      </Typography>
                      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                        Slide ID: <strong>{slideId}</strong>
                      </Typography>
                      <Button
                        variant="contained"
                        size="large"
                        onClick={handleCancerDetection}
                        startIcon={<BiotechIcon />}
                      >
                        Run Cancer Detection
                      </Button>
                    </>
                  )}

                  {detectingCancer && (
                    <>
                      <CircularProgress size={80} sx={{ mb: 3 }} />
                      <Typography variant="h5" gutterBottom>
                        Analyzing Pathology Image...
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        This may take a few moments
                      </Typography>
                    </>
                  )}

                  {cancerDetectionResult && (
                    <>
                      {cancerDetectionResult.is_cancerous ? (
                        <>
                          <Box 
                            sx={{ 
                              width: 80, 
                              height: 80, 
                              borderRadius: '50%', 
                              bgcolor: 'error.main', 
                              display: 'flex', 
                              alignItems: 'center', 
                              justifyContent: 'center',
                              mx: 'auto',
                              mb: 2
                            }}
                          >
                            <CancelIcon sx={{ fontSize: 50, color: 'white' }} />
                          </Box>
                          <Typography variant="h4" color="error.main" fontWeight={700} gutterBottom>
                            Cancer Detected
                          </Typography>
                          <Typography variant="h6" color="text.primary" sx={{ mb: 2 }}>
                            {cancerDetectionResult.cancer_type}
                          </Typography>
                          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap' }}>
                            <Chip 
                              label={`Confidence: ${(cancerDetectionResult.confidence * 100).toFixed(1)}%`} 
                              color="error" 
                            />
                            {cancerDetectionResult.tumor_regions && (
                              <Chip 
                                label={`${cancerDetectionResult.tumor_regions} tumor regions identified`} 
                                color="warning" 
                              />
                            )}
                          </Box>
                          <Alert severity="info" sx={{ mt: 3, textAlign: 'left' }}>
                            <strong>Next Step:</strong> Please provide clinical and genomic data 
                            for a comprehensive survival prediction.
                          </Alert>
                        </>
                      ) : (
                        <>
                          <Box 
                            sx={{ 
                              width: 80, 
                              height: 80, 
                              borderRadius: '50%', 
                              bgcolor: 'success.main', 
                              display: 'flex', 
                              alignItems: 'center', 
                              justifyContent: 'center',
                              mx: 'auto',
                              mb: 2
                            }}
                          >
                            <CheckCircleIcon sx={{ fontSize: 50, color: 'white' }} />
                          </Box>
                          <Typography variant="h4" color="success.main" fontWeight={700} gutterBottom>
                            No Cancer Detected
                          </Typography>
                          <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                            The pathology image does not show signs of malignancy.
                          </Typography>
                          <Chip 
                            label={`Confidence: ${(cancerDetectionResult.confidence * 100).toFixed(1)}%`} 
                            color="success" 
                          />
                          <Alert severity="success" sx={{ mt: 3, textAlign: 'left' }}>
                            No further survival analysis is needed. The tissue appears benign.
                          </Alert>
                        </>
                      )}
                      <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 2 }}>
                        Analysis completed in {cancerDetectionResult.analysis_time_seconds}s
                      </Typography>
                    </>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}

        {/* Step 2: Clinical Data */}
        {activeStep === 2 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Patient Clinical Information
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Enter the patient's clinical data. All fields are optional but more
                information improves prediction accuracy.
              </Typography>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Age at Diagnosis"
                type="number"
                value={clinicalData.age || ''}
                onChange={(e) =>
                  handleClinicalChange('age', e.target.value ? Number(e.target.value) : undefined)
                }
                inputProps={{ min: 0, max: 120 }}
                helperText="Patient age in years"
              />
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth>
                <InputLabel>Gender</InputLabel>
                <Select
                  value={clinicalData.gender || ''}
                  label="Gender"
                  onChange={(e) => handleClinicalChange('gender', e.target.value || undefined)}
                >
                  <MenuItem value="">
                    <em>Not specified</em>
                  </MenuItem>
                  <MenuItem value="male">Male</MenuItem>
                  <MenuItem value="female">Female</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth>
                <InputLabel>Tumor Stage</InputLabel>
                <Select
                  value={clinicalData.tumor_stage || ''}
                  label="Tumor Stage"
                  onChange={(e) => handleClinicalChange('tumor_stage', e.target.value || undefined)}
                >
                  <MenuItem value="">
                    <em>Not specified</em>
                  </MenuItem>
                  {TUMOR_STAGES.map((stage) => (
                    <MenuItem key={stage} value={stage}>
                      {stage.toUpperCase()}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <FormControl fullWidth>
                <InputLabel>Tumor Site</InputLabel>
                <Select
                  value={clinicalData.tumor_site || ''}
                  label="Tumor Site"
                  onChange={(e) => handleClinicalChange('tumor_site', e.target.value || undefined)}
                >
                  <MenuItem value="">
                    <em>Not specified</em>
                  </MenuItem>
                  {TUMOR_SITES.map((site) => (
                    <MenuItem key={site} value={site}>
                      {site.charAt(0).toUpperCase() + site.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={clinicalData.hpv_status || false}
                    onChange={(e) => handleClinicalChange('hpv_status', e.target.checked)}
                  />
                }
                label="HPV Positive"
              />
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <TextField
                fullWidth
                label="Smoking History (Pack-Years)"
                type="number"
                value={clinicalData.smoking_history || ''}
                onChange={(e) =>
                  handleClinicalChange(
                    'smoking_history',
                    e.target.value ? Number(e.target.value) : undefined
                  )
                }
                inputProps={{ min: 0 }}
                helperText="Total pack-years"
              />
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={clinicalData.alcohol_history || false}
                    onChange={(e) => handleClinicalChange('alcohol_history', e.target.checked)}
                  />
                }
                label="Alcohol History"
              />
            </Grid>
          </Grid>
        )}

        {/* Step 3: Genomic Data */}
        {activeStep === 3 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Genomic Data (Optional)
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Select any known gene mutations or upload genomic data files. This helps improve prediction accuracy.
              </Typography>
            </Grid>

            {/* Mutation Selection */}
            <Grid item xs={12}>
              <Typography variant="subtitle1" gutterBottom fontWeight={500}>
                Known Gene Mutations
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Click on genes that have been identified as mutated in this patient's tumor:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {COMMON_DRIVER_GENES.map((gene) => (
                  <Chip
                    key={gene}
                    label={gene}
                    onClick={() => handleMutationToggle(gene)}
                    color={selectedMutations.includes(gene) ? 'primary' : 'default'}
                    variant={selectedMutations.includes(gene) ? 'filled' : 'outlined'}
                    sx={{ 
                      cursor: 'pointer',
                      '&:hover': { opacity: 0.8 }
                    }}
                  />
                ))}
              </Box>
              {selectedMutations.length > 0 && (
                <Typography variant="body2" color="primary" sx={{ mt: 2 }}>
                  Selected: {selectedMutations.join(', ')}
                </Typography>
              )}
            </Grid>

            {/* RNA File Upload */}
            <Grid item xs={12} md={6}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  RNA Expression Data
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Upload a CSV/TSV file with gene expression values
                </Typography>
                <Button
                  variant="outlined"
                  component="label"
                  size="small"
                  startIcon={<CloudUploadIcon />}
                >
                  {rnaFile ? rnaFile.name : 'Upload RNA Data'}
                  <input
                    type="file"
                    hidden
                    accept=".csv,.tsv,.txt"
                    onChange={(e) => setRnaFile(e.target.files?.[0] || null)}
                  />
                </Button>
                {rnaFile && (
                  <Chip 
                    label={rnaFile.name} 
                    size="small" 
                    onDelete={() => setRnaFile(null)}
                    sx={{ ml: 1 }}
                  />
                )}
              </Paper>
            </Grid>

            {/* Methylation File Upload */}
            <Grid item xs={12} md={6}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Methylation Data
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Upload a CSV/TSV file with methylation beta values
                </Typography>
                <Button
                  variant="outlined"
                  component="label"
                  size="small"
                  startIcon={<CloudUploadIcon />}
                >
                  {methylationFile ? methylationFile.name : 'Upload Methylation Data'}
                  <input
                    type="file"
                    hidden
                    accept=".csv,.tsv,.txt"
                    onChange={(e) => setMethylationFile(e.target.files?.[0] || null)}
                  />
                </Button>
                {methylationFile && (
                  <Chip 
                    label={methylationFile.name} 
                    size="small" 
                    onDelete={() => setMethylationFile(null)}
                    sx={{ ml: 1 }}
                  />
                )}
              </Paper>
            </Grid>

            <Grid item xs={12}>
              <Alert severity="info" sx={{ mt: 1 }}>
                <strong>Tip:</strong> Genomic data is optional. The model can make predictions using only clinical data, 
                but adding mutation information typically improves accuracy by 5-15%.
              </Alert>
            </Grid>
          </Grid>
        )}

        {/* Step 4: Review & Submit */}
        {activeStep === 4 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Review & Submit for Survival Prediction
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Review the information below and submit for comprehensive survival prediction.
              </Typography>
            </Grid>

            {/* Cancer Detection Result */}
            <Grid item xs={12}>
              <Paper variant="outlined" sx={{ p: 2, bgcolor: 'error.50', borderColor: 'error.main' }}>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom color="error.main">
                  Cancer Detection Result
                </Typography>
                {cancerDetectionResult && (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center' }}>
                    <Chip 
                      label="Cancer Detected" 
                      color="error" 
                      icon={<CancelIcon />}
                    />
                    <Chip 
                      label={`Confidence: ${(cancerDetectionResult.confidence * 100).toFixed(1)}%`} 
                      size="small"
                      variant="outlined"
                    />
                    {cancerDetectionResult.cancer_type && (
                      <Chip 
                        label={cancerDetectionResult.cancer_type} 
                        size="small"
                        color="warning"
                      />
                    )}
                    {cancerDetectionResult.tumor_regions && (
                      <Chip 
                        label={`${cancerDetectionResult.tumor_regions} tumor regions`} 
                        size="small"
                        variant="outlined"
                      />
                    )}
                  </Box>
                )}
              </Paper>
            </Grid>

            {/* Pathology Image */}
            <Grid item xs={12} md={6}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  Pathology Image
                </Typography>
                {slideId ? (
                  <Chip label={`Slide: ${slideId}`} color="primary" />
                ) : uploadedFile ? (
                  <Chip label={uploadedFile.name} color="primary" />
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No WSI provided
                  </Typography>
                )}
              </Paper>
            </Grid>

            {/* Clinical Data Summary */}
            <Grid item xs={12} md={6}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  Clinical Data
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {clinicalData.age && (
                    <Chip label={`Age: ${clinicalData.age}`} size="small" />
                  )}
                  {clinicalData.gender && (
                    <Chip label={`Gender: ${clinicalData.gender}`} size="small" />
                  )}
                  {clinicalData.tumor_stage && (
                    <Chip label={`Stage: ${clinicalData.tumor_stage.toUpperCase()}`} size="small" />
                  )}
                  {clinicalData.hpv_status !== undefined && (
                    <Chip
                      label={`HPV: ${clinicalData.hpv_status ? 'Positive' : 'Negative'}`}
                      size="small"
                    />
                  )}
                  {clinicalData.smoking_history !== undefined && (
                    <Chip label={`Smoking: ${clinicalData.smoking_history} pack-years`} size="small" />
                  )}
                </Box>
                {Object.values(clinicalData).every((v) => v === undefined) && (
                  <Typography variant="body2" color="text.secondary">
                    No clinical data provided
                  </Typography>
                )}
              </Paper>
            </Grid>

            {/* Genomic Data Summary */}
            <Grid item xs={12}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  Genomic Data
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {selectedMutations.length > 0 && (
                    <>
                      <Typography variant="body2" color="text.secondary" sx={{ width: '100%', mb: 1 }}>
                        Selected Mutations:
                      </Typography>
                      {selectedMutations.map(gene => (
                        <Chip key={gene} label={gene} size="small" color="secondary" />
                      ))}
                    </>
                  )}
                  {rnaFile && (
                    <Chip label={`RNA: ${rnaFile.name}`} size="small" color="info" />
                  )}
                  {methylationFile && (
                    <Chip label={`Methylation: ${methylationFile.name}`} size="small" color="info" />
                  )}
                </Box>
                {selectedMutations.length === 0 && !rnaFile && !methylationFile && (
                  <Typography variant="body2" color="text.secondary">
                    No genomic data provided
                  </Typography>
                )}
              </Paper>
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={returnAttention}
                    onChange={(e) => setReturnAttention(e.target.checked)}
                  />
                }
                label="Return attention maps for interpretability"
              />
            </Grid>
          </Grid>
        )}
      </Box>

      {/* Loading Progress */}
      {loading && (
        <Box sx={{ mt: 2 }}>
          <LinearProgress />
          <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 1 }}>
            Running prediction... This may take a moment.
          </Typography>
        </Box>
      )}

      {/* Navigation Buttons */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
        <Button
          disabled={activeStep === 0 || loading || detectingCancer}
          onClick={handleBack}
        >
          Back
        </Button>

        <Box>
          {/* Step 1: Cancer Detection - special handling */}
          {activeStep === 1 && !cancerDetectionResult && (
            <Button
              variant="contained"
              onClick={handleCancerDetection}
              disabled={detectingCancer || !slideId}
              startIcon={detectingCancer ? <CircularProgress size={20} /> : <BiotechIcon />}
            >
              {detectingCancer ? 'Analyzing...' : 'Run Cancer Detection'}
            </Button>
          )}

          {/* Step 1: If cancer detected, show Next button */}
          {activeStep === 1 && cancerDetectionResult?.is_cancerous && (
            <Button
              variant="contained"
              onClick={handleNext}
              color="primary"
            >
              Continue to Clinical Data
            </Button>
          )}

          {/* Step 1: If no cancer detected, show different message */}
          {activeStep === 1 && cancerDetectionResult && !cancerDetectionResult.is_cancerous && (
            <Button
              variant="outlined"
              onClick={() => {
                // Reset and start over
                setCancerDetectionResult(null)
                setActiveStep(0)
              }}
            >
              Upload Different Image
            </Button>
          )}

          {/* Final step: Submit button */}
          {activeStep === STEPS.length - 1 && (
            <Button
              variant="contained"
              onClick={handleSubmit}
              disabled={loading}
              endIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
            >
              {loading ? 'Predicting...' : 'Submit for Survival Prediction'}
            </Button>
          )}

          {/* Other steps: Next button */}
          {activeStep !== 1 && activeStep !== STEPS.length - 1 && (
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={!isStepComplete(activeStep)}
            >
              Next
            </Button>
          )}
        </Box>
      </Box>
    </Box>
  )
}
