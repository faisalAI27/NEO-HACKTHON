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
} from '@mui/material'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import SendIcon from '@mui/icons-material/Send'
import { predict } from '../api'
import { ClinicalData, PredictionRequest, PredictionResult } from '../types'

interface PatientUploadFormProps {
  onPredictionComplete: (result: PredictionResult) => void
  onSlideSelected: (slideId: string | null) => void
}

const STEPS = ['Clinical Data', 'Pathology Image', 'Review & Submit']

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

  // File upload state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [slideId, setSlideId] = useState<string>('')

  // Request attention maps
  const [returnAttention, setReturnAttention] = useState(true)

  // Handle file drop
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0]
        setUploadedFile(file)
        // Extract slide ID from filename
        const slideIdFromFile = file.name.replace(/\.(svs|tif|tiff|ndpi)$/i, '')
        setSlideId(slideIdFromFile)
        onSlideSelected(slideIdFromFile)
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

  // Submit prediction request
  const handleSubmit = async () => {
    setLoading(true)
    setError(null)

    try {
      const request: PredictionRequest = {
        patient_id: `patient_${Date.now()}`,
        clinical: clinicalData,
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

  // Check if step is complete
  const isStepComplete = (step: number) => {
    switch (step) {
      case 0:
        return clinicalData.age !== undefined || clinicalData.tumor_stage !== undefined
      case 1:
        return true // WSI is optional
      case 2:
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
        {/* Step 0: Clinical Data */}
        {activeStep === 0 && (
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

        {/* Step 1: WSI Upload */}
        {activeStep === 1 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Pathology Image (Optional)
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Upload a whole slide image (WSI) for histopathology analysis. Supported
                formats: SVS, TIFF, NDPI.
              </Typography>
            </Grid>

            <Grid item xs={12}>
              <Paper
                {...getRootProps()}
                sx={{
                  p: 4,
                  border: '2px dashed',
                  borderColor: isDragActive ? 'primary.main' : 'grey.400',
                  backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
                  cursor: 'pointer',
                  textAlign: 'center',
                  minHeight: 200,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  '&:hover': {
                    borderColor: 'primary.main',
                    backgroundColor: 'action.hover',
                  },
                }}
              >
                <input {...getInputProps()} />
                <CloudUploadIcon sx={{ fontSize: 48, color: 'grey.500', mb: 2 }} />

                {uploadedFile ? (
                  <>
                    <Typography variant="h6" color="primary">
                      {uploadedFile.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </Typography>
                    <Chip
                      label="Click or drag to replace"
                      size="small"
                      sx={{ mt: 1 }}
                    />
                  </>
                ) : (
                  <>
                    <Typography variant="h6" color="text.secondary">
                      {isDragActive
                        ? 'Drop the file here...'
                        : 'Drag & drop a WSI file here'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      or click to select a file
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                      Supported: .svs, .tif, .tiff, .ndpi (up to 5GB)
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
                }}
                helperText="Enter the ID of a slide already on the server"
              />
            </Grid>
          </Grid>
        )}

        {/* Step 2: Review & Submit */}
        {activeStep === 2 && (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Review & Submit
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Review the information below and submit for prediction.
              </Typography>
            </Grid>

            {/* Summary */}
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
                    No WSI provided (clinical-only prediction)
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
          disabled={activeStep === 0 || loading}
          onClick={handleBack}
        >
          Back
        </Button>

        <Box>
          {activeStep === STEPS.length - 1 ? (
            <Button
              variant="contained"
              onClick={handleSubmit}
              disabled={loading}
              endIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
            >
              {loading ? 'Predicting...' : 'Submit Prediction'}
            </Button>
          ) : (
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
