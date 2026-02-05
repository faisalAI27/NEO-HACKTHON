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
  LinearProgress,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  Divider,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  FormGroup,
  InputAdornment,
} from '@mui/material'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import SendIcon from '@mui/icons-material/Send'
import ImageIcon from '@mui/icons-material/Image'
import PersonIcon from '@mui/icons-material/Person'
import LocalHospitalIcon from '@mui/icons-material/LocalHospital'
import BiotechIcon from '@mui/icons-material/Biotech'
import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import { PredictionResult } from '../types'

interface PatientInputFormProps {
  onPredictionComplete: (result: PredictionResult) => void
  onSlideSelected: (slideId: string | null) => void
}

// Driver genes known to affect HNSC survival
const DRIVER_GENES = [
  { gene: 'TP53', description: 'Tumor suppressor (most common in HNSC)' },
  { gene: 'CDKN2A', description: 'Cell cycle regulation' },
  { gene: 'PIK3CA', description: 'PI3K pathway oncogene' },
  { gene: 'NOTCH1', description: 'Cell signaling receptor' },
  { gene: 'FAT1', description: 'Tumor suppressor' },
  { gene: 'CASP8', description: 'Apoptosis regulator' },
  { gene: 'NSD1', description: 'Histone methyltransferase' },
  { gene: 'HRAS', description: 'RAS oncogene' },
  { gene: 'EGFR', description: 'Growth factor receptor' },
  { gene: 'PTEN', description: 'PI3K pathway suppressor' },
  { gene: 'NFE2L2', description: 'Oxidative stress response' },
  { gene: 'KEAP1', description: 'NRF2 pathway regulator' },
]

// Tumor sites for HNSC
const TUMOR_SITES = [
  'Oral Cavity',
  'Oropharynx',
  'Larynx',
  'Hypopharynx',
  'Tongue',
  'Floor of Mouth',
  'Tonsil',
  'Base of Tongue',
  'Other',
]

// Tumor stages
const TUMOR_STAGES = [
  { value: 'stage i', label: 'Stage I' },
  { value: 'stage ii', label: 'Stage II' },
  { value: 'stage iii', label: 'Stage III' },
  { value: 'stage iva', label: 'Stage IVA' },
  { value: 'stage ivb', label: 'Stage IVB' },
  { value: 'stage ivc', label: 'Stage IVC' },
]

// Tumor grades
const TUMOR_GRADES = [
  { value: 'G1', label: 'G1 - Well Differentiated' },
  { value: 'G2', label: 'G2 - Moderately Differentiated' },
  { value: 'G3', label: 'G3 - Poorly Differentiated' },
  { value: 'G4', label: 'G4 - Undifferentiated' },
]

// Treatment types
const TREATMENT_TYPES = [
  'Surgery',
  'Radiation, External Beam',
  'Chemotherapy',
  'Immunotherapy',
  'Targeted Therapy',
  'Concurrent Chemoradiation',
]

const steps = [
  'Pathology Image',
  'Demographics',
  'Clinical Data',
  'Mutations',
]

export default function PatientInputForm({
  onPredictionComplete,
  onSlideSelected,
}: PatientInputFormProps) {
  const [activeStep, setActiveStep] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // WSI State
  const [wsiFile, setWsiFile] = useState<File | null>(null)
  const [wsiUploading, setWsiUploading] = useState(false)
  const [wsiSlideId, setWsiSlideId] = useState<string | null>(null)

  // Demographics State
  const [age, setAge] = useState<number | ''>('')
  const [gender, setGender] = useState<string>('')
  const [race, setRace] = useState<string>('')
  const [ethnicity, setEthnicity] = useState<string>('')

  // Clinical State
  const [tumorStage, setTumorStage] = useState<string>('')
  const [tumorGrade, setTumorGrade] = useState<string>('')
  const [tumorSite, setTumorSite] = useState<string>('')
  const [hpvStatus, setHpvStatus] = useState<string>('')
  const [smokingPackYears, setSmokingPackYears] = useState<number | ''>('')
  const [alcoholHistory, setAlcoholHistory] = useState<string>('')
  const [treatments, setTreatments] = useState<string[]>([])

  // Mutation State
  const [selectedMutations, setSelectedMutations] = useState<string[]>([])

  // Handle WSI upload
  const onDropWSI = useCallback(async (files: File[]) => {
    if (files.length === 0) return
    
    const file = files[0]
    setWsiFile(file)
    setWsiUploading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/upload/wsi', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const result = await response.json()
      setWsiSlideId(result.slide_id)
      onSlideSelected(result.slide_id)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload image')
      setWsiFile(null)
    } finally {
      setWsiUploading(false)
    }
  }, [onSlideSelected])

  const wsiDropzone = useDropzone({
    onDrop: onDropWSI,
    accept: {
      'image/tiff': ['.svs', '.tif', '.tiff', '.ndpi'],
    },
    maxFiles: 1,
  })

  // Handle mutation toggle
  const handleMutationToggle = (gene: string) => {
    setSelectedMutations(prev =>
      prev.includes(gene)
        ? prev.filter(g => g !== gene)
        : [...prev, gene]
    )
  }

  // Handle treatment toggle
  const handleTreatmentToggle = (treatment: string) => {
    setTreatments(prev =>
      prev.includes(treatment)
        ? prev.filter(t => t !== treatment)
        : [...prev, treatment]
    )
  }

  // Navigation
  const handleNext = () => setActiveStep(prev => Math.min(prev + 1, steps.length - 1))
  const handleBack = () => setActiveStep(prev => Math.max(prev - 1, 0))

  // Submit prediction
  const handleSubmit = async () => {
    setLoading(true)
    setError(null)

    try {
      const request = {
        patient_id: `patient_${Date.now()}`,
        clinical: {
          age: age || undefined,
          gender: gender || undefined,
          tumor_stage: tumorStage || undefined,
          tumor_site: tumorSite || undefined,
          hpv_status: hpvStatus === 'positive' ? true : hpvStatus === 'negative' ? false : undefined,
          smoking_history: smokingPackYears || undefined,
          alcohol_history: alcoholHistory === 'yes' ? true : alcoholHistory === 'no' ? false : undefined,
        },
        omics: {
          mutated_genes: selectedMutations,
          driver_mutations: selectedMutations.filter(g => 
            ['TP53', 'CDKN2A', 'PIK3CA', 'NOTCH1', 'FAT1'].includes(g)
          ),
        },
        wsi: wsiSlideId ? { slide_id: wsiSlideId } : undefined,
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

  // Check if we can submit
  const canSubmit = age !== '' || gender !== '' || tumorStage !== '' || 
                    selectedMutations.length > 0 || wsiSlideId !== null

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h5" gutterBottom fontWeight={600}>
          Patient Survival Prediction
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Enter patient data to predict survival outcomes using the MOSAIC model
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Stepper */}
      <Stepper activeStep={activeStep} orientation="vertical">
        {/* Step 1: Pathology Image */}
        <Step>
          <StepLabel
            StepIconComponent={() => (
              <Box sx={{
                width: 32, height: 32, borderRadius: '50%',
                bgcolor: activeStep >= 0 ? 'primary.main' : 'grey.300',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}>
                <ImageIcon sx={{ color: 'white', fontSize: 18 }} />
              </Box>
            )}
          >
            <Typography fontWeight={500}>Pathology Image (Optional)</Typography>
          </StepLabel>
          <StepContent>
            <Paper
              {...wsiDropzone.getRootProps()}
              sx={{
                p: 4,
                border: '2px dashed',
                borderColor: wsiSlideId ? 'success.main' : wsiDropzone.isDragActive ? 'primary.main' : 'grey.400',
                bgcolor: wsiSlideId ? 'success.50' : wsiDropzone.isDragActive ? 'action.hover' : 'background.paper',
                cursor: 'pointer',
                textAlign: 'center',
                transition: 'all 0.3s ease',
                '&:hover': { borderColor: 'primary.main', bgcolor: 'action.hover' },
              }}
            >
              <input {...wsiDropzone.getInputProps()} />
              
              {wsiUploading ? (
                <Box>
                  <CircularProgress size={40} />
                  <Typography sx={{ mt: 2 }}>Uploading image...</Typography>
                </Box>
              ) : wsiSlideId ? (
                <Box>
                  <CheckCircleIcon sx={{ fontSize: 48, color: 'success.main', mb: 1 }} />
                  <Typography variant="h6" color="success.main">✓ {wsiFile?.name}</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Slide ID: {wsiSlideId}
                  </Typography>
                </Box>
              ) : (
                <Box>
                  <CloudUploadIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
                  <Typography variant="h6">Drop SVS file here or click to browse</Typography>
                  <Typography variant="body2" color="text.secondary">
                    Supports: .svs, .tif, .tiff, .ndpi
                  </Typography>
                </Box>
              )}
            </Paper>

            <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
              <Button variant="contained" onClick={handleNext} endIcon={<ArrowForwardIcon />}>
                Continue
              </Button>
              <Button onClick={handleNext}>Skip</Button>
            </Box>
          </StepContent>
        </Step>

        {/* Step 2: Demographics */}
        <Step>
          <StepLabel
            StepIconComponent={() => (
              <Box sx={{
                width: 32, height: 32, borderRadius: '50%',
                bgcolor: activeStep >= 1 ? 'primary.main' : 'grey.300',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}>
                <PersonIcon sx={{ color: 'white', fontSize: 18 }} />
              </Box>
            )}
          >
            <Typography fontWeight={500}>Patient Demographics</Typography>
          </StepLabel>
          <StepContent>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Age at Diagnosis"
                  type="number"
                  value={age}
                  onChange={(e) => setAge(e.target.value ? parseInt(e.target.value) : '')}
                  InputProps={{
                    endAdornment: <InputAdornment position="end">years</InputAdornment>,
                  }}
                  inputProps={{ min: 0, max: 120 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Gender</InputLabel>
                  <Select value={gender} onChange={(e) => setGender(e.target.value)} label="Gender">
                    <MenuItem value="">Not specified</MenuItem>
                    <MenuItem value="male">Male</MenuItem>
                    <MenuItem value="female">Female</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Race</InputLabel>
                  <Select value={race} onChange={(e) => setRace(e.target.value)} label="Race">
                    <MenuItem value="">Not specified</MenuItem>
                    <MenuItem value="white">White</MenuItem>
                    <MenuItem value="black">Black or African American</MenuItem>
                    <MenuItem value="asian">Asian</MenuItem>
                    <MenuItem value="other">Other</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Ethnicity</InputLabel>
                  <Select value={ethnicity} onChange={(e) => setEthnicity(e.target.value)} label="Ethnicity">
                    <MenuItem value="">Not specified</MenuItem>
                    <MenuItem value="hispanic">Hispanic or Latino</MenuItem>
                    <MenuItem value="not_hispanic">Not Hispanic or Latino</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>

            <Box sx={{ mt: 3, display: 'flex', gap: 1 }}>
              <Button onClick={handleBack} startIcon={<ArrowBackIcon />}>Back</Button>
              <Button variant="contained" onClick={handleNext} endIcon={<ArrowForwardIcon />}>
                Continue
              </Button>
            </Box>
          </StepContent>
        </Step>

        {/* Step 3: Clinical Data */}
        <Step>
          <StepLabel
            StepIconComponent={() => (
              <Box sx={{
                width: 32, height: 32, borderRadius: '50%',
                bgcolor: activeStep >= 2 ? 'primary.main' : 'grey.300',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}>
                <LocalHospitalIcon sx={{ color: 'white', fontSize: 18 }} />
              </Box>
            )}
          >
            <Typography fontWeight={500}>Clinical Information</Typography>
          </StepLabel>
          <StepContent>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Tumor Stage (AJCC)</InputLabel>
                  <Select value={tumorStage} onChange={(e) => setTumorStage(e.target.value)} label="Tumor Stage (AJCC)">
                    <MenuItem value="">Not specified</MenuItem>
                    {TUMOR_STAGES.map(s => (
                      <MenuItem key={s.value} value={s.value}>{s.label}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Tumor Grade</InputLabel>
                  <Select value={tumorGrade} onChange={(e) => setTumorGrade(e.target.value)} label="Tumor Grade">
                    <MenuItem value="">Not specified</MenuItem>
                    {TUMOR_GRADES.map(g => (
                      <MenuItem key={g.value} value={g.value}>{g.label}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Primary Tumor Site</InputLabel>
                  <Select value={tumorSite} onChange={(e) => setTumorSite(e.target.value)} label="Primary Tumor Site">
                    <MenuItem value="">Not specified</MenuItem>
                    {TUMOR_SITES.map(site => (
                      <MenuItem key={site} value={site.toLowerCase()}>{site}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>HPV Status</InputLabel>
                  <Select value={hpvStatus} onChange={(e) => setHpvStatus(e.target.value)} label="HPV Status">
                    <MenuItem value="">Unknown</MenuItem>
                    <MenuItem value="positive">HPV Positive</MenuItem>
                    <MenuItem value="negative">HPV Negative</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Smoking History"
                  type="number"
                  value={smokingPackYears}
                  onChange={(e) => setSmokingPackYears(e.target.value ? parseInt(e.target.value) : '')}
                  InputProps={{
                    endAdornment: <InputAdornment position="end">pack-years</InputAdornment>,
                  }}
                  inputProps={{ min: 0 }}
                  helperText="Pack-years = (packs per day) × (years smoked)"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Alcohol History</InputLabel>
                  <Select value={alcoholHistory} onChange={(e) => setAlcoholHistory(e.target.value)} label="Alcohol History">
                    <MenuItem value="">Unknown</MenuItem>
                    <MenuItem value="yes">Yes</MenuItem>
                    <MenuItem value="no">No</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>Treatment Received</Typography>
                <FormGroup row>
                  {TREATMENT_TYPES.map(treatment => (
                    <FormControlLabel
                      key={treatment}
                      control={
                        <Checkbox
                          checked={treatments.includes(treatment)}
                          onChange={() => handleTreatmentToggle(treatment)}
                        />
                      }
                      label={treatment}
                    />
                  ))}
                </FormGroup>
              </Grid>
            </Grid>

            <Box sx={{ mt: 3, display: 'flex', gap: 1 }}>
              <Button onClick={handleBack} startIcon={<ArrowBackIcon />}>Back</Button>
              <Button variant="contained" onClick={handleNext} endIcon={<ArrowForwardIcon />}>
                Continue
              </Button>
            </Box>
          </StepContent>
        </Step>

        {/* Step 4: Mutations */}
        <Step>
          <StepLabel
            StepIconComponent={() => (
              <Box sx={{
                width: 32, height: 32, borderRadius: '50%',
                bgcolor: activeStep >= 3 ? 'primary.main' : 'grey.300',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}>
                <BiotechIcon sx={{ color: 'white', fontSize: 18 }} />
              </Box>
            )}
          >
            <Typography fontWeight={500}>Gene Mutations</Typography>
          </StepLabel>
          <StepContent>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Select which driver gene mutations are present in the patient's tumor:
            </Typography>

            <Grid container spacing={2}>
              {DRIVER_GENES.map(({ gene, description }) => (
                <Grid item xs={12} sm={6} md={4} key={gene}>
                  <Paper
                    sx={{
                      p: 2,
                      cursor: 'pointer',
                      border: '2px solid',
                      borderColor: selectedMutations.includes(gene) ? 'error.main' : 'grey.200',
                      bgcolor: selectedMutations.includes(gene) ? 'error.50' : 'background.paper',
                      transition: 'all 0.2s ease',
                      '&:hover': {
                        borderColor: selectedMutations.includes(gene) ? 'error.dark' : 'primary.main',
                        bgcolor: selectedMutations.includes(gene) ? 'error.100' : 'action.hover',
                      },
                    }}
                    onClick={() => handleMutationToggle(gene)}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Checkbox
                        checked={selectedMutations.includes(gene)}
                        color="error"
                        sx={{ p: 0 }}
                      />
                      <Box>
                        <Typography fontWeight={600} color={selectedMutations.includes(gene) ? 'error.main' : 'text.primary'}>
                          {gene}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {description}
                        </Typography>
                      </Box>
                    </Box>
                  </Paper>
                </Grid>
              ))}
            </Grid>

            {selectedMutations.length > 0 && (
              <Alert severity="info" sx={{ mt: 2 }}>
                <strong>{selectedMutations.length}</strong> mutation(s) selected: {selectedMutations.join(', ')}
              </Alert>
            )}

            <Box sx={{ mt: 3, display: 'flex', gap: 1 }}>
              <Button onClick={handleBack} startIcon={<ArrowBackIcon />}>Back</Button>
            </Box>
          </StepContent>
        </Step>
      </Stepper>

      {/* Summary & Submit */}
      <Paper sx={{ mt: 4, p: 3 }}>
        <Typography variant="h6" gutterBottom>Summary</Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Pathology Image</Typography>
            <Typography>{wsiSlideId ? '✓ Uploaded' : 'Not provided'}</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Age / Gender</Typography>
            <Typography>
              {age ? `${age} years` : 'N/A'} / {gender || 'N/A'}
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Tumor Stage</Typography>
            <Typography>{tumorStage ? tumorStage.toUpperCase() : 'N/A'}</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="caption" color="text.secondary">Mutations</Typography>
            <Typography>
              {selectedMutations.length > 0 ? selectedMutations.join(', ') : 'None selected'}
            </Typography>
          </Grid>
        </Grid>

        <Divider sx={{ my: 3 }} />

        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Button
            variant="contained"
            size="large"
            onClick={handleSubmit}
            disabled={!canSubmit || loading}
            startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
            sx={{ px: 6, py: 1.5 }}
          >
            {loading ? 'Analyzing...' : 'Generate Survival Prediction'}
          </Button>
        </Box>

        {loading && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress />
            <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 1 }}>
              Processing through MOSAIC neural network...
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  )
}
