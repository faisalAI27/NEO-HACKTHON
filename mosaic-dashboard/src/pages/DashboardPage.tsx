import { useState } from 'react';
import {
  Box,
  Container,
  Grid,
  Paper,
  Tab,
  Tabs,
  Typography,
  IconButton,
  Chip,
  AppBar,
  Toolbar,
  Button,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Avatar,
  Tooltip,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  CloudUpload,
  Timeline,
  Visibility,
  Psychology,
  Menu as MenuIcon,
  Home,
  Settings,
  Help,
  Biotech,
  ChevronLeft,
} from '@mui/icons-material';
import PatientInputForm from '../components/PatientInputForm';
import SurvivalCurve from '../components/SurvivalCurve';
import WSIViewer from '../components/WSIViewer';
import ExplainabilityDashboard from '../components/ExplainabilityDashboard';
import { PredictionResult } from '../types';
import { gradients } from '../theme';

interface DashboardPageProps {
  onBack: () => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 0 }}>{children}</Box>}
    </div>
  );
}

const DRAWER_WIDTH = 260;

const navigationItems = [
  { icon: <CloudUpload />, label: 'Patient Input', id: 0 },
  { icon: <Timeline />, label: 'Survival Analysis', id: 1 },
  { icon: <Visibility />, label: 'WSI Viewer', id: 2 },
  { icon: <Psychology />, label: 'Explainability', id: 3 },
];

export default function DashboardPage({ onBack }: DashboardPageProps) {
  const [tabValue, setTabValue] = useState(0);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [selectedSlideId, setSelectedSlideId] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(true);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  const handleTabChange = (newValue: number) => {
    setTabValue(newValue);
  };

  const handlePredictionComplete = (result: PredictionResult) => {
    setPredictionResult(result);
    setTabValue(1);
  };

  // Sample prediction data for demo
  const demoResult: PredictionResult = predictionResult || {
    patient_id: 'TCGA-CV-A6JY',
    risk_score: 0.73,
    risk_group: 'high',
    survival_probability: {
      '12_months': 0.85,
      '24_months': 0.72,
      '36_months': 0.61,
      '60_months': 0.48,
    },
    confidence_interval: { lower: 0.65, upper: 0.81 },
    attention_weights: {
      clinical: 0.22,
      transcriptomics: 0.28,
      methylation: 0.18,
      mutations: 0.12,
      pathology: 0.20,
    },
    gene_importance: [
      { gene: 'TP53', importance: 0.89 },
      { gene: 'CDKN2A', importance: 0.76 },
      { gene: 'PIK3CA', importance: 0.68 },
      { gene: 'NOTCH1', importance: 0.61 },
      { gene: 'FAT1', importance: 0.54 },
      { gene: 'CASP8', importance: 0.49 },
      { gene: 'NSD1', importance: 0.42 },
      { gene: 'HRAS', importance: 0.38 },
    ],
  };

  const drawer = (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: gradients.hero,
        color: 'white',
      }}
    >
      {/* Logo */}
      <Box sx={{ p: 3, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Avatar sx={{ bgcolor: 'secondary.main', width: 44, height: 44 }}>
          <Biotech />
        </Avatar>
        <Box>
          <Typography variant="h6" fontWeight={700} letterSpacing={1}>
            MOSAIC
          </Typography>
          <Typography variant="caption" sx={{ opacity: 0.7 }}>
            Survival Analysis
          </Typography>
        </Box>
      </Box>

      <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)' }} />

      {/* Navigation */}
      <List sx={{ px: 2, py: 3, flexGrow: 1 }}>
        {navigationItems.map((item) => (
          <ListItem
            key={item.id}
            onClick={() => handleTabChange(item.id)}
            sx={{
              mb: 1,
              borderRadius: 2,
              cursor: 'pointer',
              bgcolor: tabValue === item.id ? 'rgba(56,161,105,0.2)' : 'transparent',
              border: tabValue === item.id ? '1px solid rgba(56,161,105,0.3)' : '1px solid transparent',
              '&:hover': {
                bgcolor: 'rgba(255,255,255,0.1)',
              },
            }}
          >
            <ListItemIcon sx={{ color: tabValue === item.id ? 'secondary.main' : 'rgba(255,255,255,0.7)', minWidth: 40 }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{
                fontWeight: tabValue === item.id ? 600 : 400,
                color: tabValue === item.id ? 'white' : 'rgba(255,255,255,0.8)',
              }}
            />
            {item.id === 1 && predictionResult && (
              <Chip size="small" label="Ready" color="success" sx={{ height: 20, fontSize: '0.7rem' }} />
            )}
          </ListItem>
        ))}
      </List>

      <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)' }} />

      {/* Footer */}
      <Box sx={{ p: 2 }}>
        <Button
          fullWidth
          startIcon={<Home />}
          onClick={onBack}
          sx={{
            color: 'rgba(255,255,255,0.7)',
            justifyContent: 'flex-start',
            '&:hover': { bgcolor: 'rgba(255,255,255,0.1)', color: 'white' },
          }}
        >
          Back to Home
        </Button>
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* Sidebar */}
      {!isMobile && (
        <Drawer
          variant="permanent"
          sx={{
            width: DRAWER_WIDTH,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: DRAWER_WIDTH,
              boxSizing: 'border-box',
              border: 'none',
            },
          }}
        >
          {drawer}
        </Drawer>
      )}

      {/* Mobile Drawer */}
      {isMobile && (
        <Drawer
          variant="temporary"
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
          sx={{
            '& .MuiDrawer-paper': {
              width: DRAWER_WIDTH,
              boxSizing: 'border-box',
            },
          }}
        >
          {drawer}
        </Drawer>
      )}

      {/* Main Content */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Top Bar */}
        <AppBar
          position="static"
          elevation={0}
          sx={{ bgcolor: 'white', borderBottom: 1, borderColor: 'divider' }}
        >
          <Toolbar>
            {isMobile && (
              <IconButton onClick={() => setDrawerOpen(true)} sx={{ mr: 2 }}>
                <MenuIcon />
              </IconButton>
            )}
            <Typography variant="h5" color="text.primary" fontWeight={600} sx={{ flexGrow: 1 }}>
              {navigationItems.find((item) => item.id === tabValue)?.label}
            </Typography>
            
            {predictionResult && (
              <Chip
                label={`Patient: ${predictionResult.patient_id}`}
                color="primary"
                variant="outlined"
                sx={{ mr: 2 }}
              />
            )}
            
            <Tooltip title="Help">
              <IconButton>
                <Help />
              </IconButton>
            </Tooltip>
            <Tooltip title="Settings">
              <IconButton>
                <Settings />
              </IconButton>
            </Tooltip>
          </Toolbar>
        </AppBar>

        {/* Content Area */}
        <Box sx={{ flexGrow: 1, p: 3, overflow: 'auto' }}>
          {/* Tab 0: Patient Upload */}
          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              <Grid item xs={12} lg={8}>
                <Paper sx={{ p: 3 }}>
                  <PatientInputForm
                    onPredictionComplete={handlePredictionComplete}
                    onSlideSelected={setSelectedSlideId}
                  />
                </Paper>
              </Grid>
              <Grid item xs={12} lg={4}>
                <Paper sx={{ p: 3, mb: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Input Guide
                  </Typography>
                  <Box component="ul" sx={{ pl: 2, '& li': { mb: 1.5 } }}>
                    <li>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Pathology:</strong> Upload .svs whole-slide image
                      </Typography>
                    </li>
                    <li>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Demographics:</strong> Age, gender, ethnicity
                      </Typography>
                    </li>
                    <li>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Clinical:</strong> Stage, grade, HPV status, smoking
                      </Typography>
                    </li>
                    <li>
                      <Typography variant="body2" color="text.secondary">
                        <strong>Mutations:</strong> Select driver gene mutations
                      </Typography>
                    </li>
                  </Box>
                </Paper>
                <Paper
                  sx={{
                    p: 3,
                    background: gradients.secondary,
                    color: 'white',
                    cursor: 'pointer',
                    '&:hover': { filter: 'brightness(1.05)' },
                  }}
                  onClick={() => {
                    setPredictionResult(demoResult);
                    setTabValue(1);
                  }}
                >
                  <Typography variant="h6" gutterBottom>
                    🎯 Try Demo Mode
                  </Typography>
                  <Typography variant="body2" sx={{ opacity: 0.9 }}>
                    Skip data entry and explore the dashboard with sample predictions.
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Tab 1: Survival Analysis */}
          <TabPanel value={tabValue} index={1}>
            <Grid container spacing={3}>
              {/* Risk Summary Card */}
              <Grid item xs={12}>
                <Paper sx={{ p: 3 }}>
                  <Grid container spacing={3} alignItems="center">
                    <Grid item xs={12} md={4}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                          Risk Score
                        </Typography>
                        <Typography
                          variant="h2"
                          color={(demoResult.risk_group || 'medium') === 'high' ? 'error.main' : 'success.main'}
                          fontWeight={700}
                        >
                          {(demoResult.risk_score ?? 0).toFixed(2)}
                        </Typography>
                        <Chip
                          label={(demoResult.risk_group || 'medium').toUpperCase() + ' RISK'}
                          color={(demoResult.risk_group || 'medium') === 'high' ? 'error' : 'success'}
                          sx={{ mt: 1 }}
                        />
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={8}>
                      <Grid container spacing={2}>
                        {Object.entries(demoResult.survival_probability || {}).map(([time, prob]) => (
                          <Grid item xs={6} sm={3} key={time}>
                            <Paper elevation={0} sx={{ p: 2, bgcolor: 'background.default', textAlign: 'center' }}>
                              <Typography variant="h4" color="primary.main" fontWeight={600}>
                                {((prob as number) * 100).toFixed(0)}%
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {String(time).includes('_') ? String(time).replace('_', ' ') : `${Math.round(Number(time) / 30)} months`}
                              </Typography>
                            </Paper>
                          </Grid>
                        ))}
                      </Grid>
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>

              {/* Survival Curve */}
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 3, height: 400 }}>
                  <Typography variant="h6" gutterBottom>
                    Kaplan-Meier Survival Curve
                  </Typography>
                  <SurvivalCurve
                    data={Object.entries(demoResult.survival_probability || {}).map(([time, prob]) => {
                      // Handle both "12_months" format and numeric "365" format
                      let timeValue: number;
                      if (time.includes('_')) {
                        timeValue = parseInt(time.split('_')[0]);
                      } else {
                        // Convert days to months for display
                        timeValue = Math.round(parseInt(time) / 30);
                      }
                      return {
                        time: timeValue,
                        probability: prob as number,
                        lower: (prob as number) - 0.08,
                        upper: (prob as number) + 0.08,
                      };
                    }).sort((a, b) => a.time - b.time)}
                    patientId={demoResult.patient_id}
                  />
                </Paper>
              </Grid>

              {/* Explainability */}
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 3, height: 400 }}>
                  <Typography variant="h6" gutterBottom>
                    Modality Contributions
                  </Typography>
                  <ExplainabilityDashboard
                    attentionWeights={demoResult.attention_weights || {}}
                    geneImportance={demoResult.gene_importance || []}
                  />
                </Paper>
              </Grid>

              {/* Gene Importance */}
              <Grid item xs={12}>
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Top Contributing Genes
                  </Typography>
                  <Grid container spacing={2}>
                    {(demoResult.gene_importance && demoResult.gene_importance.length > 0) ? (
                      demoResult.gene_importance.map((gene, idx) => (
                        <Grid item xs={6} sm={4} md={3} lg={1.5} key={gene.gene}>
                          <Paper
                            elevation={0}
                            sx={{
                              p: 2,
                              textAlign: 'center',
                              bgcolor: idx < 3 ? 'primary.50' : 'background.default',
                              border: idx < 3 ? 1 : 0,
                              borderColor: 'primary.200',
                            }}
                          >
                            <Typography variant="subtitle2" fontWeight={600}>
                              {gene.gene}
                            </Typography>
                            <Typography variant="h6" color="primary.main">
                              {(gene.importance * 100).toFixed(0)}%
                            </Typography>
                          </Paper>
                        </Grid>
                      ))
                    ) : (
                      <Grid item xs={12}>
                        <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                          Gene importance data not available in demo mode. Upload omics data for detailed gene analysis.
                        </Typography>
                      </Grid>
                    )}
                  </Grid>
                </Paper>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Tab 2: WSI Viewer */}
          <TabPanel value={tabValue} index={2}>
            <Paper sx={{ p: 3, height: 'calc(100vh - 200px)' }}>
              <WSIViewer
                slideId={selectedSlideId || 'demo_slide'}
                attentionHeatmap={demoResult.attention_weights}
              />
            </Paper>
          </TabPanel>

          {/* Tab 3: Explainability */}
          <TabPanel value={tabValue} index={3}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Cross-Modal Attention Analysis
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    The attention weights show how the model weighs different data modalities when making predictions.
                  </Typography>
                  <ExplainabilityDashboard
                    attentionWeights={demoResult.attention_weights}
                    geneImportance={demoResult.gene_importance || []}
                    detailed
                  />
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    Interpretation Guide
                  </Typography>
                  <Box sx={{ '& > div': { mb: 2 } }}>
                    <Box>
                      <Chip label="High Pathology Weight" size="small" sx={{ mb: 1 }} />
                      <Typography variant="body2" color="text.secondary">
                        Tumor morphology and tissue architecture are key predictors
                      </Typography>
                    </Box>
                    <Box>
                      <Chip label="High Transcriptomics Weight" size="small" sx={{ mb: 1 }} />
                      <Typography variant="body2" color="text.secondary">
                        Gene expression patterns drive the survival prediction
                      </Typography>
                    </Box>
                    <Box>
                      <Chip label="High Mutation Weight" size="small" sx={{ mb: 1 }} />
                      <Typography variant="body2" color="text.secondary">
                        Specific genomic alterations influence prognosis
                      </Typography>
                    </Box>
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          </TabPanel>
        </Box>
      </Box>
    </Box>
  );
}
