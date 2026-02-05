import { Box, Container, Typography, Button, Grid, Card, CardContent, Chip, Avatar, Stack, LinearProgress } from '@mui/material';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  Visibility,
  Psychology,
  Science,
  DataObject,
  BarChart,
  LocalHospital,
  Speed,
} from '@mui/icons-material';
import { gradients } from '../theme';

interface LandingPageProps {
  onGetStarted: () => void;
}

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: 'easeOut' as const } },
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.15, delayChildren: 0.2 },
  },
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.8 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.5, ease: 'easeOut' as const } },
};

// Stat Card Component
const StatCard = ({ value, label, color }: { value: string; label: string; color: string }) => (
  <motion.div variants={scaleIn}>
    <Card
      sx={{
        background: 'rgba(255,255,255,0.1)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255,255,255,0.2)',
        color: 'white',
        textAlign: 'center',
        py: 3,
        px: 2,
      }}
    >
      <Typography variant="h3" fontWeight={700} color={color}>
        {value}
      </Typography>
      <Typography variant="body2" sx={{ opacity: 0.8, mt: 1 }}>
        {label}
      </Typography>
    </Card>
  </motion.div>
);

// Feature Card Component
const FeatureCard = ({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) => (
  <motion.div variants={fadeInUp}>
    <Card sx={{ height: '100%', p: 1 }}>
      <CardContent>
        <Box
          sx={{
            width: 56,
            height: 56,
            borderRadius: 3,
            background: gradients.secondary,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            mb: 2,
            color: 'white',
          }}
        >
          {icon}
        </Box>
        <Typography variant="h6" gutterBottom fontWeight={600}>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
      </CardContent>
    </Card>
  </motion.div>
);

// Modality Card Component
const ModalityCard = ({ name, icon, features, color }: { name: string; icon: React.ReactNode; features: string[]; color: string }) => (
  <motion.div variants={fadeInUp}>
    <Card
      sx={{
        height: '100%',
        borderTop: `4px solid ${color}`,
        '&:hover': { borderColor: color },
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Avatar sx={{ bgcolor: color, mr: 2 }}>{icon}</Avatar>
          <Typography variant="h6" fontWeight={600}>
            {name}
          </Typography>
        </Box>
        <Stack spacing={1}>
          {features.map((feature, idx) => (
            <Chip
              key={idx}
              label={feature}
              size="small"
              sx={{ justifyContent: 'flex-start', bgcolor: `${color}15`, color: color }}
            />
          ))}
        </Stack>
      </CardContent>
    </Card>
  </motion.div>
);

// Performance Metric Component
const PerformanceMetric = ({ label, value, target }: { label: string; value: number; target: number }) => (
  <Box sx={{ mb: 3 }}>
    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
      <Typography variant="body2" fontWeight={500}>
        {label}
      </Typography>
      <Typography variant="body2" fontWeight={600} color="secondary.main">
        {value.toFixed(3)} / {target.toFixed(2)}
      </Typography>
    </Box>
    <LinearProgress
      variant="determinate"
      value={(value / target) * 100}
      color="secondary"
      sx={{ height: 10, borderRadius: 5 }}
    />
  </Box>
);

export default function LandingPage({ onGetStarted }: LandingPageProps) {
  return (
    <Box sx={{ overflow: 'hidden' }}>
      {/* Hero Section */}
      <Box
        sx={{
          background: gradients.hero,
          color: 'white',
          py: { xs: 8, md: 12 },
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Background decoration */}
        <Box
          sx={{
            position: 'absolute',
            top: '-50%',
            right: '-10%',
            width: '60%',
            height: '200%',
            background: 'radial-gradient(circle, rgba(56,161,105,0.15) 0%, transparent 70%)',
            pointerEvents: 'none',
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            bottom: '-30%',
            left: '-10%',
            width: '50%',
            height: '100%',
            background: 'radial-gradient(circle, rgba(128,90,213,0.1) 0%, transparent 70%)',
            pointerEvents: 'none',
          }}
        />

        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <motion.div initial="hidden" animate="visible" variants={staggerContainer}>
            <Grid container spacing={6} alignItems="center">
              <Grid item xs={12} md={7}>
                <motion.div variants={fadeInUp}>
                  <Chip
                    label="🔬 Multi-Modal Deep Learning"
                    sx={{
                      bgcolor: 'rgba(56,161,105,0.2)',
                      color: '#68d391',
                      mb: 3,
                      fontWeight: 600,
                    }}
                  />
                </motion.div>
                <motion.div variants={fadeInUp}>
                  <Typography variant="h1" sx={{ mb: 3, fontSize: { xs: '2.5rem', md: '3.5rem' } }}>
                    MOSAIC
                  </Typography>
                </motion.div>
                <motion.div variants={fadeInUp}>
                  <Typography
                    variant="h5"
                    sx={{
                      mb: 2,
                      fontWeight: 400,
                      opacity: 0.9,
                      lineHeight: 1.6,
                    }}
                  >
                    <strong>M</strong>ulti-<strong>O</strong>mics <strong>S</strong>urvival <strong>A</strong>nalysis with{' '}
                    <strong>I</strong>nterpretable <strong>C</strong>ross-modal Attention
                  </Typography>
                </motion.div>
                <motion.div variants={fadeInUp}>
                  <Typography variant="body1" sx={{ mb: 4, opacity: 0.8, maxWidth: 500 }}>
                    A state-of-the-art deep learning framework for cancer survival prediction,
                    integrating pathology images, genomics, and clinical data with explainable AI.
                  </Typography>
                </motion.div>
                <motion.div variants={fadeInUp}>
                  <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                    <Button
                      variant="contained"
                      color="secondary"
                      size="large"
                      onClick={onGetStarted}
                      sx={{ px: 4, py: 1.5 }}
                    >
                      Launch Dashboard
                    </Button>
                    <Button
                      variant="outlined"
                      size="large"
                      sx={{
                        borderColor: 'rgba(255,255,255,0.5)',
                        color: 'white',
                        '&:hover': { borderColor: 'white', bgcolor: 'rgba(255,255,255,0.1)' },
                      }}
                    >
                      View Documentation
                    </Button>
                  </Stack>
                </motion.div>
              </Grid>

              {/* Stats */}
              <Grid item xs={12} md={5}>
                <motion.div variants={staggerContainer} initial="hidden" animate="visible">
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <StatCard value="0.728" label="Mean C-Index" color="#68d391" />
                    </Grid>
                    <Grid item xs={6}>
                      <StatCard value="5" label="Data Modalities" color="#63b3ed" />
                    </Grid>
                    <Grid item xs={6}>
                      <StatCard value="82" label="TCGA Patients" color="#f6ad55" />
                    </Grid>
                    <Grid item xs={6}>
                      <StatCard value="100%" label="Interpretable" color="#b794f4" />
                    </Grid>
                  </Grid>
                </motion.div>
              </Grid>
            </Grid>
          </motion.div>
        </Container>
      </Box>

      {/* Features Section */}
      <Box sx={{ py: { xs: 8, md: 12 }, bgcolor: 'background.default' }}>
        <Container maxWidth="lg">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true }} variants={staggerContainer}>
            <motion.div variants={fadeInUp}>
              <Typography variant="h2" textAlign="center" gutterBottom>
                Key Capabilities
              </Typography>
              <Typography
                variant="body1"
                textAlign="center"
                color="text.secondary"
                sx={{ mb: 6, maxWidth: 600, mx: 'auto' }}
              >
                MOSAIC leverages cutting-edge deep learning architecture to provide accurate and interpretable survival predictions.
              </Typography>
            </motion.div>

            <Grid container spacing={4}>
              <Grid item xs={12} sm={6} md={3}>
                <FeatureCard
                  icon={<Psychology sx={{ fontSize: 28 }} />}
                  title="Cross-Modal Attention"
                  description="Perceiver-based fusion learns complex interactions between different data modalities."
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <FeatureCard
                  icon={<Visibility sx={{ fontSize: 28 }} />}
                  title="WSI Analysis"
                  description="UNI foundation model extracts histopathology features from whole-slide images."
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <FeatureCard
                  icon={<Science sx={{ fontSize: 28 }} />}
                  title="Multi-Omics Integration"
                  description="Seamlessly combines transcriptomics, methylation, and mutation data."
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <FeatureCard
                  icon={<BarChart sx={{ fontSize: 28 }} />}
                  title="Explainable AI"
                  description="Attention heatmaps reveal which features drive survival predictions."
                />
              </Grid>
            </Grid>
          </motion.div>
        </Container>
      </Box>

      {/* Data Modalities Section */}
      <Box sx={{ py: { xs: 8, md: 12 }, bgcolor: 'white' }}>
        <Container maxWidth="lg">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true }} variants={staggerContainer}>
            <motion.div variants={fadeInUp}>
              <Typography variant="h2" textAlign="center" gutterBottom>
                Multi-Modal Data Integration
              </Typography>
              <Typography
                variant="body1"
                textAlign="center"
                color="text.secondary"
                sx={{ mb: 6, maxWidth: 700, mx: 'auto' }}
              >
                MOSAIC processes five distinct data modalities, each providing unique insights into tumor biology and patient prognosis.
              </Typography>
            </motion.div>

            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={4}>
                <ModalityCard
                  name="Pathology"
                  icon={<LocalHospital />}
                  color="#e53e3e"
                  features={['H&E Whole Slide Images', 'UNI Feature Extraction', 'Gated Attention Pooling']}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <ModalityCard
                  name="Transcriptomics"
                  icon={<DataObject />}
                  color="#38a169"
                  features={['RNA-seq Expression', '7,923 Selected Genes', 'Z-Score Normalization']}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <ModalityCard
                  name="Methylation"
                  icon={<Science />}
                  color="#805ad5"
                  features={['DNA Methylation Beta', '10,057 CpG Sites', 'M-Value Transformation']}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <ModalityCard
                  name="Mutations"
                  icon={<TrendingUp />}
                  color="#ed8936"
                  features={['Somatic Mutations', '15 Driver Genes', 'Mutation Burden Score']}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <ModalityCard
                  name="Clinical"
                  icon={<Speed />}
                  color="#3182ce"
                  features={['Age & Demographics', 'Tumor Stage/Grade', 'Treatment History']}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={4}>
                <Card
                  sx={{
                    height: '100%',
                    background: gradients.primary,
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    textAlign: 'center',
                    cursor: 'pointer',
                    '&:hover': { filter: 'brightness(1.1)' },
                  }}
                  onClick={onGetStarted}
                >
                  <CardContent>
                    <Typography variant="h5" fontWeight={600} gutterBottom>
                      Try It Now
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      Upload patient data and get survival predictions
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </motion.div>
        </Container>
      </Box>

      {/* Performance Section */}
      <Box sx={{ py: { xs: 8, md: 12 }, bgcolor: 'background.default' }}>
        <Container maxWidth="md">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true }} variants={staggerContainer}>
            <motion.div variants={fadeInUp}>
              <Typography variant="h2" textAlign="center" gutterBottom>
                Model Performance
              </Typography>
              <Typography variant="body1" textAlign="center" color="text.secondary" sx={{ mb: 6 }}>
                Validated through rigorous 5-fold cross-validation on TCGA-HNSC dataset
              </Typography>
            </motion.div>

            <motion.div variants={fadeInUp}>
              <Card sx={{ p: 4 }}>
                <PerformanceMetric label="Concordance Index (C-Index)" value={0.728} target={1.0} />
                <PerformanceMetric label="Time-Dependent AUC @ 3 Years" value={0.812} target={1.0} />
                <PerformanceMetric label="Brier Score (Lower is Better)" value={0.18} target={0.25} />

                <Box sx={{ mt: 4, p: 3, bgcolor: 'background.default', borderRadius: 2 }}>
                  <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    Key Findings from Ablation Study
                  </Typography>
                  <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={6} md={3}>
                      <Typography variant="h6" color="primary.main">+12%</Typography>
                      <Typography variant="caption" color="text.secondary">
                        vs Clinical Only
                      </Typography>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="h6" color="secondary.main">+8%</Typography>
                      <Typography variant="caption" color="text.secondary">
                        vs WSI Only
                      </Typography>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="h6" color="warning.main">+15%</Typography>
                      <Typography variant="caption" color="text.secondary">
                        vs RNA Only
                      </Typography>
                    </Grid>
                    <Grid item xs={6} md={3}>
                      <Typography variant="h6" color="info.main">Best</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Multi-Modal Fusion
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
              </Card>
            </motion.div>
          </motion.div>
        </Container>
      </Box>

      {/* Architecture Section */}
      <Box sx={{ py: { xs: 8, md: 12 }, background: gradients.hero, color: 'white' }}>
        <Container maxWidth="lg">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true }} variants={staggerContainer}>
            <motion.div variants={fadeInUp}>
              <Typography variant="h2" textAlign="center" gutterBottom>
                Model Architecture
              </Typography>
              <Typography variant="body1" textAlign="center" sx={{ mb: 6, opacity: 0.8, maxWidth: 600, mx: 'auto' }}>
                MOSAIC employs a sophisticated encoder-fusion-predictor architecture
              </Typography>
            </motion.div>

            <motion.div variants={fadeInUp}>
              <Box
                sx={{
                  bgcolor: 'rgba(255,255,255,0.05)',
                  borderRadius: 4,
                  p: 4,
                  border: '1px solid rgba(255,255,255,0.1)',
                }}
              >
                <pre
                  style={{
                    fontFamily: 'monospace',
                    fontSize: '0.85rem',
                    overflow: 'auto',
                    margin: 0,
                    lineHeight: 1.6,
                  }}
                >
{`┌─────────────────────────────────────────────────────────────────────────────┐
│                           MOSAIC ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   WSI    │  │   RNA    │  │   Meth   │  │ Mutation │  │ Clinical │       │
│  │ Encoder  │  │ Encoder  │  │ Encoder  │  │ Encoder  │  │ Encoder  │       │
│  │  (UNI)   │  │  (MLP)   │  │  (MLP)   │  │  (MLP)   │  │  (MLP)   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │             │              │
│       └──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘              │
│              │             │             │             │                     │
│              ▼             ▼             ▼             ▼                     │
│        ┌─────────────────────────────────────────────────┐                   │
│        │         Cross-Modal Attention (Perceiver)       │                   │
│        │  • Learnable Latent Queries (16 × 256)          │                   │
│        │  • Multi-Head Cross-Attention (8 heads)         │                   │
│        │  • Feed-Forward Networks                        │                   │
│        └────────────────────┬────────────────────────────┘                   │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │  Survival Head  │                                       │
│                    │   (MLP → σ)     │                                       │
│                    └────────┬────────┘                                       │
│                             │                                                │
│                             ▼                                                │
│                    ┌─────────────────┐                                       │
│                    │   Risk Score    │───▶ Cox PH Loss                       │
│                    └─────────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘`}
                </pre>
              </Box>
            </motion.div>
          </motion.div>
        </Container>
      </Box>

      {/* CTA Section */}
      <Box sx={{ py: { xs: 8, md: 10 }, bgcolor: 'white', textAlign: 'center' }}>
        <Container maxWidth="sm">
          <motion.div initial="hidden" whileInView="visible" viewport={{ once: true }} variants={fadeInUp}>
            <Typography variant="h3" gutterBottom>
              Ready to Explore?
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
              Upload patient data and experience the power of multi-modal survival prediction with explainable AI.
            </Typography>
            <Button variant="contained" color="secondary" size="large" onClick={onGetStarted} sx={{ px: 5, py: 1.5 }}>
              Get Started
            </Button>
          </motion.div>
        </Container>
      </Box>

      {/* Footer */}
      <Box sx={{ py: 4, bgcolor: 'background.default', borderTop: 1, borderColor: 'divider' }}>
        <Container>
          <Typography variant="body2" color="text.secondary" textAlign="center">
            MOSAIC — Multi-Omics Survival Analysis with Interpretable Cross-modal Attention
            <br />
            Built with PyTorch Lightning, FastAPI, and React • TCGA-HNSC Dataset
          </Typography>
        </Container>
      </Box>
    </Box>
  );
}
