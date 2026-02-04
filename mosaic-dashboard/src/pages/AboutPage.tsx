import { Box, Container, Typography, Grid, Paper, Chip, Avatar, Divider, Link } from '@mui/material';
import { motion } from 'framer-motion';
import ModelArchitecture from '../components/ModelArchitecture';
import { gradients } from '../theme';

const MotionBox = motion(Box);

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6 } },
};

const stagger = {
  visible: { transition: { staggerChildren: 0.1 } },
};

interface TeamMember {
  name: string;
  role: string;
  avatar: string;
}

const teamMembers: TeamMember[] = [
  { name: 'Research Team', role: 'Model Development', avatar: '🧬' },
  { name: 'Clinical Team', role: 'Data Annotation', avatar: '⚕️' },
  { name: 'Engineering Team', role: 'System Architecture', avatar: '💻' },
];

const publications = [
  {
    title: 'Multi-Modal Deep Learning for Survival Prediction in Head and Neck Cancer',
    journal: 'Nature Medicine (in preparation)',
    year: 2024,
  },
];

export default function AboutPage() {
  return (
    <Box sx={{ py: 6 }}>
      <Container maxWidth="lg">
        <motion.div initial="hidden" animate="visible" variants={stagger}>
          {/* Header */}
          <MotionBox variants={fadeInUp} sx={{ textAlign: 'center', mb: 6 }}>
            <Chip
              label="About the Project"
              sx={{ mb: 2, bgcolor: 'secondary.100', color: 'secondary.main' }}
            />
            <Typography variant="h2" fontWeight={700} gutterBottom>
              MOSAIC
            </Typography>
            <Typography variant="h5" color="text.secondary" sx={{ maxWidth: 700, mx: 'auto' }}>
              <strong>M</strong>ulti-<strong>O</strong>mics <strong>S</strong>urvival <strong>A</strong>nalysis with{' '}
              <strong>I</strong>nterpretable <strong>C</strong>ross-modal Attention
            </Typography>
          </MotionBox>

          {/* Overview */}
          <MotionBox variants={fadeInUp}>
            <Paper sx={{ p: 4, mb: 4 }}>
              <Typography variant="h5" fontWeight={600} gutterBottom>
                Project Overview
              </Typography>
              <Typography variant="body1" color="text.secondary" paragraph>
                MOSAIC is a state-of-the-art deep learning framework for predicting cancer patient survival
                by integrating multiple data modalities. Our model combines clinical information, genomic data
                (mutations, gene expression, methylation), and whole-slide pathology images using a novel
                cross-modal attention mechanism.
              </Typography>
              <Typography variant="body1" color="text.secondary" paragraph>
                The model was developed and validated using data from The Cancer Genome Atlas (TCGA)
                Head and Neck Squamous Cell Carcinoma (HNSC) cohort, comprising 82 patients with comprehensive
                multi-modal data.
              </Typography>

              <Grid container spacing={2} sx={{ mt: 2 }}>
                <Grid item xs={6} sm={3}>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'primary.50', textAlign: 'center' }}>
                    <Typography variant="h4" fontWeight={700} color="primary.main">82</Typography>
                    <Typography variant="body2">Patients</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'secondary.50', textAlign: 'center' }}>
                    <Typography variant="h4" fontWeight={700} color="secondary.main">5</Typography>
                    <Typography variant="body2">Modalities</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'warning.50', textAlign: 'center' }}>
                    <Typography variant="h4" fontWeight={700} color="warning.main">0.728</Typography>
                    <Typography variant="body2">C-Index</Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'info.50', textAlign: 'center' }}>
                    <Typography variant="h4" fontWeight={700} color="info.main">78%</Typography>
                    <Typography variant="body2">5-Year AUC</Typography>
                  </Paper>
                </Grid>
              </Grid>
            </Paper>
          </MotionBox>

          {/* Architecture */}
          <MotionBox variants={fadeInUp}>
            <Paper sx={{ p: 4, mb: 4 }}>
              <Typography variant="h5" fontWeight={600} gutterBottom>
                Model Architecture
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                Our architecture uses modality-specific encoders followed by a cross-modal attention fusion layer
                to learn optimal combinations of different data types for survival prediction.
              </Typography>
              <ModelArchitecture />
            </Paper>
          </MotionBox>

          {/* Key Features */}
          <MotionBox variants={fadeInUp}>
            <Paper sx={{ p: 4, mb: 4 }}>
              <Typography variant="h5" fontWeight={600} gutterBottom>
                Key Innovations
              </Typography>
              <Grid container spacing={3} sx={{ mt: 1 }}>
                <Grid item xs={12} md={4}>
                  <Box sx={{ p: 2 }}>
                    <Typography variant="h3" sx={{ mb: 1 }}>🔗</Typography>
                    <Typography variant="h6" fontWeight={600} gutterBottom>
                      Cross-Modal Attention
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Self-attention mechanism learns how different modalities interact and contribute
                      to the final prediction, enabling dynamic feature weighting.
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Box sx={{ p: 2 }}>
                    <Typography variant="h3" sx={{ mb: 1 }}>🧩</Typography>
                    <Typography variant="h6" fontWeight={600} gutterBottom>
                      Missing Modality Handling
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Robust to missing data modalities through learned imputation and masked attention,
                      enabling predictions even with incomplete patient data.
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Box sx={{ p: 2 }}>
                    <Typography variant="h3" sx={{ mb: 1 }}>🔍</Typography>
                    <Typography variant="h6" fontWeight={600} gutterBottom>
                      Interpretability
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Attention maps provide insights into which features and modalities drive predictions,
                      supporting clinical decision-making.
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          </MotionBox>

          {/* Data Sources */}
          <MotionBox variants={fadeInUp}>
            <Paper sx={{ p: 4, mb: 4 }}>
              <Typography variant="h5" fontWeight={600} gutterBottom>
                Data Modalities
              </Typography>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                {[
                  { name: 'Clinical', icon: '🏥', desc: 'Age, stage, HPV status, tumor site, smoking/alcohol history' },
                  { name: 'Transcriptomics', icon: '📊', desc: 'RNA-seq gene expression profiles (20,000+ genes)' },
                  { name: 'Methylation', icon: '🔷', desc: 'DNA methylation (450K array, CpG sites)' },
                  { name: 'Mutations', icon: '🧬', desc: 'Somatic mutations (driver genes, TMB)' },
                  { name: 'Pathology', icon: '🔬', desc: 'Whole-slide images (WSI) with ResNet50 features' },
                ].map((mod) => (
                  <Grid item xs={12} sm={6} md={4} key={mod.name}>
                    <Paper
                      variant="outlined"
                      sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Typography variant="h4" sx={{ mr: 2 }}>{mod.icon}</Typography>
                        <Typography variant="subtitle1" fontWeight={600}>{mod.name}</Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        {mod.desc}
                      </Typography>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </MotionBox>

          {/* Technical Details */}
          <MotionBox variants={fadeInUp}>
            <Paper sx={{ p: 4, mb: 4 }}>
              <Typography variant="h5" fontWeight={600} gutterBottom>
                Technical Stack
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 2 }}>
                {[
                  'PyTorch', 'PyTorch Lightning', 'Transformers', 'OpenSlide', 'FastAPI',
                  'React', 'TypeScript', 'Material-UI', 'Docker', 'GitHub Actions',
                ].map((tech) => (
                  <Chip key={tech} label={tech} variant="outlined" />
                ))}
              </Box>
              
              <Divider sx={{ my: 3 }} />
              
              <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                Training Configuration
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={4}>
                  <Typography variant="body2" color="text.secondary">Optimizer</Typography>
                  <Typography variant="body1" fontWeight={500}>AdamW</Typography>
                </Grid>
                <Grid item xs={6} sm={4}>
                  <Typography variant="body2" color="text.secondary">Learning Rate</Typography>
                  <Typography variant="body1" fontWeight={500}>1e-4</Typography>
                </Grid>
                <Grid item xs={6} sm={4}>
                  <Typography variant="body2" color="text.secondary">Batch Size</Typography>
                  <Typography variant="body1" fontWeight={500}>16</Typography>
                </Grid>
                <Grid item xs={6} sm={4}>
                  <Typography variant="body2" color="text.secondary">Hidden Dimension</Typography>
                  <Typography variant="body1" fontWeight={500}>256</Typography>
                </Grid>
                <Grid item xs={6} sm={4}>
                  <Typography variant="body2" color="text.secondary">Attention Heads</Typography>
                  <Typography variant="body1" fontWeight={500}>8</Typography>
                </Grid>
                <Grid item xs={6} sm={4}>
                  <Typography variant="body2" color="text.secondary">Dropout</Typography>
                  <Typography variant="body1" fontWeight={500}>0.3</Typography>
                </Grid>
              </Grid>
            </Paper>
          </MotionBox>

          {/* Publications */}
          <MotionBox variants={fadeInUp}>
            <Paper sx={{ p: 4, background: gradients.hero, color: 'white' }}>
              <Typography variant="h5" fontWeight={600} gutterBottom>
                Publications & Citations
              </Typography>
              <Box sx={{ mt: 2 }}>
                {publications.map((pub, idx) => (
                  <Box key={idx} sx={{ mb: 2 }}>
                    <Typography variant="subtitle1" fontWeight={600}>
                      {pub.title}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.8 }}>
                      {pub.journal} • {pub.year}
                    </Typography>
                  </Box>
                ))}
              </Box>
              <Divider sx={{ borderColor: 'rgba(255,255,255,0.2)', my: 3 }} />
              <Typography variant="body2" sx={{ opacity: 0.8 }}>
                For questions or collaborations, please contact the research team.
              </Typography>
            </Paper>
          </MotionBox>
        </motion.div>
      </Container>
    </Box>
  );
}
