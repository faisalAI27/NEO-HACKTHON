import { Box, Typography, Paper, Chip } from '@mui/material';
import { motion } from 'framer-motion';

const MotionPaper = motion(Paper);

interface ModalityNode {
  id: string;
  name: string;
  color: string;
  icon: string;
  features: string[];
}

const modalities: ModalityNode[] = [
  {
    id: 'clinical',
    name: 'Clinical',
    color: '#38a169',
    icon: '🏥',
    features: ['Age', 'Stage', 'HPV Status', 'Site'],
  },
  {
    id: 'genomic',
    name: 'Genomic',
    color: '#805ad5',
    icon: '🧬',
    features: ['Mutations', 'CNV', 'Gene Expression'],
  },
  {
    id: 'pathology',
    name: 'Pathology',
    color: '#dd6b20',
    icon: '🔬',
    features: ['WSI Features', 'Tumor Regions', 'Cell Types'],
  },
  {
    id: 'methylation',
    name: 'Methylation',
    color: '#3182ce',
    icon: '🔷',
    features: ['CpG Sites', 'Promoter Status', 'Gene Silencing'],
  },
  {
    id: 'transcriptomic',
    name: 'Transcriptomic',
    color: '#d53f8c',
    icon: '📊',
    features: ['RNA-seq', 'Expression Levels', 'Pathways'],
  },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: 'easeOut' as const,
    },
  },
};

export default function ModelArchitecture() {
  return (
    <Box sx={{ py: 4 }}>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Input Modalities */}
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap', mb: 4 }}>
          {modalities.map((modality, index) => (
            <motion.div key={modality.id} variants={itemVariants}>
              <MotionPaper
                elevation={3}
                whileHover={{ scale: 1.05, boxShadow: '0 8px 30px rgba(0,0,0,0.15)' }}
                sx={{
                  p: 2,
                  minWidth: 160,
                  textAlign: 'center',
                  borderTop: `4px solid ${modality.color}`,
                  borderRadius: 2,
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                }}
              >
                <Typography variant="h4" sx={{ mb: 1 }}>
                  {modality.icon}
                </Typography>
                <Typography variant="subtitle1" fontWeight={600} sx={{ color: modality.color }}>
                  {modality.name}
                </Typography>
                <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5, justifyContent: 'center' }}>
                  {modality.features.slice(0, 2).map((feature) => (
                    <Chip
                      key={feature}
                      label={feature}
                      size="small"
                      sx={{
                        fontSize: '0.65rem',
                        height: 20,
                        backgroundColor: `${modality.color}15`,
                        color: modality.color,
                      }}
                    />
                  ))}
                </Box>
              </MotionPaper>
            </motion.div>
          ))}
        </Box>

        {/* Arrows Down */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <Typography sx={{ fontSize: '2rem', color: 'grey.400' }}>↓↓↓↓↓</Typography>
            </Box>
          </motion.div>
        </Box>

        {/* Encoder Layer */}
        <motion.div variants={itemVariants}>
          <Paper
            elevation={4}
            sx={{
              p: 3,
              mb: 3,
              mx: 'auto',
              maxWidth: 600,
              textAlign: 'center',
              background: 'linear-gradient(135deg, #1a365d 0%, #2d3748 100%)',
              color: 'white',
              borderRadius: 3,
            }}
          >
            <Typography variant="h6" fontWeight={600} sx={{ mb: 1 }}>
              🧠 Modality-Specific Encoders
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              Separate encoders extract latent representations from each data type
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, mt: 2 }}>
              <Chip label="MLP" sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} />
              <Chip label="Transformer" sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} />
              <Chip label="CNN" sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} />
            </Box>
          </Paper>
        </motion.div>

        {/* Arrow */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
          <Typography sx={{ fontSize: '2rem', color: 'grey.400' }}>↓</Typography>
        </Box>

        {/* Cross-Modal Attention */}
        <motion.div variants={itemVariants}>
          <Paper
            elevation={4}
            sx={{
              p: 3,
              mb: 3,
              mx: 'auto',
              maxWidth: 700,
              textAlign: 'center',
              background: 'linear-gradient(135deg, #38a169 0%, #2f855a 100%)',
              color: 'white',
              borderRadius: 3,
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 0%, transparent 70%)',
              }}
            />
            <Typography variant="h5" fontWeight={700} sx={{ mb: 1, position: 'relative' }}>
              ⚡ Cross-Modal Attention Fusion
            </Typography>
            <Typography variant="body1" sx={{ opacity: 0.95, position: 'relative', mb: 2 }}>
              Self-attention mechanism learns optimal weights for combining modalities
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, flexWrap: 'wrap', position: 'relative' }}>
              <Chip 
                label="Multi-Head Attention" 
                sx={{ bgcolor: 'rgba(255,255,255,0.25)', color: 'white', fontWeight: 600 }} 
              />
              <Chip 
                label="Learnable Weights" 
                sx={{ bgcolor: 'rgba(255,255,255,0.25)', color: 'white', fontWeight: 600 }} 
              />
              <Chip 
                label="Missing Modality Handling" 
                sx={{ bgcolor: 'rgba(255,255,255,0.25)', color: 'white', fontWeight: 600 }} 
              />
            </Box>
          </Paper>
        </motion.div>

        {/* Arrow */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
          <Typography sx={{ fontSize: '2rem', color: 'grey.400' }}>↓</Typography>
        </Box>

        {/* Survival Prediction Head */}
        <motion.div variants={itemVariants}>
          <Paper
            elevation={4}
            sx={{
              p: 3,
              mb: 3,
              mx: 'auto',
              maxWidth: 500,
              textAlign: 'center',
              background: 'linear-gradient(135deg, #805ad5 0%, #6b46c1 100%)',
              color: 'white',
              borderRadius: 3,
            }}
          >
            <Typography variant="h6" fontWeight={600} sx={{ mb: 1 }}>
              📈 Survival Prediction Head
            </Typography>
            <Typography variant="body2" sx={{ opacity: 0.9, mb: 2 }}>
              Cox Proportional Hazards model with neural network
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
              <Chip label="Risk Score" sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} />
              <Chip label="Survival Curves" sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white' }} />
            </Box>
          </Paper>
        </motion.div>

        {/* Arrow */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
          <Typography sx={{ fontSize: '2rem', color: 'grey.400' }}>↓</Typography>
        </Box>

        {/* Output */}
        <motion.div variants={itemVariants}>
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 3, flexWrap: 'wrap' }}>
            <Paper
              elevation={3}
              sx={{
                p: 2.5,
                minWidth: 180,
                textAlign: 'center',
                borderRadius: 2,
                border: '2px solid #38a169',
              }}
            >
              <Typography variant="h3" sx={{ mb: 1 }}>📊</Typography>
              <Typography variant="subtitle1" fontWeight={600} color="secondary">
                Survival Probability
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Time-to-event predictions
              </Typography>
            </Paper>

            <Paper
              elevation={3}
              sx={{
                p: 2.5,
                minWidth: 180,
                textAlign: 'center',
                borderRadius: 2,
                border: '2px solid #805ad5',
              }}
            >
              <Typography variant="h3" sx={{ mb: 1 }}>🎯</Typography>
              <Typography variant="subtitle1" fontWeight={600} sx={{ color: '#805ad5' }}>
                Risk Stratification
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Low / Medium / High risk
              </Typography>
            </Paper>

            <Paper
              elevation={3}
              sx={{
                p: 2.5,
                minWidth: 180,
                textAlign: 'center',
                borderRadius: 2,
                border: '2px solid #dd6b20',
              }}
            >
              <Typography variant="h3" sx={{ mb: 1 }}>🔍</Typography>
              <Typography variant="subtitle1" fontWeight={600} sx={{ color: '#dd6b20' }}>
                Attention Maps
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Interpretable insights
              </Typography>
            </Paper>
          </Box>
        </motion.div>
      </motion.div>
    </Box>
  );
}
