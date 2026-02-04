import { Box, Typography, Paper, Chip, LinearProgress, Divider, Grid } from '@mui/material';
import { motion } from 'framer-motion';
import { PredictionResult } from '../types';

const MotionPaper = motion(Paper);

interface ResultsSummaryProps {
  prediction: PredictionResult;
}

const riskColors = {
  low: { bg: '#c6f6d5', text: '#22543d', border: '#38a169' },
  medium: { bg: '#fef3c7', text: '#92400e', border: '#f59e0b' },
  high: { bg: '#fed7d7', text: '#742a2a', border: '#e53e3e' },
};

export default function ResultsSummary({ prediction }: ResultsSummaryProps) {
  const riskGroup = prediction.risk_group || 'medium';
  const colors = riskColors[riskGroup as keyof typeof riskColors] || riskColors.medium;
  
  // Calculate survival probabilities at key timepoints
  const survivalProbs = prediction.survival_probabilities;
  const timepoints = [
    { time: '1 Year', key: '365' },
    { time: '2 Years', key: '730' },
    { time: '3 Years', key: '1095' },
    { time: '5 Years', key: '1825' },
  ];

  return (
    <Box sx={{ p: 2 }}>
      <Grid container spacing={3}>
        {/* Risk Score Card */}
        <Grid item xs={12} md={4}>
          <MotionPaper
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            elevation={3}
            sx={{
              p: 3,
              textAlign: 'center',
              borderRadius: 3,
              height: '100%',
              background: `linear-gradient(135deg, ${colors.bg} 0%, white 100%)`,
              border: `2px solid ${colors.border}`,
            }}
          >
            <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: 2 }}>
              Overall Risk Score
            </Typography>
            <Box sx={{ my: 2 }}>
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.3, type: 'spring', stiffness: 200 }}
              >
                <Typography
                  variant="h2"
                  fontWeight={700}
                  sx={{ color: colors.border }}
                >
                  {prediction.risk_score.toFixed(3)}
                </Typography>
              </motion.div>
            </Box>
            <Chip
              label={`${riskGroup.toUpperCase()} RISK`}
              sx={{
                backgroundColor: colors.border,
                color: 'white',
                fontWeight: 700,
                fontSize: '1rem',
                px: 2,
                py: 2.5,
              }}
            />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Based on {prediction.modalities_used?.length || 0} modalities
            </Typography>
          </MotionPaper>
        </Grid>

        {/* Survival Probabilities */}
        <Grid item xs={12} md={4}>
          <MotionPaper
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            elevation={3}
            sx={{
              p: 3,
              borderRadius: 3,
              height: '100%',
            }}
          >
            <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
              📊 Survival Probabilities
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {timepoints.map((tp, index) => {
                const prob = survivalProbs[tp.key] || 0;
                const percentage = prob * 100;
                return (
                  <motion.div
                    key={tp.time}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 + index * 0.1 }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2" fontWeight={500}>
                        {tp.time}
                      </Typography>
                      <Typography variant="body2" fontWeight={600} color="primary">
                        {percentage.toFixed(1)}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={percentage}
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        backgroundColor: 'grey.200',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 4,
                          background: percentage > 70 
                            ? 'linear-gradient(90deg, #38a169, #48bb78)'
                            : percentage > 40
                            ? 'linear-gradient(90deg, #f59e0b, #fbbf24)'
                            : 'linear-gradient(90deg, #e53e3e, #f56565)',
                        },
                      }}
                    />
                  </motion.div>
                );
              })}
            </Box>
          </MotionPaper>
        </Grid>

        {/* Modalities Used */}
        <Grid item xs={12} md={4}>
          <MotionPaper
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            elevation={3}
            sx={{
              p: 3,
              borderRadius: 3,
              height: '100%',
            }}
          >
            <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
              🔬 Data Sources Analyzed
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              {prediction.modalities_used?.map((modality, index) => {
                const modalityInfo: Record<string, { icon: string; color: string }> = {
                  clinical: { icon: '🏥', color: '#38a169' },
                  genomic: { icon: '🧬', color: '#805ad5' },
                  pathology: { icon: '🔬', color: '#dd6b20' },
                  methylation: { icon: '🔷', color: '#3182ce' },
                  transcriptomic: { icon: '📊', color: '#d53f8c' },
                  wsi: { icon: '🖼️', color: '#dd6b20' },
                };
                const info = modalityInfo[modality.toLowerCase()] || { icon: '📁', color: '#718096' };
                const weight = prediction.attention_weights?.[modality.toLowerCase()] || 0;
                
                return (
                  <motion.div
                    key={modality}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.5 + index * 0.1 }}
                  >
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 2,
                        p: 1.5,
                        borderRadius: 2,
                        backgroundColor: `${info.color}10`,
                        border: `1px solid ${info.color}30`,
                      }}
                    >
                      <Typography variant="h5">{info.icon}</Typography>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="body2" fontWeight={600} sx={{ textTransform: 'capitalize' }}>
                          {modality}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Weight: {(weight * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <Box
                        sx={{
                          width: 40,
                          height: 40,
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          backgroundColor: info.color,
                          color: 'white',
                          fontWeight: 700,
                          fontSize: '0.75rem',
                        }}
                      >
                        {(weight * 100).toFixed(0)}%
                      </Box>
                    </Box>
                  </motion.div>
                );
              })}
            </Box>
          </MotionPaper>
        </Grid>

        {/* Confidence Interval */}
        {prediction.confidence_interval && (
          <Grid item xs={12}>
            <MotionPaper
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.6 }}
              elevation={2}
              sx={{
                p: 2,
                borderRadius: 2,
                backgroundColor: 'grey.50',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4, flexWrap: 'wrap' }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    95% Confidence Interval
                  </Typography>
                  <Typography variant="h6" fontWeight={600} color="primary">
                    [{prediction.confidence_interval.lower.toFixed(3)} - {prediction.confidence_interval.upper.toFixed(3)}]
                  </Typography>
                </Box>
                <Divider orientation="vertical" flexItem />
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    Patient ID
                  </Typography>
                  <Typography variant="h6" fontWeight={600}>
                    {prediction.patient_id}
                  </Typography>
                </Box>
                <Divider orientation="vertical" flexItem />
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    Model Confidence
                  </Typography>
                  <Chip 
                    label="High" 
                    size="small" 
                    sx={{ 
                      backgroundColor: '#38a169', 
                      color: 'white',
                      fontWeight: 600,
                    }} 
                  />
                </Box>
              </Box>
            </MotionPaper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
}
