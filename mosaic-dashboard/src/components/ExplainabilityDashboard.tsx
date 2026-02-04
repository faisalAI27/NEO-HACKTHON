import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import { Box, Grid, Typography, Paper, Chip, LinearProgress, Stack } from '@mui/material';

interface GeneImportance {
  gene: string;
  importance: number;
}

interface ExplainabilityDashboardProps {
  attentionWeights: Record<string, number>;
  geneImportance: GeneImportance[];
  detailed?: boolean;
}

// Color palette for modalities
const MODALITY_COLORS: Record<string, string> = {
  clinical: '#3182ce',
  transcriptomics: '#38a169',
  methylation: '#805ad5',
  mutations: '#ed8936',
  pathology: '#e53e3e',
  wsi: '#e53e3e',
  rna: '#38a169',
};

const MODALITY_ICONS: Record<string, string> = {
  clinical: '🏥',
  transcriptomics: '🧬',
  methylation: '🔬',
  mutations: '⚡',
  pathology: '🔍',
};

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <Paper sx={{ p: 1.5, boxShadow: 3 }}>
        <Typography variant="subtitle2" fontWeight={600}>
          {label}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Importance: {(payload[0].value * 100).toFixed(1)}%
        </Typography>
      </Paper>
    );
  }
  return null;
};

export default function ExplainabilityDashboard({
  attentionWeights,
  geneImportance,
  detailed = false,
}: ExplainabilityDashboardProps) {
  // Process modality data for charts
  const modalityData = useMemo(() => {
    return Object.entries(attentionWeights).map(([modality, weight]) => ({
      modality: modality.charAt(0).toUpperCase() + modality.slice(1),
      key: modality,
      importance: weight,
      color: MODALITY_COLORS[modality] || '#718096',
      icon: MODALITY_ICONS[modality] || '📊',
    }));
  }, [attentionWeights]);

  // Process gene data
  const geneData = useMemo(() => {
    return geneImportance.slice(0, 10).map((g) => ({
      ...g,
      importance: g.importance,
    }));
  }, [geneImportance]);

  if (detailed) {
    return (
      <Box>
        <Grid container spacing={3}>
          {/* Modality Radar Chart */}
          <Grid item xs={12} md={6}>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer>
                <RadarChart data={modalityData}>
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis
                    dataKey="modality"
                    tick={{ fill: '#4a5568', fontSize: 12 }}
                  />
                  <PolarRadiusAxis
                    angle={30}
                    domain={[0, 0.5]}
                    tick={{ fill: '#a0aec0', fontSize: 10 }}
                  />
                  <Radar
                    name="Importance"
                    dataKey="importance"
                    stroke="#38a169"
                    fill="#38a169"
                    fillOpacity={0.5}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </Box>
          </Grid>

          {/* Modality Pie Chart */}
          <Grid item xs={12} md={6}>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={modalityData}
                    dataKey="importance"
                    nameKey="modality"
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    label={({ modality, importance }) =>
                      `${modality}: ${(importance * 100).toFixed(0)}%`
                    }
                    labelLine={false}
                  >
                    {modalityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Grid>

          {/* Detailed Modality Breakdown */}
          <Grid item xs={12}>
            <Stack spacing={2}>
              {modalityData.map((m) => (
                <Box key={m.key}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                    <Typography variant="body2" fontWeight={500}>
                      {m.icon} {m.modality}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {(m.importance * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={m.importance * 100 * 2} // Scale for visibility (max ~50%)
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      bgcolor: '#e2e8f0',
                      '& .MuiLinearProgress-bar': {
                        bgcolor: m.color,
                        borderRadius: 4,
                      },
                    }}
                  />
                </Box>
              ))}
            </Stack>
          </Grid>
        </Grid>
      </Box>
    );
  }

  // Compact view (default)
  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Modality Bar Chart */}
      <Box sx={{ flexGrow: 1, minHeight: 200 }}>
        <ResponsiveContainer>
          <BarChart
            data={modalityData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              type="number"
              domain={[0, 0.4]}
              tick={{ fill: '#718096', fontSize: 11 }}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <YAxis
              type="category"
              dataKey="modality"
              tick={{ fill: '#2d3748', fontSize: 12 }}
              width={75}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {modalityData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Box>

      {/* Quick Stats */}
      <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
        {modalityData.slice(0, 3).map((m) => (
          <Chip
            key={m.key}
            label={`${m.icon} ${(m.importance * 100).toFixed(0)}%`}
            size="small"
            sx={{ bgcolor: `${m.color}20`, color: m.color, fontWeight: 600 }}
          />
        ))}
      </Box>
    </Box>
  );
}
