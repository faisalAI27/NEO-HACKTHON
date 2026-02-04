import { useMemo } from 'react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  ReferenceLine,
} from 'recharts';
import { Box, Typography, Paper, Chip } from '@mui/material';

interface SurvivalDataPoint {
  time: number;
  probability: number;
  lower?: number;
  upper?: number;
}

interface SurvivalCurveProps {
  data: SurvivalDataPoint[];
  patientId?: string;
}

// Custom tooltip component
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <Paper sx={{ p: 2, boxShadow: 3, border: '1px solid #e2e8f0' }}>
        <Typography variant="subtitle2" fontWeight={600} gutterBottom>
          {data.time} Months
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              bgcolor: '#38a169',
            }}
          />
          <Typography variant="body2">
            Survival: <strong>{(data.probability * 100).toFixed(1)}%</strong>
          </Typography>
        </Box>
        {data.lower && data.upper && (
          <Typography variant="caption" color="text.secondary">
            95% CI: {(data.lower * 100).toFixed(1)}% - {(data.upper * 100).toFixed(1)}%
          </Typography>
        )}
      </Paper>
    );
  }
  return null;
};

export default function SurvivalCurve({ data, patientId }: SurvivalCurveProps) {
  // Prepare chart data with baseline
  const chartData = useMemo(() => {
    // Add baseline point at time 0
    const withBaseline = [
      { time: 0, probability: 1.0, lower: 1.0, upper: 1.0 },
      ...data,
    ];
    return withBaseline;
  }, [data]);

  // Calculate median survival (time at 50% probability)
  const medianSurvival = useMemo(() => {
    for (let i = 1; i < chartData.length; i++) {
      if (chartData[i].probability <= 0.5) {
        // Linear interpolation
        const t1 = chartData[i - 1].time;
        const t2 = chartData[i].time;
        const p1 = chartData[i - 1].probability;
        const p2 = chartData[i].probability;
        return t1 + ((0.5 - p1) / (p2 - p1)) * (t2 - t1);
      }
    }
    return null; // Median not reached
  }, [chartData]);

  // Get 5-year survival
  const fiveYearSurvival = chartData.find((d) => d.time >= 60)?.probability;

  return (
    <Box sx={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header with stats */}
      <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
        {patientId && (
          <Chip
            label={`Patient: ${patientId}`}
            variant="outlined"
            size="small"
          />
        )}
        {medianSurvival && (
          <Chip
            label={`Median: ${medianSurvival.toFixed(0)} months`}
            color="primary"
            size="small"
            variant="outlined"
          />
        )}
        {fiveYearSurvival && (
          <Chip
            label={`5-Year: ${(fiveYearSurvival * 100).toFixed(0)}%`}
            color="secondary"
            size="small"
          />
        )}
      </Box>

      {/* Chart */}
      <Box sx={{ flexGrow: 1, minHeight: 250 }}>
        <ResponsiveContainer>
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
          >
            <defs>
              <linearGradient id="survivalGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#38a169" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#38a169" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="ciGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#38a169" stopOpacity={0.15} />
                <stop offset="95%" stopColor="#38a169" stopOpacity={0.05} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#e2e8f0"
              vertical={false}
            />

            <XAxis
              dataKey="time"
              type="number"
              domain={[0, 'auto']}
              tick={{ fill: '#718096', fontSize: 11 }}
              tickLine={{ stroke: '#cbd5e0' }}
              axisLine={{ stroke: '#cbd5e0' }}
              label={{
                value: 'Time (Months)',
                position: 'insideBottom',
                offset: -5,
                style: { fill: '#4a5568', fontSize: 12 },
              }}
            />

            <YAxis
              domain={[0, 1]}
              tick={{ fill: '#718096', fontSize: 11 }}
              tickLine={{ stroke: '#cbd5e0' }}
              axisLine={{ stroke: '#cbd5e0' }}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              label={{
                value: 'Survival Probability',
                angle: -90,
                position: 'insideLeft',
                style: { fill: '#4a5568', fontSize: 12, textAnchor: 'middle' },
              }}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Reference lines */}
            <ReferenceLine
              y={0.5}
              stroke="#a0aec0"
              strokeDasharray="4 4"
              label={{
                value: '50%',
                position: 'right',
                fill: '#a0aec0',
                fontSize: 10,
              }}
            />

            {/* Confidence interval area */}
            <Area
              type="stepAfter"
              dataKey="upper"
              stroke="none"
              fill="url(#ciGradient)"
              fillOpacity={1}
            />
            <Area
              type="stepAfter"
              dataKey="lower"
              stroke="none"
              fill="white"
              fillOpacity={1}
            />

            {/* Main survival curve */}
            <Area
              type="stepAfter"
              dataKey="probability"
              stroke="#38a169"
              strokeWidth={3}
              fill="url(#survivalGradient)"
              fillOpacity={1}
              dot={{
                fill: '#38a169',
                stroke: '#fff',
                strokeWidth: 2,
                r: 5,
              }}
              activeDot={{
                fill: '#38a169',
                stroke: '#fff',
                strokeWidth: 2,
                r: 7,
              }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </Box>

      {/* Legend */}
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 3, mt: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 20,
              height: 3,
              bgcolor: '#38a169',
              borderRadius: 1,
            }}
          />
          <Typography variant="caption" color="text.secondary">
            Survival Curve
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 20,
              height: 10,
              bgcolor: 'rgba(56, 161, 105, 0.15)',
              borderRadius: 1,
            }}
          />
          <Typography variant="caption" color="text.secondary">
            95% Confidence Interval
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}
