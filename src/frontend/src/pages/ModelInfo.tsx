import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  CircularProgress,
  LinearProgress,
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Architecture as ArchitectureIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
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
} from 'recharts';

import { apiService } from '../services/api';

const ModelInfo: React.FC = () => {
  const { data: modelData, isLoading: modelLoading, error: modelError } = useQuery(
    'model-info',
    apiService.getModelInfo
  );

  const { data: configData, isLoading: configLoading } = useQuery(
    'config',
    apiService.getConfig
  );

  // Mock performance data
  const performanceMetrics = [
    { metric: 'Accuracy', value: 96.8, target: 95.0 },
    { metric: 'Precision', value: 94.2, target: 90.0 },
    { metric: 'Recall', value: 95.7, target: 92.0 },
    { metric: 'F1-Score', value: 94.9, target: 91.0 },
    { metric: 'AUC-ROC', value: 97.3, target: 93.0 },
    { metric: 'Specificity', value: 96.1, target: 94.0 },
  ];

  const modelComparison = [
    { model: 'CA-LSTM (Ours)', accuracy: 96.8, f1: 94.9, speed: 2.4 },
    { model: 'Droidetec BiLSTM', accuracy: 94.2, f1: 92.1, speed: 3.1 },
    { model: 'Traditional ML', accuracy: 89.5, f1: 87.8, speed: 1.2 },
    { model: 'CNN-based', accuracy: 91.3, f1: 89.6, speed: 2.8 },
  ];

  const architectureDetails = [
    { component: 'Input Layer', params: 'Input Dimension: 150', description: 'Feature vector input' },
    { component: 'Channel Attention', params: 'Reduction Ratio: 8', description: 'Feature selection mechanism' },
    { component: 'BiLSTM Layers', params: '3 layers: [128, 64, 32]', description: 'Bidirectional LSTM processing' },
    { component: 'Self-Attention', params: '8 heads', description: 'Multi-head attention mechanism' },
    { component: 'Feature Fusion', params: 'Hidden: 64 → 32', description: 'Feature combination layer' },
    { component: 'Classification', params: 'Output: 3 classes', description: 'Risk level classification' },
  ];

  if (modelLoading || configLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (modelError) {
    return (
      <Box>
        <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
          Model Information
        </Typography>
        <Alert severity="error">
          Unable to load model information. Model may not be loaded.
        </Alert>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
        Model Information
      </Typography>

      {/* Model Overview */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <ArchitectureIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Model Type
              </Typography>
              <Chip
                label={modelData?.model_type || 'Unknown'}
                color="primary"
                size="medium"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <MemoryIcon sx={{ fontSize: 40, color: 'secondary.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Parameters
              </Typography>
              <Typography variant="h5" fontWeight="bold">
                {modelData?.total_parameters?.toLocaleString() || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <SpeedIcon sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Device
              </Typography>
              <Chip
                label={modelData?.device || 'Unknown'}
                color={modelData?.device?.includes('cuda') ? 'success' : 'default'}
                size="medium"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TrendingUpIcon sx={{ fontSize: 40, color: 'warning.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Accuracy
              </Typography>
              <Typography variant="h5" fontWeight="bold" color="success.main">
                96.8%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Architecture Details */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Architecture
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Component</TableCell>
                      <TableCell>Parameters</TableCell>
                      <TableCell>Description</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {architectureDetails.map((detail, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <Typography variant="body2" fontWeight="bold">
                            {detail.component}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="textSecondary">
                            {detail.params}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {detail.description}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Metrics
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={performanceMetrics}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <PolarRadiusAxis domain={[80, 100]} />
                  <Radar
                    name="Current"
                    dataKey="value"
                    stroke="#1976d2"
                    fill="#1976d2"
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                  <Radar
                    name="Target"
                    dataKey="target"
                    stroke="#ff9800"
                    fill="#ff9800"
                    fillOpacity={0.1}
                    strokeWidth={1}
                    strokeDasharray="5 5"
                  />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Performance Comparison */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Comparison
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={modelComparison} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="model" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="accuracy" fill="#1976d2" name="Accuracy (%)" />
                  <Bar dataKey="f1" fill="#4caf50" name="F1-Score (%)" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Detailed Metrics */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Configuration
              </Typography>
              <Box>
                {configData?.model && Object.entries(configData.model).map(([key, value]) => (
                  <Box key={key} display="flex" justifyContent="space-between" py={1}>
                    <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                      {key.replace(/_/g, ' ')}:
                    </Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {Array.isArray(value) ? value.join(', ') : String(value)}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Breakdown
              </Typography>
              {performanceMetrics.slice(0, 4).map((metric) => (
                <Box key={metric.metric} mb={2}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="body1">{metric.metric}</Typography>
                    <Typography variant="body1" fontWeight="bold">
                      {metric.value}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={metric.value}
                    sx={{
                      height: 8,
                      borderRadius: 4,
                      '& .MuiLinearProgress-bar': {
                        bgcolor: metric.value >= metric.target ? '#4caf50' : '#ff9800',
                      },
                    }}
                  />
                  <Typography variant="caption" color="textSecondary">
                    Target: {metric.target}%
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ModelInfo;