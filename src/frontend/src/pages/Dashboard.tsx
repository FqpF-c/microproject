import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  Security as SecurityIcon,
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
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
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
} from 'recharts';

import { apiService } from '../services/api';

const Dashboard: React.FC = () => {
  const { data: healthData, isLoading: healthLoading } = useQuery(
    'health',
    apiService.getHealth,
    { refetchInterval: 30000 }
  );

  const { data: statsData, isLoading: statsLoading } = useQuery(
    'stats',
    apiService.getStats,
    { refetchInterval: 60000 }
  );

  const { data: modelData, isLoading: modelLoading } = useQuery(
    'model-info',
    apiService.getModelInfo
  );

  // Mock data for charts
  const riskDistributionData = [
    { name: 'Low Risk', value: 65, color: '#4caf50' },
    { name: 'Medium Risk', value: 25, color: '#ff9800' },
    { name: 'High Risk', value: 10, color: '#f44336' },
  ];

  const analysisHistoryData = [
    { day: 'Mon', analyses: 24, threats: 3 },
    { day: 'Tue', analyses: 31, threats: 5 },
    { day: 'Wed', analyses: 28, threats: 2 },
    { day: 'Thu', analyses: 35, threats: 7 },
    { day: 'Fri', analyses: 42, threats: 4 },
    { day: 'Sat', analyses: 19, threats: 1 },
    { day: 'Sun', analyses: 15, threats: 2 },
  ];

  const performanceMetrics = [
    { metric: 'Accuracy', value: 96.8, color: '#4caf50' },
    { metric: 'Precision', value: 94.2, color: '#2196f3' },
    { metric: 'Recall', value: 95.7, color: '#ff9800' },
    { metric: 'F1-Score', value: 94.9, color: '#9c27b0' },
  ];

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    icon: React.ReactNode;
    color: string;
    subtitle?: string;
  }> = ({ title, value, icon, color, subtitle }) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="textSecondary" gutterBottom variant="h6">
              {title}
            </Typography>
            <Typography variant="h4" component="div" sx={{ color }}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant="body2" color="textSecondary">
                {subtitle}
              </Typography>
            )}
          </Box>
          <Box sx={{ color, fontSize: 40 }}>{icon}</Box>
        </Box>
      </CardContent>
    </Card>
  );

  if (healthLoading || statsLoading || modelLoading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="50vh"
      >
        <CircularProgress size={60} />
      </Box>
    );
  }

  return (
    <Box>
      {/* System Status */}
      <Box mb={3}>
        <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
          System Dashboard
        </Typography>
        {healthData ? (
          <Alert
            severity={healthData.components.model_loaded ? 'success' : 'warning'}
            icon={
              healthData.components.model_loaded ? (
                <CheckCircleIcon />
              ) : (
                <WarningIcon />
              )
            }
          >
            System Status: {healthData.status} •{' '}
            {healthData.components.model_loaded
              ? 'All components operational'
              : 'Model not loaded - some features unavailable'}
          </Alert>
        ) : (
          <Alert severity="error" icon={<ErrorIcon />}>
            Unable to connect to backend service
          </Alert>
        )}
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Analyses"
            value={statsData?.uploaded_files || 0}
            icon={<SecurityIcon />}
            color="#1976d2"
            subtitle="APK files analyzed"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Threats Detected"
            value="24"
            icon={<ErrorIcon />}
            color="#f44336"
            subtitle="High-risk applications"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Model Accuracy"
            value="96.8%"
            icon={<TrendingUpIcon />}
            color="#4caf50"
            subtitle="Classification accuracy"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Avg Response Time"
            value="2.4s"
            icon={<SpeedIcon />}
            color="#ff9800"
            subtitle="Per analysis"
          />
        </Grid>
      </Grid>

      {/* Charts Row 1 */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Analysis History (Last 7 Days)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analysisHistoryData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="analyses"
                    stroke="#1976d2"
                    strokeWidth={3}
                    name="Total Analyses"
                  />
                  <Line
                    type="monotone"
                    dataKey="threats"
                    stroke="#f44336"
                    strokeWidth={3}
                    name="Threats Found"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Risk Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={riskDistributionData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) =>
                      `${name} ${(percent * 100).toFixed(0)}%`
                    }
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {riskDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Model Performance */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                Model Performance Metrics
              </Typography>
              {performanceMetrics.map((metric) => (
                <Box key={metric.metric} mb={2}>
                  <Box
                    display="flex"
                    justifyContent="space-between"
                    alignItems="center"
                    mb={1}
                  >
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
                        bgcolor: metric.color,
                      },
                    }}
                  />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" component="h2" gutterBottom>
                System Information
              </Typography>
              <Box>
                <Box display="flex" justifyContent="space-between" py={1}>
                  <Typography>Model Type:</Typography>
                  <Chip
                    label="Channel Attention LSTM"
                    color="primary"
                    size="small"
                  />
                </Box>
                <Box display="flex" justifyContent="space-between" py={1}>
                  <Typography>Parameters:</Typography>
                  <Typography>
                    {modelData?.total_parameters?.toLocaleString() || 'N/A'}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between" py={1}>
                  <Typography>Device:</Typography>
                  <Chip
                    label={modelData?.device || 'Unknown'}
                    color="secondary"
                    size="small"
                  />
                </Box>
                <Box display="flex" justifyContent="space-between" py={1}>
                  <Typography>PyTorch Version:</Typography>
                  <Typography>
                    {statsData?.system_info?.torch_version || 'N/A'}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between" py={1}>
                  <Typography>CUDA Available:</Typography>
                  <Chip
                    label={
                      statsData?.system_info?.device_available ? 'Yes' : 'No'
                    }
                    color={
                      statsData?.system_info?.device_available
                        ? 'success'
                        : 'default'
                    }
                    size="small"
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;