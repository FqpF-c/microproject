import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Alert,
  CircularProgress,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  ExpandMore as ExpandMoreIcon,
  Security as SecurityIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useMutation } from 'react-query';
import { useSnackbar } from 'notistack';

import { apiService, AnalysisResponse } from '../services/api';

const Analysis: React.FC = () => {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const { enqueueSnackbar } = useSnackbar();

  const uploadMutation = useMutation(apiService.uploadAPK, {
    onSuccess: (data) => {
      enqueueSnackbar('APK uploaded successfully', { variant: 'success' });
      // Automatically analyze after upload
      analyzeMutation.mutate({ filepath: data.filepath });
    },
    onError: (error: any) => {
      enqueueSnackbar(error.response?.data?.error || 'Upload failed', {
        variant: 'error',
      });
    },
  });

  const analyzeMutation = useMutation(apiService.analyzeAPK, {
    onSuccess: (data) => {
      setAnalysisResult(data);
      enqueueSnackbar('Analysis completed', { variant: 'success' });
    },
    onError: (error: any) => {
      enqueueSnackbar(error.response?.data?.error || 'Analysis failed', {
        variant: 'error',
      });
    },
  });

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'application/vnd.android.package-archive': ['.apk'] },
    multiple: false,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const formData = new FormData();
        formData.append('file', acceptedFiles[0]);
        uploadMutation.mutate(formData);
      }
    },
  });

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Low':
        return '#4caf50';
      case 'Medium':
        return '#ff9800';
      case 'High':
        return '#f44336';
      default:
        return '#757575';
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Low':
        return <CheckCircleIcon sx={{ color: '#4caf50' }} />;
      case 'Medium':
        return <WarningIcon sx={{ color: '#ff9800' }} />;
      case 'High':
        return <ErrorIcon sx={{ color: '#f44336' }} />;
      default:
        return <InfoIcon />;
    }
  };

  const isLoading = uploadMutation.isLoading || analyzeMutation.isLoading;

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
        APK Malware Analysis
      </Typography>

      {/* Upload Section */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Upload APK File
          </Typography>
          <Box
            {...getRootProps()}
            sx={{
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.500',
              borderRadius: 2,
              p: 4,
              textAlign: 'center',
              cursor: 'pointer',
              bgcolor: isDragActive ? 'action.hover' : 'transparent',
              transition: 'all 0.3s ease',
            }}
          >
            <input {...getInputProps()} />
            <UploadIcon sx={{ fontSize: 48, color: 'grey.500', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive
                ? 'Drop the APK file here...'
                : 'Drag & drop an APK file here, or click to select'}
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Maximum file size: 100MB
            </Typography>
            {isLoading && (
              <Box mt={2}>
                <CircularProgress size={30} />
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {uploadMutation.isLoading ? 'Uploading...' : 'Analyzing...'}
                </Typography>
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {analysisResult && (
        <Grid container spacing={3}>
          {/* Risk Assessment */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Risk Assessment
                </Typography>
                {analysisResult.prediction ? (
                  <Box>
                    <Box display="flex" alignItems="center" mb={2}>
                      {getRiskIcon(analysisResult.prediction.risk_level)}
                      <Typography
                        variant="h4"
                        sx={{
                          ml: 1,
                          color: getRiskColor(analysisResult.prediction.risk_level),
                          fontWeight: 'bold',
                        }}
                      >
                        {analysisResult.prediction.risk_level} Risk
                      </Typography>
                    </Box>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      Confidence: {(analysisResult.prediction.confidence * 100).toFixed(1)}%
                    </Typography>

                    {/* Probability Distribution */}
                    <Typography variant="subtitle1" gutterBottom>
                      Risk Probabilities:
                    </Typography>
                    {Object.entries(analysisResult.prediction.probabilities).map(
                      ([risk, prob]) => (
                        <Box key={risk} sx={{ mb: 1 }}>
                          <Box
                            display="flex"
                            justifyContent="space-between"
                            alignItems="center"
                          >
                            <Typography variant="body2">{risk}</Typography>
                            <Typography variant="body2">
                              {(prob * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={prob * 100}
                            sx={{
                              height: 6,
                              borderRadius: 3,
                              '& .MuiLinearProgress-bar': {
                                bgcolor: getRiskColor(risk),
                              },
                            }}
                          />
                        </Box>
                      )
                    )}
                  </Box>
                ) : (
                  <Alert severity="info">No prediction available</Alert>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* File Information */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  File Information
                </Typography>
                <List dense>
                  <ListItem>
                    <ListItemText
                      primary="Filename"
                      secondary={analysisResult.filename}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Features Extracted"
                      secondary={analysisResult.feature_count}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText
                      primary="Analysis Time"
                      secondary={new Date(analysisResult.timestamp).toLocaleString()}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Explanation */}
          {analysisResult.prediction?.explanation && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    AI Analysis Explanation
                  </Typography>
                  <Alert
                    severity={
                      analysisResult.prediction.risk_level === 'High'
                        ? 'error'
                        : analysisResult.prediction.risk_level === 'Medium'
                        ? 'warning'
                        : 'success'
                    }
                    sx={{ mb: 3 }}
                  >
                    {analysisResult.prediction.explanation.explanations.primary}
                  </Alert>

                  {/* Top Features */}
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">
                        Top Contributing Features
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List>
                        {analysisResult.prediction.explanation.top_features
                          .slice(0, 5)
                          .map((feature, index) => (
                            <ListItem key={index}>
                              <ListItemIcon>
                                <Chip
                                  label={`#${index + 1}`}
                                  size="small"
                                  color="primary"
                                />
                              </ListItemIcon>
                              <ListItemText
                                primary={feature.description}
                                secondary={`Value: ${feature.value} | Importance: ${(
                                  feature.importance * 100
                                ).toFixed(1)}%`}
                              />
                            </ListItem>
                          ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>

                  {/* Risk Factors */}
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">Risk Factors</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      {Object.entries(
                        analysisResult.prediction.explanation.risk_factors
                      ).map(([category, factors]) => (
                        <Box key={category} sx={{ mb: 2 }}>
                          {factors.length > 0 && (
                            <>
                              <Typography
                                variant="subtitle2"
                                sx={{ textTransform: 'capitalize', mb: 1 }}
                              >
                                {category}:
                              </Typography>
                              <List dense>
                                {factors.map((factor, index) => (
                                  <ListItem key={index} sx={{ py: 0.5 }}>
                                    <ListItemIcon>
                                      <SecurityIcon fontSize="small" />
                                    </ListItemIcon>
                                    <ListItemText primary={factor} />
                                  </ListItem>
                                ))}
                              </List>
                              {category !== 'certificate' && <Divider />}
                            </>
                          )}
                        </Box>
                      ))}
                    </AccordionDetails>
                  </Accordion>

                  {/* Recommendations */}
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle1">Recommendations</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List>
                        {analysisResult.prediction.explanation.recommendations.map(
                          (recommendation, index) => (
                            <ListItem key={index}>
                              <ListItemText primary={recommendation} />
                            </ListItem>
                          )
                        )}
                      </List>
                    </AccordionDetails>
                  </Accordion>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}
    </Box>
  );
};

export default Analysis;