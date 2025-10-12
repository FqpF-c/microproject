import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  Switch,
  FormControlLabel,
  Button,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { useSnackbar } from 'notistack';

import { apiService } from '../services/api';

interface TextSetting {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type: 'text' | 'password' | 'email' | 'number';
  helperText?: string;
}

interface SwitchSetting {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  type: 'switch';
  helperText?: string;
}

type Setting = TextSetting | SwitchSetting;

interface SettingSection {
  title: string;
  description: string;
  settings: Setting[];
}

const Settings: React.FC = () => {
  const [geminiApiKey, setGeminiApiKey] = useState('');
  const [explainabilityEnabled, setExplainabilityEnabled] = useState(true);
  const [autoAnalysis, setAutoAnalysis] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const { enqueueSnackbar } = useSnackbar();

  const { data: configData, isLoading: configLoading, refetch: refetchConfig } = useQuery(
    'config',
    apiService.getConfig
  );

  const { data: healthData, refetch: refetchHealth } = useQuery(
    'health',
    apiService.getHealth
  );

  const handleSaveSettings = () => {
    // In a real implementation, this would call an API to update settings
    enqueueSnackbar('Settings saved successfully', { variant: 'success' });
  };

  const handleRefreshConfig = () => {
    refetchConfig();
    refetchHealth();
    enqueueSnackbar('Configuration refreshed', { variant: 'info' });
  };

  const settingSections: SettingSection[] = [
    {
      title: 'API Configuration',
      description: 'Configure external API keys and endpoints',
      settings: [
        {
          label: 'Gemini API Key',
          value: geminiApiKey,
          onChange: setGeminiApiKey,
          type: 'password',
          helperText: 'API key for AI explanations (keep secure)',
        },
      ],
    },
    {
      title: 'Analysis Settings',
      description: 'Configure analysis behavior and features',
      settings: [
        {
          label: 'Enable AI Explanations',
          value: explainabilityEnabled,
          onChange: setExplainabilityEnabled,
          type: 'switch',
          helperText: 'Generate detailed explanations using AI',
        },
        {
          label: 'Auto-analyze uploads',
          value: autoAnalysis,
          onChange: setAutoAnalysis,
          type: 'switch',
          helperText: 'Automatically start analysis after APK upload',
        },
      ],
    },
    {
      title: 'User Interface',
      description: 'Customize dashboard appearance and notifications',
      settings: [
        {
          label: 'Enable Notifications',
          value: notifications,
          onChange: setNotifications,
          type: 'switch',
          helperText: 'Show success/error notifications',
        },
      ],
    },
  ];

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
        Settings
      </Typography>

      {/* System Status */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Box display="flex" justifyContent="between" alignItems="center" mb={2}>
            <Typography variant="h6">System Status</Typography>
            <Button
              startIcon={<RefreshIcon />}
              onClick={handleRefreshConfig}
              variant="outlined"
              size="small"
            >
              Refresh
            </Button>
          </Box>

          {healthData && (
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography variant="body2">Model:</Typography>
                  <Chip
                    label={healthData.components.model_loaded ? 'Loaded' : 'Not Loaded'}
                    color={healthData.components.model_loaded ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography variant="body2">Feature Extractor:</Typography>
                  <Chip
                    label={healthData.components.feature_extractor ? 'Ready' : 'Error'}
                    color={healthData.components.feature_extractor ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography variant="body2">Preprocessor:</Typography>
                  <Chip
                    label={healthData.components.preprocessor ? 'Ready' : 'Error'}
                    color={healthData.components.preprocessor ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Box display="flex" alignItems="center" gap={1}>
                  <Typography variant="body2">Explainer:</Typography>
                  <Chip
                    label={healthData.components.explainer ? 'Ready' : 'Error'}
                    color={healthData.components.explainer ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
              </Grid>
            </Grid>
          )}
        </CardContent>
      </Card>

      {/* Configuration Settings */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Application Settings
              </Typography>

              {settingSections.map((section, sectionIndex) => (
                <Accordion key={sectionIndex} sx={{ mb: 1 }}>
                  <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    sx={{ bgcolor: 'action.hover' }}
                  >
                    <Box>
                      <Typography variant="subtitle1" fontWeight="bold">
                        {section.title}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        {section.description}
                      </Typography>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    {section.settings.map((setting, settingIndex) => (
                      <Box key={settingIndex} mb={2}>
                        {setting.type === 'switch' ? (
                          <FormControlLabel
                            control={
                              <Switch
                                checked={(setting as SwitchSetting).value}
                                onChange={(e) => (setting as SwitchSetting).onChange(e.target.checked)}
                              />
                            }
                            label={setting.label}
                          />
                        ) : (
                          <TextField
                            fullWidth
                            label={setting.label}
                            type={setting.type}
                            value={(setting as TextSetting).value}
                            onChange={(e) => (setting as TextSetting).onChange(e.target.value)}
                            helperText={setting.helperText}
                            variant="outlined"
                            size="small"
                          />
                        )}
                        {setting.helperText && setting.type !== 'password' && (
                          <Typography variant="caption" color="textSecondary" display="block" mt={0.5}>
                            {setting.helperText}
                          </Typography>
                        )}
                      </Box>
                    ))}
                  </AccordionDetails>
                </Accordion>
              ))}

              <Box mt={3}>
                <Button
                  variant="contained"
                  startIcon={<SaveIcon />}
                  onClick={handleSaveSettings}
                  size="large"
                >
                  Save Settings
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Configuration
              </Typography>

              {configLoading ? (
                <Alert severity="info">Loading configuration...</Alert>
              ) : configData ? (
                <Box>
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle2">Feature Extraction</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List dense>
                        {Object.entries(configData.feature_extraction || {}).map(([key, value]) => (
                          <ListItem key={key}>
                            <ListItemText
                              primary={key.replace(/_/g, ' ')}
                              secondary={String(value)}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>

                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle2">Model Configuration</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List dense>
                        {Object.entries(configData.model || {}).map(([key, value]) => (
                          <ListItem key={key}>
                            <ListItemText
                              primary={key.replace(/_/g, ' ')}
                              secondary={Array.isArray(value) ? value.join(', ') : String(value)}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>

                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Typography variant="subtitle2">API Configuration</Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List dense>
                        {Object.entries(configData.api || {}).map(([key, value]) => (
                          <ListItem key={key}>
                            <ListItemText
                              primary={key.replace(/_/g, ' ')}
                              secondary={String(value)}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>
                </Box>
              ) : (
                <Alert severity="error">Failed to load configuration</Alert>
              )}
            </CardContent>
          </Card>

          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Button
                  variant="outlined"
                  startIcon={<RefreshIcon />}
                  onClick={() => window.location.reload()}
                  fullWidth
                >
                  Reload Application
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<SettingsIcon />}
                  onClick={() => {
                    enqueueSnackbar('Reset to defaults would be implemented here', {
                      variant: 'info',
                    });
                  }}
                  fullWidth
                >
                  Reset to Defaults
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;