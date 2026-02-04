import { createTheme, alpha } from '@mui/material/styles';

// MOSAIC color palette - inspired by medical/scientific aesthetics
const palette = {
  primary: {
    main: '#1a365d', // Deep blue
    light: '#2c5282',
    dark: '#0d1f3c',
  },
  secondary: {
    main: '#38a169', // Medical green
    light: '#48bb78',
    dark: '#276749',
  },
  accent: {
    purple: '#805ad5',
    orange: '#ed8936',
    pink: '#d53f8c',
    teal: '#319795',
    cyan: '#00b5d8',
  },
  background: {
    default: '#f7fafc',
    paper: '#ffffff',
    dark: '#0a192f',
  },
  text: {
    primary: '#1a202c',
    secondary: '#4a5568',
    light: '#a0aec0',
  },
};

// Gradient definitions
export const gradients = {
  primary: 'linear-gradient(135deg, #1a365d 0%, #2c5282 100%)',
  secondary: 'linear-gradient(135deg, #38a169 0%, #319795 100%)',
  hero: 'linear-gradient(135deg, #0a192f 0%, #1a365d 50%, #2c5282 100%)',
  card: 'linear-gradient(180deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%)',
  glow: 'radial-gradient(circle at 50% 50%, rgba(56, 161, 105, 0.15) 0%, transparent 70%)',
  mesh: `
    radial-gradient(at 40% 20%, rgba(56, 161, 105, 0.1) 0px, transparent 50%),
    radial-gradient(at 80% 0%, rgba(45, 82, 130, 0.15) 0px, transparent 50%),
    radial-gradient(at 0% 50%, rgba(128, 90, 213, 0.1) 0px, transparent 50%),
    radial-gradient(at 80% 50%, rgba(237, 137, 54, 0.1) 0px, transparent 50%),
    radial-gradient(at 0% 100%, rgba(49, 151, 149, 0.15) 0px, transparent 50%)
  `,
};

// Glassmorphism styles
export const glass = {
  light: {
    background: 'rgba(255, 255, 255, 0.8)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.3)',
  },
  dark: {
    background: 'rgba(10, 25, 47, 0.85)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
};

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: palette.primary,
    secondary: palette.secondary,
    background: palette.background,
    text: palette.text,
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '3.5rem',
      letterSpacing: '-0.02em',
      lineHeight: 1.2,
    },
    h2: {
      fontWeight: 700,
      fontSize: '2.5rem',
      letterSpacing: '-0.01em',
      lineHeight: 1.3,
    },
    h3: {
      fontWeight: 600,
      fontSize: '1.75rem',
      letterSpacing: '-0.01em',
    },
    h4: {
      fontWeight: 600,
      fontSize: '1.5rem',
    },
    h5: {
      fontWeight: 600,
      fontSize: '1.25rem',
    },
    h6: {
      fontWeight: 600,
      fontSize: '1rem',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.7,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0 1px 3px rgba(0,0,0,0.08)',
    '0 4px 6px rgba(0,0,0,0.07)',
    '0 5px 15px rgba(0,0,0,0.08)',
    '0 10px 24px rgba(0,0,0,0.1)',
    '0 15px 35px rgba(0,0,0,0.12)',
    '0 20px 40px rgba(0,0,0,0.15)',
    ...Array(18).fill('0 20px 40px rgba(0,0,0,0.15)'),
  ] as any,
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          padding: '10px 24px',
          fontSize: '0.95rem',
        },
        contained: {
          boxShadow: '0 4px 14px 0 rgba(0,0,0,0.15)',
          '&:hover': {
            boxShadow: '0 6px 20px rgba(0,0,0,0.2)',
          },
        },
        containedPrimary: {
          background: gradients.primary,
          '&:hover': {
            background: gradients.primary,
            filter: 'brightness(1.1)',
          },
        },
        containedSecondary: {
          background: gradients.secondary,
          '&:hover': {
            background: gradients.secondary,
            filter: 'brightness(1.1)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
          border: '1px solid rgba(0,0,0,0.05)',
          transition: 'transform 0.2s ease, box-shadow 0.2s ease',
          '&:hover': {
            transform: 'translateY(-4px)',
            boxShadow: '0 12px 40px rgba(0,0,0,0.12)',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 16,
        },
        elevation1: {
          boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
        },
        elevation2: {
          boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
        },
        elevation3: {
          boxShadow: '0 8px 30px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
        },
        filled: {
          '&.MuiChip-colorSuccess': {
            background: alpha('#38a169', 0.15),
            color: '#276749',
          },
          '&.MuiChip-colorWarning': {
            background: alpha('#ed8936', 0.15),
            color: '#c05621',
          },
          '&.MuiChip-colorError': {
            background: alpha('#e53e3e', 0.15),
            color: '#c53030',
          },
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: gradients.hero,
          boxShadow: '0 4px 30px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          fontSize: '0.95rem',
          minHeight: 56,
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          height: 8,
        },
        bar: {
          borderRadius: 10,
        },
      },
    },
  },
});

export default theme;
