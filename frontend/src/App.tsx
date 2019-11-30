import React from 'react';
import { ThemeProvider } from '@material-ui/core/styles';
import theme from './theme';
import { Button, CssBaseline } from '@material-ui/core';

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
        <CssBaseline />
        <div>
            Here is some plain text
        <Button>Hi</Button>
            </div>
    </ThemeProvider>
  );
}

export default App;
