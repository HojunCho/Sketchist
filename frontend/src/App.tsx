import React, { useState, useRef } from 'react';
import { ThemeProvider } from '@material-ui/core/styles';
import theme from './theme';
import Cropper from 'react-cropper';
import { CssBaseline, AppBar, Toolbar, Typography, Button } from '@material-ui/core';
import 'cropperjs/dist/cropper.css';

const api = 'http://localhost:5000/generate';

if (!HTMLCanvasElement.prototype.toBlob) {
  Object.defineProperty(HTMLCanvasElement.prototype, 'toBlob', {
     value: function (callback, type, quality) {
       var canvas = this;
       setTimeout(function() {
         var binStr = atob( canvas.toDataURL(type, quality).split(',')[1] ),
         len = binStr.length,
         arr = new Uint8Array(len);

         for (var i = 0; i < len; i++ ) {
            arr[i] = binStr.charCodeAt(i);
         }

         callback( new Blob( [arr], {type: type || 'image/png'} ) );
       });
     }
  });
}

const UploadImage: React.FC = () => {
  const [imgSrc, setImgSrc] = useState('');
  const [cropped, setCropped] = useState('');
  const [generatedImg, setGeneratedImg] = useState(null);

  const [sentImg, setSentImg] = useState(null);
  const cropper = useRef(null);

  return (
    <div>
      <input
        accept="image/*"
        id="contained-button-file"
        multiple
        type="file"
        style={{ display: 'none' }}
        onChange={(e): void => {
          if (e && e.target && e.target.files) {
            return setImgSrc(URL.createObjectURL(e.target.files[0]));
          }
        }}
      />
      <label htmlFor="contained-button-file">
        <Button variant="contained" color="primary" component="span">
          Upload
        </Button>
      </label>
      <Cropper
        ref={cropper}
        src={imgSrc}
        style={{ height: 400, width: '100%' }}
        aspectRatio={1}
        guides={false}
        crop={(): void => setCropped(cropper.current.getCroppedCanvas().toDataURL())}
      />
      <img alt="cropped sketch" src={cropped} />
      <Button
        variant="contained"
        color="primary"
        component="span"
        onClick={(): void => {
          const canvas = cropper.current.getCroppedCanvas({ width: 256, height: 256 });
          canvas.toBlob(blob => {
            fetch(api, {
              method: 'post',
              headers: { 'Content-Type': 'image/*' },
              body: blob
            })
              .then(response => response.blob())
              .then(img => {
                setGeneratedImg(URL.createObjectURL(img));
              });

            setSentImg(URL.createObjectURL(blob));
          });
        }}
      >
        Submit
      </Button>
      <img alt="sent-img" src={sentImg} />
            <img alt="generated-img" src={generatedImg} />
    </div>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">Sketchist</Typography>
        </Toolbar>
      </AppBar>
      <UploadImage />
    </ThemeProvider>
  );
};

export default App;
