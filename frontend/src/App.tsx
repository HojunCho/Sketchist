import React, { useState, useRef } from "react";
import { ThemeProvider } from "@material-ui/core/styles";
import theme from "./theme";
import Cropper from 'react-cropper';
import { CssBaseline, AppBar, Toolbar, Typography, Button } from "@material-ui/core";
import 'cropperjs/dist/cropper.css'

const api = "http://localhost:5000/generate"

const UploadImage: React.FC = () => {
    const [imgSrc, setImgSrc] = useState("")
    const [cropped, setCropped] = useState("")
    const [generatedImg, setGeneratedImg] = useState()
    const cropper = useRef(null);

    return (
    <div>
        <input
            accept="image/*"
            id="contained-button-file"
            multiple
            type="file"
            style={{display: 'none'}}
            onChange={(e): void => {
                if (e && e.target && e.target.files) {
                    return setImgSrc(URL.createObjectURL(e.target.files[0]))
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
          style={{height:400, width: '100%'}}
          aspectRatio={1}
          guides={false}
          crop={(): void => setCropped(cropper.current.getCroppedCanvas().toDataURL())}
      />
        <img alt="cropped sketch" src={cropped} />
        <Button variant="contained" color="primary" component="span" onClick={(): void => {
            let canvas = cropper.current.getCroppedCanvas()
            canvas.width = 256
            canvas.height = 256
            canvas.toBlob(blob => fetch(api, {
                method: 'post',
                headers: {'Content-Type': 'image/*'},
                body: blob,
            }).then(response => response.blob()).then(img => {
                setGeneratedImg(URL.createObjectURL(img))}))}}>
            Submit
        </Button>
        <img src={generatedImg} />
    </div>
    )
}

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
        <AppBar position="static">
            <Toolbar>
                <Typography variant="h6" >
                    Sketchist
                </Typography>
            </Toolbar>
        </AppBar>
        <UploadImage />
    </ThemeProvider>
  );
};

export default App;
