import React, {useRef, useState} from 'react';
import * as tf from "@tensorflow/tfjs";
import * as facemesh from "@tensorflow-models/facemesh";
import Webcam from "react-webcam";
import './App.css';
import { drawMesh, calculateHeadPose, calculateRelativePose, displayPoseInfo } from './utilities';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrationPose, setCalibrationPose] = useState(null);

  const handleCalibration = async () => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const net = await facemesh.load({
        inputResolution: {width:640, height:480},
        scale:0.8
      });

      const face = await net.estimateFaces(video);

      if (face.length > 0) {
        const landmarks = face[0].scaledMesh;
        const pose = calculateHeadPose(landmarks);
        setCalibrationPose(pose);
        setIsCalibrated(true);
        console.log("Calibration completed with pose:", pose);
      } else {
        alert("No face detected! Please make sure your face is visible.");
      }
    }
  };

  const runFacemesh = async () => {
    const net = await facemesh.load({
      inputResolution: {width:640, height:480}, 
      scale:0.8
    });
    setInterval(() => {
      detect(net)
    }, 100)
  };

  const detect =  async (net) => {
    if (
      typeof webcamRef.current !== "undefined" && 
      webcamRef.current !== null && 
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const face = await net.estimateFaces(video);
      // console.log(face);

      const ctx = canvasRef.current.getContext("2d");
      drawMesh(face, ctx);

      if (isCalibrated && face.length > 0) {
        const landmarks = face[0].scaledMesh;
        const currentPose = calculateHeadPose(landmarks);
        const relativePose = calculateRelativePose(calibrationPose, currentPose);

        displayPoseInfo(ctx, relativePose);
      }
    }
  };

  runFacemesh();

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position:"absolute",
            marginLeft:"auto",
            marginRight:"auto",
            left:0,
            right:0,
            textAlign:"center",
            zIndex:9,
            width:640,
            height:480
          }}
        />

        <canvas 
          ref={canvasRef}
          style={{
            position:"absolute",
            marginLeft:"auto",
            marginRight:"auto",
            left:0,
            right:0,
            textAlign:"center",
            zIndex:9,
            width:640,
            height:480
          }}
        />

        <button
          onClick={handleCalibration}
          disabled={isCalibrated}
          style={{
            position:"absolute",
            top:20,
            left:20,
            zIndex:10,
            padding:"10px 20px",
            backgroundColor:isCalibrated ? "green" : "blue",
            color:"white",
            border:"none",
            borderRadius:"5px",
            cursor:isCalibrated ? "default" : "pointer"
          }}
        >
          {isCalibrated ? "Calibrated" : "I confirm my head stays straight"}
        </button>
      </header>
    </div>
  );
}

export default App;
