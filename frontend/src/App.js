import React, {useRef, useState} from 'react';
import * as tf from "@tensorflow/tfjs";
import * as facemesh from "@tensorflow-models/facemesh";
import Webcam from "react-webcam";
import './App.css';
import { drawMesh, calculateHeadPose, calculateRelativePose, displayPoseInfo,
         getEyeLocalFrames, drawEyeFrame, rectifyEyePatch, 
         CANON_H, CANON_W, rectifyEyePatchH
 } from './utilities';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const frameCanvasRef = useRef(null); 
  const leftPatchRef = useRef(null); 
  const rightPatchRef = useRef(null);

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

      const ctx = canvasRef.current.getContext("2d");
      drawMesh(face, ctx);

      if (isCalibrated && face.length > 0) {
        const landmarks = face[0].scaledMesh;
        const grabber = frameCanvasRef.current;
        const gctx = grabber.getContext("2d"); 
        grabber.width = videoWidth; 
        grabber.height = videoHeight; 
        gctx.drawImage(video, 0, 0, videoWidth, videoHeight);

        const { left: leftEyeFrame, right: rightEyeFrame } = getEyeLocalFrames(landmarks);

        drawEyeFrame(ctx, leftEyeFrame, "yellow", 40); 
        drawEyeFrame(ctx, rightEyeFrame, "lime", 40);

        if (window.cv && window.cv.getPerspectiveTransform) { 
          rectifyEyePatchH(grabber, leftEyeFrame, leftPatchRef.current); 
          rectifyEyePatchH(grabber, rightEyeFrame, rightPatchRef.current); 
        } else if (window.cv && window.cv.getAffineTransform) { 
          rectifyEyePatch(grabber, leftEyeFrame, leftPatchRef.current); 
          rectifyEyePatch(grabber, rightEyeFrame, rightPatchRef.current); 
        }

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

        <canvas 
          ref={frameCanvasRef} 
          style={{ 
            display: "none" 
          }} 
        />

        <canvas 
          ref={leftPatchRef} 
          style={{ 
            position:"absolute", 
            left:20, 
            bottom:20, 
            zIndex:10, 
            width:120, 
            height:60, 
            border:"1px solid #ff0" 
          }} 
        />
        <canvas 
          ref={rightPatchRef} 
          style={{ 
            position:"absolute", 
            left:160, 
            bottom:20, 
            zIndex:10, 
            width:120, 
            height:60, 
            border:"1px solid #0f0" 
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
