import React, {useRef, useState, useEffect} from 'react';
import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection'
import Webcam from "react-webcam";
import './App.css';
import { drawMesh, calculateHeadPose, calculateRelativePose, displayPoseInfo,
         getEyeLocalFrames, drawEyeFrame, getIrisCenters, getNormalizedIris, makeEMA2
 } from './utilities';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const frameCanvasRef = useRef(null); 
  const emaLeftRef = useRef(makeEMA2(0.6));
  const emaRightRef = useRef(makeEMA2(0.6));
  const lastNormRef = useRef(null);
  const gazeOffsetRef = useRef({ left: {x: 0, y: 0}, right: {x: 0, y: 0} });

  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrationPose, setCalibrationPose] = useState(null);

  // useEffect(() => {
  //   runFacemesh();
  // }, []);

  const handleCalibration = async () => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
      const detectorConfig = {
        runtime: 'tfjs',
        refineLandmarks: true,
      };
      const detector = await faceLandmarksDetection.createDetector(model, detectorConfig);

      const face = await detector.estimateFaces(video);

      if (face.length > 0) {
        const landmarks = face[0].keypoints.map(kp => [kp.x, kp.y, kp.z]);
        const pose = calculateHeadPose(landmarks);
        setCalibrationPose(pose);
        setIsCalibrated(true);
        console.log("Calibration completed with pose:", pose);
        try {
          if (lastNormRef.current) {
            gazeOffsetRef.current = {
              left: { ...lastNormRef.current.left },
              right: { ...lastNormRef.current.right },
            };
          } else {
            const norm0 = getNormalizedIris(landmarks);
            if (norm0) {
              const mirrored0 = landmarks[263][0] < landmarks[33][0];
              if (mirrored0) {
                norm0.left.x *= -1;
                norm0.right.x *= -1;
              }
              gazeOffsetRef.current = {
                left: { ...norm0.left },
                right: { ...norm0.right },
              };
            }
          }
          console.log("Gaze offset set:", gazeOffsetRef.current);
        } catch (e) {
          console.warn("Failed to set gaze offset:", e);
        }
      } else {
        alert("No face detected! Please make sure your face is visible.");
      }
    }
  };

  const runFacemesh = async () => {
    const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
    const detectorConfig = {
      runtime: 'tfjs',
      refineLandmarks: true,
    };
    const detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
    
    setInterval(() => {
      detect(detector)
    }, 100)
  };

  const detect = async (net) => {
    if (
      typeof webcamRef.current === "undefined" ||
      webcamRef.current === null ||
      webcamRef.current.video.readyState !== 4
    ) return;

    const video = webcamRef.current.video;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    video.width = videoWidth;
    video.height = videoHeight;
    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    const faces = await net.estimateFaces(video);
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, videoWidth, videoHeight); // <- QUAN TRỌNG

    if (!faces.length) return;

    const landmarks = faces[0].keypoints.map(kp => [kp.x, kp.y, kp.z]);

    // Luôn tính norm + mirror + cache, để nút Calibrate hứng được frame mới nhất
    let norm = getNormalizedIris(landmarks);
    if (norm) {
      const mirrored = landmarks[263][0] < landmarks[33][0];
      if (mirrored) {
        norm = {
          left:  { x: -norm.left.x,  y: norm.left.y  },
          right: { x: -norm.right.x, y: norm.right.y },
        };
      }
      lastNormRef.current = { left: { ...norm.left }, right: { ...norm.right } };
    }

    // Nếu đã calibrate **và** có norm thì mới vẽ/gửi đầu vào cho B4
    if (isCalibrated && norm) {
      // (tuỳ chọn) nếu chưa dùng, có thể bỏ block grabber này
      // const grabber = frameCanvasRef.current;
      // const gctx = grabber.getContext("2d");
      // grabber.width = videoWidth;
      // grabber.height = videoHeight;
      // gctx.drawImage(video, 0, 0, videoWidth, videoHeight);

      const { left: leftEyeFrame, right: rightEyeFrame } = getEyeLocalFrames(landmarks);
      drawEyeFrame(ctx, leftEyeFrame, "yellow", 40);
      drawEyeFrame(ctx, rightEyeFrame, "lime", 40);

      const irisCenters = getIrisCenters(landmarks);
      if (irisCenters) {
        ctx.beginPath();
        ctx.arc(irisCenters.left[0],  irisCenters.left[1],  3, 0, Math.PI * 2);
        ctx.fillStyle = "#00ff00";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(irisCenters.right[0], irisCenters.right[1], 3, 0, Math.PI * 2);
        ctx.fillStyle = "#00ff00";
        ctx.fill();
      }

      // EMA và zero-center (đầu vào B4)
      const smL = emaLeftRef.current(norm.left);
      const smR = emaRightRef.current(norm.right);
      if (smL && smR) {
        const zeroL = {
          x: smL.x - gazeOffsetRef.current.left.x,
          y: smL.y - gazeOffsetRef.current.left.y,
        };
        const zeroR = {
          x: smR.x - gazeOffsetRef.current.right.x,
          y: smR.y - gazeOffsetRef.current.right.y,
        };

        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(10, 140, 300, 60);
        ctx.fillStyle = "#fff";
        ctx.font = "14px Arial";
        ctx.fillText(`L(x, y) ~ (${zeroL.x.toFixed(2)}, ${zeroL.y.toFixed(2)})`, 20, 160);
        ctx.fillText(`R(x, y) ~ (${zeroR.x.toFixed(2)}, ${zeroR.y.toFixed(2)})`, 20, 180);
      }

      const currentPose = calculateHeadPose(landmarks);
      const relativePose = calculateRelativePose(calibrationPose, currentPose);
      displayPoseInfo(ctx, relativePose);
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
