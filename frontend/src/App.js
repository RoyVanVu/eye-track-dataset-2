import React, {useRef, useState, useEffect} from 'react';
import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection'
import Webcam from "react-webcam";
import './App.css';
import { drawMesh, displayPoseInfo,
         getEyeLocalFrames, drawEyeFrame, getIrisCenters, getNormalizedIris, makeEMA2,
         pickPoint3D, estimateRtHorn, eulerFromR,
         warpEyePatch, fitYawPitchModel, evalYawPitch,
         eyeAnglesToHeadVec,
         headToCamVec,
         combineCyclopean,
         vecToYawPitchDeg,
         drawArrow2D,
 } from './utilities';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const frameCanvasRef = useRef(null); 
  const leftEyePatchRef = useRef(null);
  const rightEyePatchRef = useRef(null);
  const emaLeftRef = useRef(makeEMA2(0.6));
  const emaRightRef = useRef(makeEMA2(0.6));
  const lastNormRef = useRef(null);
  const gazeOffsetRef = useRef({ left: {x: 0, y: 0}, right: {x: 0, y: 0} });
  const headTemplateRef = useRef(null);
  const RtRef = useRef({ R_now: null, t_now: null });
  const calibRef = useRef({
    left: { samples: [], model: null },
    right: { samples: [], model: null }
  });
  const defaultGainRef = useRef({
    left: { kx: 20, ky: 15 },
    right: { kx: 20, ky: 15 }
  });
  const gazeHeadRef = useRef({
    left: [0, 0, 1],
    right: [0, 0, 1]
  });

  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrationPose, setCalibrationPose] = useState(null);

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
        headTemplateRef.current = pickPoint3D(landmarks);
        setIsCalibrated(true);
        console.log("Saved head template (rigid set) with", headTemplateRef.current.length, "points");

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

  const addCalibSample = (eye, dx, dy, yawLabelDeg, pitchLabelDeg) => {
    calibRef.current[eye].samples.push({
      dx, dy, yaw: yawLabelDeg, pitch: pitchLabelDeg
    });
    console.log(`Add sample ${eye}:`, dx, dy, '->', yawLabelDeg, pitchLabelDeg);
  };

  const fitCalib = (eye, order=1) => {
    const samples = calibRef.current[eye].samples;
    if (samples.length < (order === 1 ? 2 : 6)) {
      console.warn("Not enough samples for:", eye);
      return;
    }
    const model = fitYawPitchModel(samples, order);
    calibRef.current[eye].model = model;
    console.log('Fitted model', eye, model);
  };

  const predictEyeAngles = (eye, dx, dy) => {
    const side = calibRef.current[eye];
    if (side.model) return evalYawPitch(dx, dy, side.model);
    const g = defaultGainRef.current[eye];
    return {
      yaw: g.kx * dx,
      pitch: g.ky * dy
    };
  };

  useEffect(() => {
    let intervalId = null;
    let isRunning = true;

    const runFacemesh = async () => {
      const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
      const detectorConfig = {
        runtime: 'tfjs',
        refineLandmarks: true,
      };
      const detector = await faceLandmarksDetection.createDetector(model, detectorConfig);

      intervalId = setInterval(() => {
        if (isRunning) {
          detect(detector);
        }
      }, 100);
    };

    runFacemesh();

    return () => {
      isRunning = false;
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isCalibrated]);

  useEffect(() => {
    const keyMap = {
      '1': { yaw: -10, pitch: -10 }, '2': { yaw: 0, pitch: -10 }, '3': { yaw: 10, pitch: -10 },
      '4': { yaw: -10, pitch: 0 }, '5': { yaw: 0, pitch: 0 }, '6': { yaw: 10, pitch: 0 },
      '7': { yaw: -10, pitch: 10 }, '8': { yaw: 0, pitch: 10}, '9': { yaw: 10, pitch: 10 },
    };

    const onKey = (e) => {
      if (!isCalibrated || !lastNormRef.current) return;

      const zeroL = {
        x: lastNormRef.current.left.x - gazeOffsetRef.current.left.x,
        y: lastNormRef.current.left.y - gazeOffsetRef.current.left.y,
      };
      const zeroR = {
        x: lastNormRef.current.right.x - gazeOffsetRef.current.right.x,
        y: lastNormRef.current.right.y - gazeOffsetRef.current.right.y,
      };

      if (keyMap[e.key]) {
        const { yaw, pitch } = keyMap[e.key];
        addCalibSample('left', zeroL.x, zeroL.y, yaw, pitch);
        addCalibSample('right', zeroR.x, zeroR.y, yaw, pitch);
      }
      if (e.key === 'f') {
        fitCalib('left', 1);
        fitCalib('right', 1);
      }
      if (e.key === 'F') {
        fitCalib('left', 2);
        fitCalib('right', 2);
      }
    };

    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isCalibrated]);

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

    const fcv = frameCanvasRef.current;
    fcv.width = videoWidth;
    fcv.height = videoHeight;
    const fctx = fcv.getContext("2d");
    fctx.drawImage(video, 0, 0, videoWidth, videoHeight);
    const frameImage = fctx.getImageData(0, 0, videoWidth, videoHeight);

    const faces = await net.estimateFaces(video);
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, videoWidth, videoHeight); 
    // drawMesh(faces, ctx);

    if (!faces.length) return;

    const landmarks = faces[0].keypoints.map(kp => [kp.x, kp.y, kp.z]);
    if (headTemplateRef.current) {
      const obsPts = pickPoint3D(landmarks);
      const Rt = estimateRtHorn(headTemplateRef.current, obsPts);
      if (Rt) {
        RtRef.current = Rt;
        // console.log("R_now:", Rt.R_now, "t_now:", Rt.t_now);

        const { yaw, pitch, roll } = eulerFromR(Rt.R_now);
        displayPoseInfo(ctx, { angles: { yaw, pitch, roll }, matrices: { R: Rt.R_now } });
      }
    }

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

    if (isCalibrated && norm) {
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

        const angL = predictEyeAngles('left', zeroL.x, zeroL.y);
        const angR = predictEyeAngles('right', zeroR.x, zeroR.y);

        const angL_fix = { yaw: -angL.yaw, pitch: -angL.pitch };
        const angR_fix = { yaw: -angR.yaw, pitch: -angR.pitch };

        const gL_head = eyeAnglesToHeadVec(angL_fix.yaw, angL_fix.pitch);
        const gR_head = eyeAnglesToHeadVec(angR_fix.yaw, angR_fix.pitch);
        gazeHeadRef.current.left = gL_head;
        gazeHeadRef.current.right = gR_head;
        // console.log('gL_head', gL_head, 'gR_head', gR_head);

        const R = RtRef.current.R_now;
        if (R) {
          const gL_cam = headToCamVec(R, gL_head, true);
          const gR_cam = headToCamVec(R, gR_head, true);
          console.log('gL_cam', gL_cam, 'gR_cam', gR_cam);
          const gC_cam = combineCyclopean(gL_cam, gR_cam, 0.5, 0.5);
          const cy = vecToYawPitchDeg(gC_cam);

          // ctx.fillStyle = "rgba(0,0,0,0.6)";
          // ctx.fillRect(10, 260, 360, 40);
          // ctx.fillStyle = "#fff";
          // ctx.font = "14px Arial";
          // ctx.fillText(`Cyclopean yaw/pitch = (${cy.yaw.toFixed(1)}°, ${cy.pitch.toFixed(1)}°)`, 20, 285);

          const cCx = 0.5 * (leftEyeFrame.c[0] + rightEyeFrame.c[0]);
          const cCy = 0.5 * (leftEyeFrame.c[1] + rightEyeFrame.c[1]);

          const scalePx = 140;
          const denom = Math.max(0.35, gC_cam[2]);
          const endX = cCx + scalePx * (gC_cam[0] / denom);
          const endY = cCy - scalePx * (gC_cam[1] / denom);

          drawArrow2D(ctx, cCx, cCy, endX, endY, "#ffeb3b");
        }

        // ctx.fillStyle = "rgba(0,0,0,0.6)";
        // ctx.fillRect(10, 140, 300, 60);
        // ctx.fillStyle = "#fff";
        // ctx.font = "14px Arial";
        // ctx.fillText(`L(x, y) ~ (${zeroL.x.toFixed(2)}, ${zeroL.y.toFixed(2)})`, 20, 160);
        // ctx.fillText(`R(x, y) ~ (${zeroR.x.toFixed(2)}, ${zeroR.y.toFixed(2)})`, 20, 180);

        // ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
        // ctx.fillRect(10, 200, 360, 60);
        // ctx.fillStyle = "#fff";
        // ctx.font = "14px Arial";
        // ctx.fillText(`L yaw/pitch = (${angL.yaw.toFixed(1)} degree, ${angL.pitch.toFixed(1)} degree)`, 20, 220);
        // ctx.fillText(`R yaw/pitch = (${angR.yaw.toFixed(1)} degree, ${angR.pitch.toFixed(1)} degree)`, 20, 240);
      }

      const Lpatch = warpEyePatch(frameImage, leftEyeFrame, 128, 96, 1.4, 1.6);
      const Rpatch = warpEyePatch(frameImage, rightEyeFrame, 128, 96, 1.4, 1.6);

      const lcv = leftEyePatchRef.current, rcv = rightEyePatchRef.current;
      if (lcv && rcv) {
        lcv.width = Lpatch.width; lcv.height = Lpatch.height;
        rcv.width = Rpatch.width; rcv.height = Rpatch.height;
        lcv.getContext("2d").putImageData(Lpatch, 0, 0);
        rcv.getContext("2d").putImageData(Rpatch, 0, 0);
      }
    }
  };

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
          ref={leftEyePatchRef}
          style={{
            position:"absolute",
            left:20,
            bottom:20,
            width:128,
            height:96,
            zIndex:11,
            background:"rgba(0, 0, 0, 0.4)"
          }}
        />

        <canvas 
          ref={rightEyePatchRef}
          style={{
            position:"absolute",
            left:170,
            bottom:20,
            width:128,
            height:96,
            zIndex:11,
            background:"rgba(0, 0, 0, 0.4)"
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
