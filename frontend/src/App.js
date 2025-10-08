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
         solveScreenPlaneLS,
         intersectRayToScreenXY,
         fitScreenXYModel,
         evalScreenXYModel,
         debiasEyeYByPitch,
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
  const eyeCalibRef = useRef({
    active: false,
    normPoints: [],
    clicked: [],
    order: 1
  });
  const overlayRef = useRef(null);
  const eyeOriginsHeadRef = useRef({ left: null, right: null });
  const intrinsicsRef = useRef(null);
  const lastRayRef = useRef({ o: null, g: null });
  const screenCalibRef = useRef({ samples: [], solved: null });
  const xyCalibRef = useRef({ samples: [], model: null });
  const emaPogRef = useRef(makeEMA2(0.25));

  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrationPose, setCalibrationPose] = useState(null);
  const [calibVersion, setCalibVersions] = useState(0);

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

        const centers = getIrisCenters(landmarks);
        if (centers) {
          const tip = landmarks[1], br = landmarks[6];
          const fwd = (() => {
            const vx = tip[0] - br[0], vy = tip[1] - br[1], vz = tip[2] - br[2];
            const n = Math.hypot(vx, vy, vz) || 1;
            return [vx / n, vy / n, vz / n];
          })();

          const pdUnits = Math.hypot(
            centers.left[0] - centers.right[0],
            centers.left[1] - centers.right[1],
            (centers.left[2] ?? 0) - (centers.right[2] ?? 0)
          );

          const EYE_RADIUS_MM = 12, PD_MM = 63;
          const rUnits = pdUnits * (EYE_RADIUS_MM / PD_MM);

          const oL_head = [
            centers.left[0] - fwd[0] * rUnits,
            centers.left[1] - fwd[1] * rUnits,
            centers.left[2] - fwd[2] * rUnits,
          ];
          const oR_head = [
            centers.right[0] - fwd[0] * rUnits,
            centers.right[1] - fwd[1] * rUnits,
            centers.right[2] - fwd[2] * rUnits,
          ];
          eyeOriginsHeadRef.current = { left: oL_head, right: oR_head };
          console.log("Saved eye origins (head frame)", eyeOriginsHeadRef.current);
        } else {
          console.warn("No iris centers at calibration; origin fallback will be eye-frame midpoint.");
        }

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

  const buildGrid = () => {
    const cols = [0.10, 0.5, 0.9];
    const rows = [0.10, 0.5, 0.9];
    const pts = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) pts.push({ u: cols[c], v: rows[r] });
    return pts;
  };

  const labelForIndex = (idx) => {
    const row = Math.floor(idx / 3);
    const col = idx % 3;
    const yawVals = [-10, 0, 10];
    const pitchVals = [-10, 0, 10];
    return {
      yaw: yawVals[col],
      pitch: pitchVals[row]
    };
  };

  const startEyeCalibration = () => {
    if (!isCalibrated) return;
    eyeCalibRef.current.active = true;
    eyeCalibRef.current.normPoints = buildGrid();
    eyeCalibRef.current.clicked = Array(9).fill(false);

    screenCalibRef.current.samples = [];
    screenCalibRef.current.solved = null;

    xyCalibRef.current.samples = [];
    xyCalibRef.current.model = null;

    setCalibVersions(v => v + 1);
  };

  const stopEyeCalibration = () => {
    eyeCalibRef.current.active = false;
    setCalibVersions(v => v + 1);
  };

  const transformRT = (R, t, p) => ([
    R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t[0],
    R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t[1],
    R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t[2],
  ])

  const projectCamToPix = (p, K) => {
    const { fx, fy, cx, cy } = K;
    const z = Math.max(1e-6, p[2]);
    return [cx + fx * (p[0] / z), cy - fy * (p[1] / z)];
  }

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
    const ocv = overlayRef.current;
    if (!ocv) return;
    
    const setSize = () => {
      ocv.width = window.innerWidth;
      ocv.height = window.innerHeight;
    };
    setSize();
    window.addEventListener("resize", setSize);

    const handleClick = (e) => {
      const ec = eyeCalibRef.current;
      if (!ec.active || !lastNormRef.current) return;

      const rect = ocv.getBoundingClientRect()
      const x = (e.clientX - rect.left) * (ocv.width / rect.width);
      const y = (e.clientY - rect.top) * (ocv.height / rect.height);

      const radius = Math.max(18, Math.min(28, 0.02 * Math.min(ocv.width, ocv.height)));
      let hitIdx = -1;
      let minD2 = Infinity;
      for (let i = 0; i < ec.normPoints.length; i++) {
        if (ec.clicked[i]) continue;
        const px = ec.normPoints[i].u * ocv.width;
        const py = ec.normPoints[i].v * ocv.height;
        const d2 = (x - px) * (x - px) + (y - py) * (y - py);
        if (d2 < radius * radius && d2 < minD2) {
          minD2 = d2;
          hitIdx = i;
        }
      }
      if (hitIdx === -1) return;

      const ray = lastRayRef.current;
      if (ray && ray.o && ray.g) {
        screenCalibRef.current.samples.push({
          x, y,
          o: [...ray.o],
          g: [...ray.g],
          idx: hitIdx,
          w: ocv.width,
          h: ocv.height
        });
      } else {
        console.warn("No 3D gaze ray at click; skip screen sample.")
      }

      const headAnglesNow = (RtRef.current?.R_now)
        ? eulerFromR(RtRef.current.R_now)
        : { yaw: 0, pitch: 0 };

      const smL = emaLeftRef.current(lastNormRef.current.left);
      const smR = emaRightRef.current(lastNormRef.current.right);

      const zeroL = {
        x: smL.x - gazeOffsetRef.current.left.x,
        y: smL.y - gazeOffsetRef.current.left.y,
      };
      const zeroR = {
        x: smR.x - gazeOffsetRef.current.right.x,
        y: smR.y - gazeOffsetRef.current.right.y,
      };

      const zL = debiasEyeYByPitch(zeroL, headAnglesNow, 0.20);
      const zR = debiasEyeYByPitch(zeroR, headAnglesNow, 0.20);

      const u = x / ocv.width;
      const v = y / ocv.height;

      xyCalibRef.current.samples.push({
        u, v, 
        zL, zR, 
        head: null
      });

      if (xyCalibRef.current.samples.length >= 6) {
        xyCalibRef.current.model = fitScreenXYModel(xyCalibRef.current.samples, 2, 1e-5, false);
        console.log("Fitted XY model:", xyCalibRef.current.model);
      } else {
        console.warn("Not enough samples for XY model.");
      }

      const { yaw, pitch } = labelForIndex(hitIdx);
      addCalibSample('left', zeroL.x, zeroL.y, yaw, pitch);
      addCalibSample('right', zeroR.x, zeroR.y, yaw, pitch);

      ec.clicked[hitIdx] = true;
      setCalibVersions(v => v + 1);

      if (ec.clicked.every(Boolean)) {
        fitCalib('left', ec.order);
        fitCalib('right', ec.order);

        const ocv = overlayRef.current;
        if (ocv && screenCalibRef.current.samples.length >= 5) {
          const plane = solveScreenPlaneLS(
            screenCalibRef.current.samples,
            ocv.width * 0.5,
            ocv.height * 0.5
          );
          screenCalibRef.current.solved = { ...plane, W: ocv.width, H: ocv.height };
          console.log("Screen plane solved:", screenCalibRef.current.solved);
        } else {
          console.warn("Not enough screen samples to solve plane.");
        }

        setCalibVersions(v => v + 1);
        stopEyeCalibration();
      }
    };

    ocv.addEventListener('click', handleClick);
    return () => {
      ocv.removeEventListener("click", handleClick);
      window.removeEventListener("resize", setSize);
    };
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

    if (
      !intrinsicsRef.current ||
      intrinsicsRef.current._w !== videoWidth ||
      intrinsicsRef.current._h !== videoHeight
    ) {
      const f = 0.8 * Math.min(videoWidth, videoHeight);
      intrinsicsRef.current = {
        fx: f,
        fy: f,
        cx: videoWidth / 2,
        cy: videoHeight / 2,
        _w: videoWidth,
        _h: videoHeight,
      };
    }

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

        const angL_fix = { yaw: angL.yaw, pitch: -angL.pitch };
        const angR_fix = { yaw: angR.yaw, pitch: -angR.pitch };

        const gL_head = eyeAnglesToHeadVec(angL_fix.yaw, angL_fix.pitch);
        const gR_head = eyeAnglesToHeadVec(angR_fix.yaw, angR_fix.pitch);
        gazeHeadRef.current.left = gL_head;
        gazeHeadRef.current.right = gR_head;

        const R = RtRef.current.R_now;
        const t = RtRef.current.t_now;
        if (R && t) {
          const gL_cam = headToCamVec(R, gL_head, true);
          const gR_cam = headToCamVec(R, gR_head, true);
          const gC_cam = combineCyclopean(gL_cam, gR_cam, 0.5, 0.5);
          
          let startX, startY, endX, endY;

          if (eyeOriginsHeadRef.current.left && eyeOriginsHeadRef.current.right) {
            const oL_cam = transformRT(R, t, eyeOriginsHeadRef.current.left);
            const oR_cam = transformRT(R, t, eyeOriginsHeadRef.current.right);
            const oC_cam = [
              0.5 * (oL_cam[0] + oR_cam[0]),
              0.5 * (oL_cam[1] + oR_cam[1]),
              0.5 * (oL_cam[2] + oR_cam[2]),
            ];
            
            const z0raw = oC_cam[2];
            const z0 = (Number.isFinite(z0raw) && z0raw > 1e-3) ? z0raw : 1;
            const oC_cam_safe = [oC_cam[0], oC_cam[1], z0];
            lastRayRef.current = { o: oC_cam_safe, g: gC_cam };
            [startX, startY] = projectCamToPix(oC_cam_safe, intrinsicsRef.current);
            
            const W = videoWidth, H = videoHeight;
            let s = 0.4 * z0;
            let tries = 0;
            const inView = (x, y) => Number.isFinite(x) && Number.isFinite(y) && x >= -W && x <= 2 * W && y >= -H && y <= 2 * H;

            while (tries < 6) {
              const p2_cam = [
                oC_cam_safe[0] + s * gC_cam[0],
                oC_cam_safe[1] + s * gC_cam[1],
                oC_cam_safe[2] + s * gC_cam[2],
              ];
              const pt = projectCamToPix(p2_cam, intrinsicsRef.current);
              if (inView(pt[0], pt[1])) {
                endX = pt[0]; endY = pt[1];
                break;
              }
              s *= 0.5;
              tries++;
            }

            const inViewStrict = (x, y) => 
              Number.isFinite(x) && Number.isFinite(y) && x >= 0 && x <= videoWidth && y >= 0 && y <= videoHeight;

            if (!Number.isFinite(endX) || !Number.isFinite(endY) || !inViewStrict(startX, startY)) {
              const cCx = 0.5 * (leftEyeFrame.c[0] + rightEyeFrame.c[0]);
              const cCy = 0.5 * (leftEyeFrame.c[1] + rightEyeFrame.c[1]);
              startX = cCx; startY = cCy;

              const denom = Math.max(0.35, gC_cam[2]);
              const scalePx = 140;
              endX = cCx + scalePx * (gC_cam[0] / denom);
              endY = cCy - scalePx * (gC_cam[1] / denom);
            }

          } else {
            const cCx = 0.5 * (leftEyeFrame.c[0] + rightEyeFrame.c[0]);
            const cCy = 0.5 * (leftEyeFrame.c[1] + rightEyeFrame.c[1]);
            startX = cCx; startY = cCy;

            const scalePx = 140;
            endX = cCx + scalePx * gC_cam[0];
            endY = cCy - scalePx * gC_cam[1];
          }      
          drawArrow2D(ctx, startX, startY, endX, endY, "#ffeb3b");
        }        
      }

      const ocv = overlayRef.current;
      if (ocv) {
        const octx = ocv.getContext("2d");
        octx.clearRect(0, 0, ocv.width, ocv.height);
    
        if (eyeCalibRef.current.active) {
          const ec = eyeCalibRef.current;
          const radius = Math.max(10, Math.min(16, 0.012 * Math.min(ocv.width, ocv.height)));

          for (let i = 0; i < ec.normPoints.length; i++) {
            const { u, v } = ec.normPoints[i];
            const px = u * ocv.width;
            const py = v * ocv.height;

            octx.beginPath();
            octx.arc(px, py, radius, 0, Math.PI * 2);
            octx.fillStyle = ec.clicked[i] ? "rgba(0, 255, 0, 0.95)" : "rgba(255, 255, 255, 0.9)";
            octx.fill();
            octx.lineWidth = 2;
            octx.strokeStyle = "#222";
            octx.stroke();
          }

          const done = ec.clicked.filter(Boolean).length;
          octx.fillStyle = "rgba(0, 0, 0, 0.6)";
          octx.fillRect(10, 320, 140, 26);
          octx.fillStyle = "#fff";
          octx.font = "14px Arial";
          octx.fillText(`Eye calib: ${done}/9`, 24, 338);
        }

        const okAperture =
          leftEyeFrame.eyeHeight  > 0.25 * leftEyeFrame.eyeWidth &&
          rightEyeFrame.eyeHeight > 0.25 * rightEyeFrame.eyeWidth;

        if (!eyeCalibRef.current.active &&
            xyCalibRef.current.model &&
            lastNormRef.current &&
            okAperture) {
          
          const headAnglesNow = (RtRef.current?.R_now)
              ? eulerFromR(RtRef.current.R_now)
              : { yaw: 0, pitch: 0 };

          const smL = emaLeftRef.current(lastNormRef.current.left);
          const smR = emaRightRef.current(lastNormRef.current.right);
          
          const zL_raw = { x: smL.x - gazeOffsetRef.current.left.x, y: smL.y - gazeOffsetRef.current.left.y };
          const zR_raw = { x: smR.x - gazeOffsetRef.current.right.x, y: smR.y - gazeOffsetRef.current.right.y };

          const zL = debiasEyeYByPitch(zL_raw, headAnglesNow, 0.20);
          const zR = debiasEyeYByPitch(zR_raw, headAnglesNow, 0.20);

          let { u, v } = evalScreenXYModel(zL, zR, null, xyCalibRef.current.model);

          const uvSm = emaPogRef.current({ x: u, y: v });
          u = uvSm.x; v = uvSm.y;

          const xpix = Math.max(0, Math.min(ocv.width, u * ocv.width));
          const ypix = Math.max(0, Math.min(ocv.height, v * ocv.height));

          octx.beginPath();
          octx.arc(xpix, ypix, 8, 0, Math.PI * 2);
          octx.fillStyle = "rgba(255, 80, 80, 0.9)";
          octx.fill();
          octx.lineWidth = 3;
          octx.strokeStyle = "white";
          octx.stroke();
        } else if (!eyeCalibRef.current.active &&
            screenCalibRef.current.solved &&
            lastRayRef.current.o && lastRayRef.current.g) {

          const hit = intersectRayToScreenXY(
            lastRayRef.current.o,
            lastRayRef.current.g,
            screenCalibRef.current.solved
          );

          if (hit && hit.t > 0 && Number.isFinite(hit.x) && Number.isFinite(hit.y)) {
            // const ocv = overlayRef.current;
            const { W, H } = screenCalibRef.current.solved;
            const sx = ocv.width / W, sy = ocv.height / H;
            // const hx = hit.x * sx, hy = hit.y * sy;
            const x = Math.max(0, Math.min(ocv.width, hit.x * sx));
            const y = Math.max(0, Math.min(ocv.height, hit.y * sy));

            // const octx = ocv.getContext("2d");
            octx.beginPath();
            octx.arc(x, y, 8, 0, Math.PI*2);
            octx.fillStyle = "rgba(255, 80, 80, 0.9)";
            octx.fill();
            octx.lineWidth = 3;
            octx.strokeStyle = "white";
            octx.stroke();
          }
        }
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
          ref={overlayRef}
          style={{
            position:"fixed",
            inset:0,
            width:"100vw",
            height:"100vh",
            zIndex:20,
            pointerEvents:eyeCalibRef.current.active ? "auto" : "none",
            background:"transparent"
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

        <button
          onClick={startEyeCalibration}
          disabled={!isCalibrated || eyeCalibRef.current.active}
          style={{
            position:"absolute",
            top:20,
            left:240,
            zIndex:12,
            padding:"10px 20px",
            backgroundColor: !isCalibrated
              ? "gray"
              : (eyeCalibRef.current.active ? "#ff9800" : "#673ab7"),
            color:"white",
            border:"none",
            borderRadius:"5px",
            cursor: (!isCalibrated || eyeCalibRef.current.active) ? "default" : "pointer"
          }}
        >
          {eyeCalibRef.current.active ? "Eye calibration: click 9 dots" : "Calibrate eyes (3x3)"}
        </button>

        <div
          style={{
            position:"absolute",
            top:70,
            left:20,
            zIndex:10,
            background:"rgba(0, 0, 0, 0.55)",
            color:"#fff",
            padding:"6px 8px",
            borderRadius:"6px"
          }}
        >
          <label>Eye model order: </label>
          <select
            value={eyeCalibRef.current.order}
            onChange={e => { eyeCalibRef.current.order = parseInt(e.target.value, 10); setCalibVersions(v => v + 1); }}
            disabled={eyeCalibRef.current.active}
            style={{ marginLeft: 6 }}
          >
            <option value={1}>Linear</option>
            <option value={2}>Quadratic</option>
          </select>
        </div>

        <div
          style={{
            position:"absolute",
            top:120,
            left:20,
            zIndex:10,
            background:"rgba(0, 0, 0, 0.55)",
            color:"#fff",
            padding:"6px 8px",
            borderRadius:"6px",
            minWidth:260
          }}
        >
          <div><b>Eye calib status</b></div>
          <div>
            Left: {
              calibRef.current.left.model
                ? `Fitted (order ${calibRef.current.left.model.order}, N=${calibRef.current.left.samples.length})`
                : 'Default gains'
            }
          </div>
          <div>
            Right: {
              calibRef.current.right.model
                ? `Fitted (order ${calibRef.current.right.model.order}, N=${calibRef.current.right.samples.length})`
                : 'Default gains'
            }
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
