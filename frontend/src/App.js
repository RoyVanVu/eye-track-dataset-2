import React, {useRef, useState, useEffect, useReducer} from 'react';
import * as tf from "@tensorflow/tfjs";
import '@tensorflow/tfjs-backend-webgl';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection'
import Webcam from "react-webcam";
import './App.css';
import { drawMesh, displayPoseInfo,
         getEyeLocalFrames, drawEyeFrame, getIrisCenters, getRectifiedIrisOffsets, makeEMA2,
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
         buildEyeLocalFrame,
         buildVerticalFeatures,
         buildHorizontalFeatures,
         getEyeApertures,
         normalizeAperture,
         getLissajousCoords,
         isBlinking,
         createGazeModel,
 } from './utilities';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const frameCanvasRef = useRef(null); 
  const leftEyePatchRef = useRef(null);
  const rightEyePatchRef = useRef(null);
  const emaLeftRef = useRef(makeEMA2(0.5));
  const emaRightRef = useRef(makeEMA2(0.5));
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
  const xyCalibRef = useRef({ samples: [], model: null });
  const emaPogRef = useRef(makeEMA2(0.40));
  const eyeCanonRef = useRef({ left: null, right: null });
  const headPoseNamesRef = useRef(['straight', 'left', 'right', 'up', 'down']);
  const currentPoseIndexRef = useRef(0);
  const poseClickedDotsRef = useRef([]);
  const errorHistoryRef = useRef([]);
  const baselineSetRef = useRef(false);
  const detectorRef = useRef(null);
  const isTestingRef = useRef(false);
  const testPointIndexRef = useRef(-1);
  const latestPogRef = useRef({ x: 0, y: 0 });

  const calibStageRef = useRef('idle');
  const trainingDataRef = useRef([]);
  const targetQueueRef = useRef([]);
  const baselineSamplesRef = useRef([]);
  const calibStartTimeRef = useRef(null);
  const nnModelRef = useRef(null);
  const calibDotRef = useRef({ u: 0.5, v: 0.5 });

  const featureNormRef = useRef({
    irisScale: 0.5,
    headYawScale: 30,
    headPitchScale: 30,
    apertureScale: 0.3
  });

  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrationPose, setCalibrationPose] = useState(null);
  const [calibVersion, setCalibVersions] = useState(0);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [irisDebugInfo, setIrisDebugInfo] = useState({
    raw: { left: null, right: null },
    rectified: { left: null, right: null },
    smoothed: { left: null, right: null },
    zeroed: { left: null, right: null }
  });

  const [pogTraceInfo, setPogTraceInfo] = useState({
    input: null,
    features: null,
    weights: null,
    rawPrediction: null,
    smoothed: null,
    clamped: null,
    final: null
  });

  const [apertureDebugInfo, setApertureDebugInfo] = useState({
    left: null,
    right: null,
    leftNorm: null,
    rightNorm: null
  });

  const [isTesting, setIsTesting] = useState(false);
  const [testPointIndex, setTestPointIndex] = useState(-1);
  const testResultsRef = useRef([]);
  const [testSummary, setTestSummary] = useState(null);

  const [calibStage, setCalibStage] = useState('idle');
  const [calibDot, setCalibDot] = useState({ u: 0.5, v: 0.5 });
  const [calibInstruction, setCalibInstruction] = useState("");

  useEffect(() => {
    isTestingRef.current = isTesting;
    testPointIndexRef.current = testPointIndex;
  }, [isTesting, testPointIndex]);

  useEffect(() => {
    calibStageRef.current = calibStage;
  }, [calibStage]);

  useEffect(() => {
    console.log('╔════════════════════════════════════════╗');
    console.log('║          CONFIG QUICK TEST             ║');
    console.log('╚════════════════════════════════════════╝');
    console.log('EMA iris alpha:', 0.5);
    console.log('EMA PoG alpha:', 0.30);
    console.log('Default gains:', defaultGainRef.current);
    console.log('Ridge (order 2/3):', 0.5);
    console.log('Feature normalization:', 'ENABLED (scale 0.5)');
    console.log('═══════════════════════════════════════════\n');
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePos({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);
  
  useEffect(() => {
    (async () => {
      try {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('TFJS backend:', tf.getBackend());
      } catch (e) {
        console.error('Init TF failed:', e);
      }
    })();
  }, []);

  const addCalibSample = (eye, dx, dy, yawLabelDeg, pitchLabelDeg) => {
    calibRef.current[eye].samples.push({
      dx, dy, yaw: yawLabelDeg, pitch: pitchLabelDeg
    });
    // console.log(`Add sample ${eye}:`, dx, dy, '->', yawLabelDeg, pitchLabelDeg);
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

  const startEyeCalibration = async () => {
    trainingDataRef.current = [];
    targetQueueRef.current = [];
    baselineSamplesRef.current = [];
    baselineSetRef.current = false;
    headTemplateRef.current = null;

    gazeOffsetRef.current = {
      left: { x: 0, y: 0 },
      right: { x: 0, y: 0 }
    };

    if (!detectorRef.current && webcamRef.current?.video.readyState === 4) {
      try {
        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        detectorRef.current = await faceLandmarksDetection.createDetector(model, {
          runtime: 'tfjs',
          refineLandmarks: true,
          modelUrl: `${window.location.origin}/models/attention_mesh/1/model.json`,
        }).catch(() => {
          return faceLandmarksDetection.createDetector(model, {
            runtime: 'tfjs',
            refineLandmarks: false,
            modelUrl: `${window.location.origin}/models/face_landmarks/1/model.json`,
          })
        });
        console.log("Detector ready");
      } catch (e) {
        console.error("Detector creation failed:", e);
        alert("Failed to initialize detector!");
        eyeCalibRef.current.active = false;
        return;
      }
    }

    eyeCalibRef.current.active = true;
    setCalibStage('anchor');
    calibDotRef.current = { u: 0.5, v: 0.5 };
    setCalibDot({ u: 0.5, v: 0.5 });
    setCalibInstruction("Look at the CENTER dot and CLICK when ready");
    setCalibVersions(v => v + 1);

    console.log("Calibration started - Click CENTER dot first!");
  };

  const stopEyeCalibration = async () => {
    setCalibStage('idle');
    calibStageRef.current = 'idle';
    setCalibInstruction("Training Neural Network... Please wait.");

    const data = trainingDataRef.current;
    if (data.length < 100) {
      alert("Not enough data collected! Please follow the dot more closely.");
      eyeCalibRef.current.active = false;
      return;
    }

    const model = createGazeModel();
    const xs = tf.tensor2d(data.map(d => d.x));
    const ys = tf.tensor2d(data.map(d => d.y));

    console.log(`Training on ${data.length} samples...`);
    await model.fit(xs, ys, {
      epochs: 50,
      batchSize: 32,
      shuffle: true,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          setCalibInstruction(`Training... Epoch ${epoch + 1}/50 (loss: ${logs.loss.toFixed(4)})`);
          if (epoch % 10 === 0) {
            console.log(`Epoch ${epoch}: loss=${logs.loss.toFixed(4)}, val_loss=${logs.val_loss.toFixed(4)}`);
          }
        }
      }
    });

    nnModelRef.current = model;
    xs.dispose();
    ys.dispose();

    eyeCalibRef.current.active = false;
    setIsCalibrated(true);
    setCalibInstruction("");
    console.log("Neural Network Bridge Built Successfully!");
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

  const runAccuracyTest = async () => {
    setIsTesting(true);
    setTestSummary(null);
    testResultsRef.current = [];
    const points = buildGrid();

    const canvasW = overlayRef.current.width;
    const canvasH = overlayRef.current.height;

    for (let i = 0; i < points.length; i++) {
      setTestPointIndex(i);
      await new Promise(res => setTimeout(res, 5000));

      if (pogTraceInfo.final) {
        const targetX = points[i].u * canvasW;
        const targetY = points[i].v * canvasH;
        const predX = latestPogRef.current.x;
        const predY = latestPogRef.current.y;

        const dist = Math.sqrt(Math.pow(targetX - predX, 2) + Math.pow(targetY - predY, 2));

        testResultsRef.current.push({
          u: points[i].u,
          v: points[i].v,
          error: dist,
          target: {
            x: targetX,
            y: targetY
          },
          predicted: {
            x: predX,
            y: predY
          }
        });
      }
    }

    calculateFinalScore();
    setIsTesting(false);
    setTestPointIndex(-1);
  };

  const calculateFinalScore = () => {
    const results = testResultsRef.current;   
    if (results.length === 0) return;

    const avgError = results.reduce((sum, item) => sum + item.error, 0) / results.length;

    const screenMin = Math.min(overlayRef.current.width, overlayRef.current.height);
    const pctError = ((avgError / screenMin) * 100).toFixed(2);

    setTestSummary({
      avgPixelError: avgError.toFixed(1),
      maxError: Math.max(...results.map(r => r.error)).toFixed(1),
      percentageError: pctError,
      points: results
    });
  };

  useEffect(() => {
    let intervalId = null;
    let isRunning = true;

    const runFacemesh = async () => {
      const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
      async function makeDetector() {
        await tf.ready();
        try {
          return await faceLandmarksDetection.createDetector(model, {
            runtime: 'tfjs',
            refineLandmarks: true,
            modelUrl: `${window.location.origin}/models/attention_mesh/1/model.json`,
          });
        } catch (e) {
          console.warn('Local attention_mesh failed, fallback to non-iris locally.', e);
          return await faceLandmarksDetection.createDetector(model, {
            runtime: 'tfjs',
            refineLandmarks: false,
            modelUrl: `${window.location.origin}/models/face_landmarks/1/model.json`,
          });
        }
      }
      const detector = await makeDetector();

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
      if (!eyeCalibRef.current.active) return;
      if (!lastNormRef.current?.landmarks) {
        alert("No face detected! Please ensure your face is visible.");
        return;
      }

      const currentStage = calibStageRef.current;

      if (currentStage === 'anchor') {
        headTemplateRef.current = pickPoint3D(lastNormRef.current.landmarks);

        setCalibInstruction("Hold still... capturing baseline...");
        setCalibStage('capturing-baseline');
        baselineSamplesRef.current = [];
        console.log("Anchor clicked - starting baseline capture...");
      }
    };

    ocv.addEventListener('click', handleClick);
    return () => {
      ocv.removeEventListener("click", handleClick);
      window.removeEventListener("resize", setSize);
    };
  }, [isCalibrated]);

  const finalizeBaseline = (lastLandmarks, lastR, lastT) => {
    const samples = baselineSamplesRef.current;
    if (samples.length === 0) return;

    const avgLeft = {
      x: samples.reduce((sum, s) => sum + s.left.x, 0) / samples.length,
      y: samples.reduce((sum, s) => sum + s.left.y, 0) / samples.length
    };
    const avgRight = {
      x: samples.reduce((sum, s) => sum + s.right.x, 0) / samples.length,
      y: samples.reduce((sum, s) => sum + s.right.y, 0) / samples.length
    };

    gazeOffsetRef.current = { left: avgLeft, right: avgRight };

    headTemplateRef.current = pickPoint3D(lastLandmarks);
    RtRef.current = { R_now: lastR, t_now: lastT };

    const centers = getIrisCenters(lastLandmarks);
    if (centers) {
      const tip = lastLandmarks[1];
      const br = lastLandmarks[6];
      const fwd = (() => {
        const vx = tip[0] - br[0];
        const vy = tip[1] - br[1];
        const vz = tip[2] - br[2];
        const n = Math.hypot(vx, vy, vz) || 1;
        return [vx / n, vy / n, vz / n];
      })();

      const pdUnits = Math.hypot(
        centers.left[0] - centers.right[0],
        centers.left[1] - centers.right[1],
        (centers.left[2] ?? 0) - (centers.right[2] ?? 0)
      );

      const EYE_RADIUS_MM = 12;
      const PD_MM = 63;
      const rUnits = pdUnits * (EYE_RADIUS_MM / PD_MM);

      eyeOriginsHeadRef.current = {
        left: [
          centers.left[0] - fwd[0] * rUnits,
          centers.left[1] - fwd[1] * rUnits,
          centers.left[2] - fwd[2] * rUnits,
        ],
        right: [
          centers.right[0] - fwd[0] * rUnits,
          centers.right[1] - fwd[1] * rUnits,
          centers.right[2] - fwd[2] * rUnits,
        ]
      };
    }

    const toHead = (p) => {
      const Rinv = [
        [lastR[0][0], lastR[1][0], lastR[2][0]],
        [lastR[0][1], lastR[1][1], lastR[2][1]],
        [lastR[0][2], lastR[1][2], lastR[2][2]]
      ];
      const q = [p[0] - lastT[0], p[1] - lastT[1], p[2] - lastT[2]];
      return [
        Rinv[0][0] * q[0] + Rinv[0][1] * q[1] + Rinv[0][2] * q[2],
        Rinv[1][0] * q[0] + Rinv[1][1] * q[1] + Rinv[1][2] * q[2],
        Rinv[2][0] * q[0] + Rinv[2][1] * q[1] + Rinv[2][2] * q[2],
      ];
    };

    const Lc = { 
      inner: lastLandmarks[133], 
      outer: lastLandmarks[33],
      up: lastLandmarks[159],
      down: lastLandmarks[145]
    };
    const Rc = {
      inner: lastLandmarks[362],
      outer: lastLandmarks[263],
      up: lastLandmarks[386],
      down: lastLandmarks[374]
    };

    const Lh = {
      inner: toHead(Lc.inner),
      outer: toHead(Lc.outer),
      up: toHead(Lc.up),
      down: toHead(Lc.down)
    };
    const Rh = {
      inner: toHead(Rc.inner),
      outer: toHead(Rc.outer),
      up: toHead(Rc.up),
      down: toHead(Rc.down)
    };

    const mkCanon = (E) => {
      const eye2D = {
        inner: [E.inner[0], E.inner[1]],
        outer: [E.outer[0], E.outer[1]],
        up: [E.up[0], E.up[1]],
        down: [E.down[0], E.down[1]],
      };
      return buildEyeLocalFrame(eye2D);
    };

    eyeCanonRef.current = { left: mkCanon(Lh), right: mkCanon(Rh) };

    baselineSetRef.current = true;
    console.log("Baseline Anchored:", gazeOffsetRef.current);
  };

  const detect = async (net) => {
    if (!webcamRef.current || !canvasRef.current || !overlayRef.current) return;

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
    if (!lastNormRef.current) {
      lastNormRef.current = {};
    }
    lastNormRef.current.landmarks = landmarks; 

    if (headTemplateRef.current) {
      const obsPts = pickPoint3D(landmarks);
      const Rt = estimateRtHorn(headTemplateRef.current, obsPts);
      if (Rt) {
        RtRef.current = Rt;
        const { yaw, pitch, roll } = eulerFromR(Rt.R_now);
        displayPoseInfo(ctx, { angles: { yaw, pitch, roll }, matrices: { R: Rt.R_now } });
      }
    }

    const R_now = RtRef.current?.R_now;
    let norm = null;
    if (R_now) {
      norm = getRectifiedIrisOffsets(landmarks, R_now, RtRef.current?.t_now, eyeCanonRef.current);
      if (norm) {
        lastNormRef.current.left = { ...norm.left };
        lastNormRef.current.right = { ...norm.right};
      }
    }

    const currentStage = calibStageRef.current;

    if (eyeCalibRef.current.active && norm) {
      if (currentStage === 'capturing-baseline') {
        baselineSamplesRef.current.push({
          left: { ...norm.left },
          right: { ...norm.right }
        });

        if (baselineSamplesRef.current.length >= 15) {
          finalizeBaseline(landmarks, RtRef.current.R_now, RtRef.current.t_now);
          setCalibStage('pursuit');
          calibStartTimeRef.current = Date.now();
        }
      }

      if (currentStage === 'pursuit') {
        const elapsed = (Date.now() - calibStartTimeRef.current) / 1000;

        if (elapsed >= 30) {
          stopEyeCalibration();
        } else {
          if (elapsed < 7) setCalibInstruction("Phase 1: Keep head STILL");
          else if (elapsed < 14) setCalibInstruction("Phase 2: Gently TILT head");
          else setCalibInstruction("Phase 3: Gently NOD head");

          const currentTarget = getLissajousCoords(elapsed);
          calibDotRef.current = currentTarget;
          setCalibDot(currentTarget);

          const apertures = getEyeApertures(landmarks);
          const leftFrame = getEyeLocalFrames(landmarks).left;
          const rightFrame = getEyeLocalFrames(landmarks).right;
          const aL = normalizeAperture(apertures?.left, leftFrame.eyeWidth);
          const aR = normalizeAperture(apertures?.right, rightFrame.eyeWidth);

          if (!isBlinking(aL, aR)) {
            targetQueueRef.current.push(currentTarget);
            if (targetQueueRef.current.length > 2) {
              const delayedTarget = targetQueueRef.current.shift();
              const scales = featureNormRef.current;
              const head = eulerFromR(RtRef.current.R_now);

              const zeroL = {
                x: norm.left.x - gazeOffsetRef.current.left.x,
                y: norm.left.y - gazeOffsetRef.current.left.y
              };
              const zeroR = {
                x: norm.right.x - gazeOffsetRef.current.right.x,
                y: norm.right.y - gazeOffsetRef.current.right.y
              };

              const features = [
                zeroL.x / scales.irisScale, zeroL.y / scales.irisScale,
                zeroR.x / scales.irisScale, zeroR.y / scales.irisScale,
                head.yaw / scales.headYawScale, head.pitch / scales.headPitchScale,
                aL / scales.apertureScale, aR / scales.apertureScale
              ];

              trainingDataRef.current.push({
                x: features,
                y: [delayedTarget.u, delayedTarget.v]
              })
            }
          }
        }
      }
    }

    if (isCalibrated && norm) {
      const { left: leftEyeFrame, right: rightEyeFrame } = getEyeLocalFrames(landmarks);
      // drawEyeFrame(ctx, leftEyeFrame, "yellow", 40);
      // drawEyeFrame(ctx, rightEyeFrame, "lime", 40);

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

        setIrisDebugInfo({
          raw: irisCenters ? {
            left: { x: irisCenters.left[0], y: irisCenters.left[1] },
            right: { x: irisCenters.right[0], y: irisCenters.right[1] }
          } : null,
          rectified: {
            left: { x: norm.left.x, y: norm.left.y },
            right: { x: norm.right.x, y: norm.right.y }
          },
          smoothed: {
            left: { x: smL.x, y: smL.y },
            right: { x: smR.x, y: smR.y }
          },
          zeroed: {
            left: { x: zeroL.x, y: zeroL.y },
            right: { x: zeroR.x, y: zeroR.y }
          }
        });

        const apertures = getEyeApertures(landmarks);
        if (apertures) {
          const apertureL_norm = normalizeAperture(apertures.left, leftEyeFrame.eyeWidth);
          const apertureR_norm = normalizeAperture(apertures.right, rightEyeFrame.eyeWidth);
          
          setApertureDebugInfo({
            left: apertures.left.toFixed(1),
            right: apertures.right.toFixed(1),
            leftNorm: apertureL_norm?.toFixed(3),
            rightNorm: apertureR_norm?.toFixed(3)
          });
        }

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

        const okAperture =
          leftEyeFrame.eyeHeight  > 0.15 * leftEyeFrame.eyeWidth &&
          rightEyeFrame.eyeHeight > 0.15 * rightEyeFrame.eyeWidth;

        if (!eyeCalibRef.current.active &&
            nnModelRef.current &&
            lastNormRef.current &&
            okAperture) {
          
          const headAnglesNow = (RtRef.current?.R_now)
              ? eulerFromR(RtRef.current.R_now)
              : { yaw: 0, pitch: 0 };

          const smL = emaLeftRef.current(lastNormRef.current.left);
          const smR = emaRightRef.current(lastNormRef.current.right);
          
          const zL = { 
            x: smL.x - gazeOffsetRef.current.left.x, 
            y: smL.y - gazeOffsetRef.current.left.y 
          };
          const zR = { 
            x: smR.x - gazeOffsetRef.current.right.x, 
            y: smR.y - gazeOffsetRef.current.right.y 
          };

          const apertures = getEyeApertures(lastNormRef.current.landmarks);
          const aL = apertures ? normalizeAperture(apertures.left, leftEyeFrame.eyeWidth) : 0;
          const aR = apertures ? normalizeAperture(apertures.right, rightEyeFrame.eyeWidth) : 0;

          const scales = featureNormRef.current;
          const features = [
            zL.x / scales.irisScale, zL.y / scales.irisScale,
            zR.x / scales.irisScale, zR.y / scales.irisScale,
            headAnglesNow.yaw / scales.headYawScale,
            headAnglesNow.pitch / scales.headPitchScale,
            aL / scales.apertureScale,
            aR / scales.apertureScale
          ];

          const [u_raw, v_raw] = tf.tidy(() => {
            const input = tf.tensor2d([features]);
            const output = nnModelRef.current.predict(input);
            return output.dataSync();
          });

          const uvSm = emaPogRef.current({ x: u_raw, y: v_raw });
          let u = Math.max(0, Math.min(1, uvSm.x));
          let v = Math.max(0, Math.min(1, uvSm.y));

          const xpix = u * ocv.width;
          const ypix = v * ocv.height;

          latestPogRef.current = { x: xpix, y: ypix };
          
          setPogTraceInfo({
            input: {
              zL_y: zL.y,
              zR_y: zR.y,
              zL_x: zL.x,
              zR_x: zR.x,
              headYaw: headAnglesNow.yaw,
              headPitch: headAnglesNow.pitch
            },
            baseline: {
              left: { ...gazeOffsetRef.current.left },
              right: { ...gazeOffsetRef.current.right }
            },
            rectified: {
              left: { x: smL.x, y: smL.y },
              right: { x: smR.x, y: smR.y }
            },
            features: {
              nn: features
            },
            weights: {
              type: 'Neural Network',
              layers: '8->64->32->2'
            },
            rawPrediction: { u: u_raw, v: v_raw },
            smoothed: { u: uvSm.x, v: uvSm.y },
            clamped: { u, v },
            final: {
              xpix,
              ypix,
              screenW: ocv.width,
              screenH: ocv.height
            }
          });

          const dx = mousePos.x - xpix;
          const dy = mousePos.y - ypix;
          const distance = Math.sqrt(dx * dx + dy * dy);
          const errorPct = (distance / Math.min(ocv.width, ocv.height) * 100);

          errorHistoryRef.current.push({
            distance,
            mouseX: mousePos.x,
            mouseY: mousePos.y,
            pogX: xpix,
            pogY: ypix,
            headYaw: headAnglesNow.yaw,
            headPitch: headAnglesNow.pitch,
            timestamp: Date.now()
          });

          if (errorHistoryRef.current.length > 100) {
            errorHistoryRef.current.shift();
          }

          if (errorHistoryRef.current.length % 10 === 0) {
            console.log(
              '🎯 Accuracy:',
              'Distance:', distance.toFixed(0), 'px',
              '| Error:', errorPct.toFixed(1), '%',
              '| Mouse:', `(${mousePos.x.toFixed(0)}, ${mousePos.y.toFixed(0)})`,
              '| PoG:', `(${xpix.toFixed(0)}, ${ypix.toFixed(0)})`
            );
          }

          octx.beginPath();
          octx.arc(xpix, ypix, 8, 0, Math.PI * 2);
          octx.fillStyle = "rgba(255, 80, 80, 0.9)";
          octx.fill();
          octx.lineWidth = 3;
          octx.strokeStyle = "white";
          octx.stroke();

          if (isTestingRef.current && testPointIndexRef.current !== -1) {
            const points = buildGrid();
            const targetU = points[testPointIndexRef.current].u;
            const targetV = points[testPointIndexRef.current].v;

            const targetX = targetU * ocv.width;
            const targetY = targetV * ocv.height;

            octx.beginPath();
            octx.moveTo(targetX, targetY);
            octx.lineTo(xpix, ypix);
            octx.strokeStyle = "rgba(0, 255, 0, 0.6)";
            octx.lineWidth = 2;
            octx.setLineDash([5, 5]);
            octx.stroke();
            octx.setLineDash([]);

            const dx = targetX - xpix;
            const dy = targetY - ypix;
            const liveDist = Math.sqrt(dx * dx + dy * dy);

            octx.fillStyle = "rgba(0, 0, 0, 0.7)";
            octx.fillRect(targetX + 20, targetY - 20, 120, 30);
            octx.fillStyle = "#00FF00";
            octx.font = "bold 14px Arial";
            octx.fillText(`Error: ${liveDist.toFixed(0)} px`, targetX + 25, targetY);       
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

    const ocv = overlayRef.current;
    if (ocv && eyeCalibRef.current.active) {
      const octx = ocv.getContext("2d");
      octx.clearRect(0, 0, ocv.width, ocv.height);

      const currentStage = calibStageRef.current;

      if (currentStage === 'pursuit' || currentStage === 'anchor') {
        octx.beginPath();
        const pathPoints = 300;
        const totalTime = 30;

        for (let i = 0; i <= pathPoints; i++) {
          const t = (i / pathPoints) * totalTime;
          const coords = getLissajousCoords(t);
          const pathX = coords.u * ocv.width;
          const pathY = coords.v * ocv.height;

          if (i === 0) {
            octx.moveTo(pathX, pathY);
          } else {
            octx.lineTo(pathX, pathY);
          }
        }

        octx.strokeStyle = "rgba(255, 255, 255, 0.25)";
        octx.lineWidth = 2;
        octx.stroke();

        // const markerTimes = [0, 5, 10, 15, 20];
        // markerTimes.forEach(t => {
        //   const coords = getLissajousCoords(t);
        //   octx.beginPath();
        //   octx.arc(coords.u * ocv.width, coords.v * ocv.height, 4, 0, Math.PI * 2);
        //   octx.fillStyle = "rgba(255, 255, 255, 0.3)";
        //   octx.fill();
        // });
      }
      const px = calibDotRef.current.u * ocv.width;
      const py = calibDotRef.current.v * ocv.height;

      octx.beginPath()
      octx.arc(px, py, 18, 0, Math.PI * 2);
      octx.fillStyle = currentStage === 'anchor' ? "#FFD700" :
                        currentStage === 'capturing-baseline' ? "#FFA500" : "#00FF00";
      octx.fill();
      octx.strokeStyle = "white";
      octx.lineWidth = 3;
      octx.stroke();

      if (currentStage === 'anchor') {
        const pulseSize = 18 + Math.sin(Date.now() / 200) * 4;
        octx.beginPath();
        octx.arc(px, py, pulseSize, 0, Math.PI * 2);
        octx.strokeStyle = "rgba(255, 215, 0, 0.5)";
        octx.lineWidth = 2;
        octx.stroke();
      }

      if (currentStage === 'pursuit' && calibStartTimeRef.current) {
        const elapsed = (Date.now() - calibStartTimeRef.current) / 1000;
        const progress = Math.min(1, elapsed / 30);

        octx.fillStyle = "rgba(0, 0, 0, 0.6)";
        octx.fillRect(ocv.width / 2 - 150, 130, 300, 20);

        octx.fillStyle = "#4CAF50";
        octx.fillRect(ocv.width / 2 - 150, 130, 300 * progress, 20);

        octx.strokeStyle = "white";
        octx.lineWidth = 2;
        octx.strokeRect(ocv.width / 2 - 150, 130, 300, 20);

        octx.fillStyle = "#ccc";
        octx.font = "14px Arial";
        octx.textAlign = "center";
        octx.fillText(`Samples: ${trainingDataRef.current.length}`, ocv.width / 2, 170);
      }

      octx.fillStyle = "rgba(0, 0, 0, 0.7)";
      octx.fillRect(ocv.width / 2 - 200, 50, 400, 50);
      octx.fillStyle = "white";
      octx.font = "bold 18px Arial";
      octx.textAlign = "center";
      octx.fillText(calibInstruction, ocv.width / 2, 82);
    }
  };

  const showAccuracyStats = () => {
    const errors = errorHistoryRef.current;
    if (errors.length === 0) {
      console.log("❌ No accuracy data yet - move your mouse around the screen!");
      return;
    }

    const distances = errors.map(e => e.distance);
    const mean = distances.reduce((a, b) => a + b, 0) / distances.length;
    const variance = distances.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / distances.length;
    const stdDev = Math.sqrt(variance);
    const min = Math.min(...distances);
    const max = Math.max(...distances);
    const sorted = distances.slice().sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];

    console.log('\n╔════════════════════════════════════════╗');
    console.log('║        ACCURACY STATISTICS             ║');
    console.log('╚════════════════════════════════════════╝');
    console.log('Samples:', errors.length);
    console.log('Mean error:', mean.toFixed(1), 'px');
    console.log('Median error:', median.toFixed(1), 'px');
    console.log('Std deviation:', stdDev.toFixed(1), 'px');
    console.log('Min error:', min.toFixed(1), 'px');
    console.log('Max error:', max.toFixed(1), 'px');
    
    const screenSize = overlayRef.current ? Math.min(overlayRef.current.width, overlayRef.current.height) : 1000;
    console.log('Mean error %:', (mean / screenSize * 100).toFixed(2), '%');
    console.log('═══════════════════════════════════════════');

    const ocv = overlayRef.current;
    if (!ocv) return;

    const topHalf = errors.filter(e => e.pogY < ocv.height / 2);
    const bottomHalf = errors.filter(e => e.pogY >= ocv.height / 2);
    const leftHalf = errors.filter(e => e.pogX < ocv.width / 2);
    const rightHalf = errors.filter(e => e.pogX >= ocv.width / 2);

    console.log('\nREGIONAL BREAKDOWN:');
    
    if (topHalf.length > 0) {
      const topMean = topHalf.reduce((a, b) => a + b.distance, 0) / topHalf.length;
      console.log(`  Top half (${topHalf.length} samples):`, topMean.toFixed(1), 'px');
    }

    if (bottomHalf.length > 0) {
      const bottomMean = bottomHalf.reduce((a, b) => a + b.distance, 0) / bottomHalf.length;
      console.log(`  Bottom half (${bottomHalf.length} samples):`, bottomMean.toFixed(1), 'px');
    }

    if (leftHalf.length > 0) {
      const leftMean = leftHalf.reduce((a, b) => a + b.distance, 0) / leftHalf.length;
      console.log(`  Left half (${leftHalf.length} samples):`, leftMean.toFixed(1), 'px');
    }

    if (rightHalf.length > 0) {
      const rightMean = rightHalf.reduce((a, b) => a + b.distance, 0) / rightHalf.length;
      console.log(`  Right half (${rightHalf.length} samples):`, rightMean.toFixed(1), 'px');
    }

    const straightPose = errors.filter(e => Math.abs(e.headYaw) < 5 && Math.abs(e.headPitch) < 5);
    const turnedHead = errors.filter(e => Math.abs(e.headYaw) > 5 || Math.abs(e.headPitch) > 5);

    console.log('\nHEAD POSE BREAKDOWN:');
    if (straightPose.length > 0) {
      const straightMean = straightPose.reduce((a, b) => a + b.distance, 0) / straightPose.length;
      console.log(`  Straight head (${straightPose.length} samples):`, straightMean.toFixed(1), 'px');
    }
    
    if (turnedHead.length > 0) {
      const turnedMean = turnedHead.reduce((a, b) => a + b.distance, 0) / turnedHead.length;
      console.log(`  Turned head (${turnedHead.length} samples):`, turnedMean.toFixed(1), 'px');
    }

    console.log('═══════════════════════════════════════════\n');
  };

  const showModelWeights = () => {
    if (!xyCalibRef.current.model) {
      console.log("❌ No model fitted yet!");
      return;
    }

    const { wV, wU } = xyCalibRef.current.model;
    
    console.log('\n╔════════════════════════════════════════╗');
    console.log('║       VERTICAL MODEL WEIGHTS           ║');
    console.log('╚════════════════════════════════════════╝');
    
    console.log('\n🔹 BASE FEATURES:');
    console.log(`  w[0] L_y:        ${wV[0]?.toFixed(4)} ${Math.abs(wV[0]) > 0.3 ? '⭐ STRONG' : ''}`);
    console.log(`  w[1] R_y:        ${wV[1]?.toFixed(4)} ${Math.abs(wV[1]) > 0.3 ? '⭐ STRONG' : ''}`);
    console.log(`  w[2] pitch:      ${wV[2]?.toFixed(4)} ${Math.abs(wV[2]) > 0.3 ? '⭐ STRONG' : ''}`);
    console.log(`  w[3] aperture_L: ${wV[3]?.toFixed(4)} ${Math.abs(wV[3]) > 0.3 ? '⭐ STRONG' : ''}`);
    console.log(`  w[4] aperture_R: ${wV[4]?.toFixed(4)} ${Math.abs(wV[4]) > 0.3 ? '⭐ STRONG' : ''}`);
    console.log(`  w[5] bias:       ${wV[5]?.toFixed(4)}`);
    
    if (wV.length > 6) {
      console.log('\n🔹 QUADRATIC TERMS:');
      for (let i = 6; i < wV.length; i++) {
        if (Math.abs(wV[i]) > 0.2) {
          console.log(`  w[${i}]: ${wV[i].toFixed(4)} ⭐`);
        }
      }
    }
    
    console.log('\n📊 FEATURE IMPORTANCE:');
    const baseWeights = wV.slice(0, 6);
    const maxWeight = Math.max(...baseWeights.map(Math.abs));
    console.log(`  Iris Y:    ${((Math.abs(wV[0]) + Math.abs(wV[1])) / (2 * maxWeight) * 100).toFixed(1)}%`);
    console.log(`  Head pitch: ${(Math.abs(wV[2]) / maxWeight * 100).toFixed(1)}%`);
    console.log(`  Aperture:   ${((Math.abs(wV[3]) + Math.abs(wV[4])) / (2 * maxWeight) * 100).toFixed(1)}%`);
    
    console.log('═══════════════════════════════════════════\n');
  };

  const renderIrisDebugPanel = () => {
    if (!irisDebugInfo.rectified.left) return null;

    return (
      <div
        style={{
          position: "absolute",
          top: 240,
          left: 20,
          zIndex: 15,
          background: "rgba(0, 0, 0, 0.85)",
          color: "#fff",
          padding: "12px",
          borderRadius: "8px",
          fontFamily: "monospace",
          fontSize: "12px",
          minWidth: 320,
          maxWidth: 400,
          border: "2px solid #4CAF50"
        }}
      >
        <div style={{ 
          fontSize: "14px", 
          fontWeight: "bold", 
          marginBottom: "10px",
          color: "#4CAF50",
          borderBottom: "1px solid #4CAF50",
          paddingBottom: "5px"
        }}>
          📊 MediaPipe Iris Debug
        </div>

        {/* Raw MediaPipe Positions */}
        {irisDebugInfo.raw && (
          <div style={{ marginBottom: "8px" }}>
            <div style={{ color: "#FFD700", fontWeight: "bold" }}>Raw (pixel coords):</div>
            <div style={{ marginLeft: "10px" }}>
              <div>Left:  x={irisDebugInfo.raw.left.x.toFixed(1)} y={irisDebugInfo.raw.left.y.toFixed(1)}</div>
              <div>Right: x={irisDebugInfo.raw.right.x.toFixed(1)} y={irisDebugInfo.raw.right.y.toFixed(1)}</div>
            </div>
          </div>
        )}

        {/* Rectified Offsets */}
        <div style={{ marginBottom: "8px" }}>
          <div style={{ color: "#00BCD4", fontWeight: "bold" }}>Rectified (normalized):</div>
          <div style={{ marginLeft: "10px" }}>
            <div>Left:  x={irisDebugInfo.rectified.left.x.toFixed(3)} y={irisDebugInfo.rectified.left.y.toFixed(3)}</div>
            <div>Right: x={irisDebugInfo.rectified.right.x.toFixed(3)} y={irisDebugInfo.rectified.right.y.toFixed(3)}</div>
          </div>
        </div>

        {/* Smoothed Values */}
        <div style={{ marginBottom: "8px" }}>
          <div style={{ color: "#FF9800", fontWeight: "bold" }}>Smoothed (EMA α=0.5):</div>
          <div style={{ marginLeft: "10px" }}>
            <div>Left:  x={irisDebugInfo.smoothed.left.x.toFixed(3)} y={irisDebugInfo.smoothed.left.y.toFixed(3)}</div>
            <div>Right: x={irisDebugInfo.smoothed.right.x.toFixed(3)} y={irisDebugInfo.smoothed.right.y.toFixed(3)}</div>
          </div>
        </div>

        {/* Zero-centered */}
        <div style={{ marginBottom: "8px" }}>
          <div style={{ color: "#E91E63", fontWeight: "bold" }}>Zero-centered (offset removed):</div>
          <div style={{ marginLeft: "10px" }}>
            <div>Left:  x={irisDebugInfo.zeroed.left.x.toFixed(3)} y={irisDebugInfo.zeroed.left.y.toFixed(3)}</div>
            <div>Right: x={irisDebugInfo.zeroed.right.x.toFixed(3)} y={irisDebugInfo.zeroed.right.y.toFixed(3)}</div>
          </div>
        </div>

        {/* Visual indicators for Y movement */}
        <div style={{ marginTop: "10px", paddingTop: "10px", borderTop: "1px solid #555" }}>
          <div style={{ fontWeight: "bold", marginBottom: "5px" }}>Y-axis indicators:</div>
          <div style={{ marginLeft: "10px" }}>
            <div>
              Left Y: {
                irisDebugInfo.rectified.left.y > 0.05 ? '⬆️ UP' :
                irisDebugInfo.rectified.left.y < -0.05 ? '⬇️ DOWN' :
                '➡️ CENTER'
              } ({irisDebugInfo.rectified.left.y.toFixed(3)})
            </div>
            <div>
              Right Y: {
                irisDebugInfo.rectified.right.y > 0.05 ? '⬆️ UP' :
                irisDebugInfo.rectified.right.y < -0.05 ? '⬇️ DOWN' :
                '➡️ CENTER'
              } ({irisDebugInfo.rectified.right.y.toFixed(3)})
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderApertureDebugPanel = () => {
    if (!apertureDebugInfo.left) return null;

    return (
      <div
        style={{
          position: "absolute",
          top: 460,
          left: 20,
          zIndex: 15,
          background: "rgba(0, 0, 0, 0.85)",
          color: "#fff",
          padding: "12px",
          borderRadius: "8px",
          fontFamily: "monospace",
          fontSize: "12px",
          minWidth: 320,
          border: "2px solid #9C27B0"
        }}
      >
        <div style={{ 
          fontSize: "14px", 
          fontWeight: "bold", 
          marginBottom: "10px",
          color: "#9C27B0",
          borderBottom: "1px solid #9C27B0",
          paddingBottom: "5px"
        }}>
          👁️ Eyelid Aperture Debug
        </div>

        <div style={{ marginBottom: "8px" }}>
          <div style={{ color: "#FFD700", fontWeight: "bold" }}>Raw Aperture (pixels):</div>
          <div style={{ marginLeft: "10px" }}>
            <div>Left:  {apertureDebugInfo.left} px</div>
            <div>Right: {apertureDebugInfo.right} px</div>
          </div>
        </div>

        <div style={{ marginBottom: "8px" }}>
          <div style={{ color: "#00BCD4", fontWeight: "bold" }}>Normalized (ratio to eye width):</div>
          <div style={{ marginLeft: "10px" }}>
            <div>Left:  {apertureDebugInfo.leftNorm}</div>
            <div>Right: {apertureDebugInfo.rightNorm}</div>
          </div>
        </div>

        <div style={{ marginTop: "10px", paddingTop: "10px", borderTop: "1px solid #555" }}>
          <div style={{ fontWeight: "bold", marginBottom: "5px" }}>Status:</div>
          <div style={{ marginLeft: "10px" }}>
            {parseFloat(apertureDebugInfo.leftNorm) > 0.35 ? '👀 Eyes wide open' :
            parseFloat(apertureDebugInfo.leftNorm) < 0.25 ? '😑 Eyes squinting' :
            '👁️ Normal aperture'}
          </div>
        </div>
      </div>
    );
  };

  const renderPoGTracePanel = () => {
    if (!pogTraceInfo.input || !isCalibrated || eyeCalibRef.current.active) return null;

    const { input, features, weights, rawPrediction, smoothed, clamped, final } = pogTraceInfo;

    const calcV = (feats, w) => {
      let sum = 0;
      for (let i = 0; i < Math.min(feats.length, w.length); i++) {
        sum += feats[i] * w[i];
      }
      return sum;
    };

    const vManual = weights.wV ? calcV(features.vertical, weights.wV) : null;
    const uManual = weights.wU ? calcV(features.horizontal, weights.wU) : null;

    return (
      <div
        style={{
          position: "absolute",
          top: 20,
          right: 20,
          zIndex: 15,
          background: "rgba(0, 0, 0, 0.9)",
          color: "#fff",
          padding: "12px",
          borderRadius: "8px",
          fontFamily: "monospace",
          fontSize: "11px",
          minWidth: 380,
          maxWidth: 450,
          border: "2px solid #FF9800",
          maxHeight: "calc(100vh - 40px)",
          overflowY: "auto"
        }}
      >
        <div
          style={{
            fontSize: "14px",
            fontWeight: "bold",
            marginBottom: "10px",
            color: "#FF9800",
            borderBottom: "1px solid #FF9800",
            paddingBottom: "5px"
          }}
        >
          PoG Computation Trace
        </div>
        
        <div style={{ marginBottom: "10px", background: "rgba(255, 255, 255, 0.05)", padding: "8px", borderRadius: "4px" }}>
          <div style={{ color: "#4CAF50", fontWeight: "bold", marginBottom: "4px" }}>INPUT (Zero-centered)</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px", fontSize: "10px" }}>
            <div>L_x: {input.zL_x.toFixed(3)}</div>
            <div>R_x: {input.zR_x.toFixed(3)}</div>
            <div style={{ color: input.zL_y > 0.05 ? "#ff6b6b" : input.zL_y < -0.05 ? "#4dabf7" : "#fff" }}>
              L_y: {input.zL_y.toFixed(3)} {input.zL_y > 0.05 ? "UP" : input.zL_y < -0.05 ? "DOWN" : "MIDDLE"}
            </div>
            <div style={{ color: input.zR_y > 0.05 ? "#ff6b6b" : input.zR_y < -0.05 ? "#4dabf7" : "#fff" }}>
              R_y: {input.zR_y.toFixed(3)} {input.zR_y > 0.05 ? "UP" : input.zR_y < -0.05 ? "DOWN" : "MIDDLE" }
            </div>
          </div>
          <div style={{ marginTop: "4px", fontSize: "10px" }}>
            Head: yaw={input.headYaw.toFixed(1)} pitch={input.headPitch.toFixed(1)}
          </div>

          <div style={{ marginTop: "4px", fontSize: "10px", color: "#9c27b0" }}>
            Aperture: L={apertureDebugInfo.leftNorm} R={apertureDebugInfo.rightNorm}
          </div>

          <div style={{ marginBottom: "10px", background: "rgba(255, 255, 255, 0.05)", padding: "8px", borderRadius: "4px" }}>
            <div style={{ color: "#FFD700", fontWeight: "bold", marginBottom: "4px"}}>
              BASELINE (Center Dot)
            </div>
            <div style={{ fontSize: "10px", lineHeight: "1.4" }}>
              <div>Left: x={pogTraceInfo.baseline?.left.x.toFixed(3)}, y={pogTraceInfo.baseline?.left.y.toFixed(3)}</div>
              <div>Right: x={pogTraceInfo.baseline?.right.x.toFixed(3)}, y={pogTraceInfo.baseline?.right.y.toFixed(3)}</div>
            </div>
          </div>

          <div style={{ marginBottom: "10px", background: "rgba(255, 255, 255, 0.05)", padding: "8px", borderRadius: "4px" }}>
            <div style={{ color: "#00BCD4", fontWeight: "bold", marginBottom: "4px" }}>
              🔧 RECTIFIED (Before Zero-centering)
            </div>
            <div style={{ fontSize: "10px", lineHeight: "1.4" }}>
              <div>Left:  x={pogTraceInfo.rectified?.left.x.toFixed(3)}, y={pogTraceInfo.rectified?.left.y.toFixed(3)}</div>
              <div>Right: x={pogTraceInfo.rectified?.right.x.toFixed(3)}, y={pogTraceInfo.rectified?.right.y.toFixed(3)}</div>
            </div>
          </div>

          {(input.zL_y * input.zR_y < 0 && Math.abs(input.zL_y - input.zR_y) > 0.1) && (
            <div 
              style={{
                marginTop: "6px",
                padding: "4px",
                background: "rgba(255, 0, 0, 0.2)",
                border: "1px solid #ff6b6b",
                borderRadius: "3px",
                fontSize: "10px"
            }}>
              Vertical vergence detected! (eye disagree)
            </div>
          )}  
        </div>

        <div style={{ marginBottom: "10px", background: "rgba(255, 255, 255, 0.05)", padding: "8px", borderRadius: "4px" }}>
          <div style={{ color: "#2196F3", fontWeight: "bold", marginBottom: "4px" }}>NN INPUT FEATURES</div>
          <div style={{ fontSize: "10px", lineHeight: "1.4" }}>
            {/* Use the new 'nn' key and check for its existence before mapping */}
            {features && features.nn ? (
              features.nn.map((f, i) => (
                <div key={i}>f[{i}] = {f.toFixed(4)}</div>
              ))
            ) : (
              <div style={{ color: '#aaa' }}>Waiting for model inference...</div>
            )}
          </div>
        </div>

        <div style={{ marginBottom: "10px", background: "rgba(255, 255, 255, 0.05)", padding: "8px", borderRadius: "4px" }}>
          <div style={{ color: "#9C27B0", fontWeight: "bold", marginBottom: "4px" }}>🧠 MODEL STATUS</div>
          <div style={{ fontSize: "10px", lineHeight: "1.4" }}>
            <div>Type: {weights.type}</div>
            <div>Architecture: {weights.layers}</div>
            <div style={{ color: "#00FF00", marginTop: "4px" }}>Active: ✅ Neural Network Bridge</div>
          </div>
        </div>

        <div style={{ marginBottom: "8px", background: "rgba(255,255,255,0.05)", padding: "8px", borderRadius: "4px" }}>
          <div style={{ color: "#00BCD4", fontWeight: "bold", marginBottom: "4px" }}>⚙️ PIPELINE</div>
          <div style={{ fontSize: "10px", lineHeight: "1.6" }}>
            <div>1. Raw: v = {rawPrediction.v.toFixed(4)}</div>
            <div style={{ color: smoothed.v !== rawPrediction.v ? "#ffeb3b" : "#fff" }}>
              2. EMA (α=0.3): v = {smoothed.v.toFixed(4)}
              {smoothed.v !== rawPrediction.v && (
                <span style={{ color: "#ff6b6b" }}> Δ={Math.abs(smoothed.v - rawPrediction.v).toFixed(4)}</span>
              )}
            </div>
            <div style={{ color: clamped.v !== smoothed.v ? "#ff6b6b" : "#fff" }}>
              3. Clamp [0,1]: v = {clamped.v.toFixed(4)}
              {clamped.v !== smoothed.v && " ✂️ CLAMPED!"}
            </div>
          </div>
        </div>

        <div style={{ background: "rgba(255, 255, 255, 0.1)", padding: "8px", borderRadius: "4px" }}>
          <div style={{ color: "#4CAF50", fontWeight: "bold", marginBottom: "4px" }}>FINAL OUTPUT</div>
          <div style={{ fontSize: "11px", lineHeight: "1.6" }}>
            <div>Screen: {final.screenW} x {final.screenH}</div>
            <div style={{ color: "#ffeb3b", fontWeight: "bold" }}>
              PoG: ({final.xpix.toFixed(0)}, {final.ypix.toFixed(0)}) px
            </div>
            <div>Normalized: u={clamped.u.toFixed(3)}, v={clamped.v.toFixed(3)}</div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      <header className="App-header">
        <div style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: "0",
          right: "0",
          textAlign: "center",
          zIndex: 9,
          width: 640,
          height: 480,
          overflow: "hidden",
          borderRadius: "10px",
          border: "2px solid #333",
        }}>
          <Webcam
            ref={webcamRef}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              transform: "scale(2.0)",
              transformOrigin: "center",
            }}
          />

          <canvas 
            ref={canvasRef}
            style={{
              position:"absolute",
              left:0,
              right:0,
              width: "100%",
              height: "100%",
              objectFit: "cover",
              transform: "scale(2.0)",
              transformOrigin: "center"
            }}
          />
        </div>

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
          onClick={startEyeCalibration}
          disabled={eyeCalibRef.current.active}
          style={{
            position:"absolute",
            top:20,
            left:20,
            zIndex:12,
            padding:"10px 20px",
            backgroundColor: eyeCalibRef.current.active ? "#ff9800" : "#673ab7",
            color:"white",
            border:"none",
            borderRadius:"5px",
            cursor: eyeCalibRef.current.active ? "default" : "pointer"
          }}
        >
          {eyeCalibRef.current.active ? "Calibrating... Click center dot first!" : "Start Calibration (45 samples)"}
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
            <option value={3}>Cubic</option>
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

        <button
          onClick={showAccuracyStats}
          disabled={!isCalibrated || eyeCalibRef.current.active}
          style={{
            position:"absolute",
            top:180,
            left:20,
            zIndex:10,
            padding:"10px 20px",
            backgroundColor: (!isCalibrated || eyeCalibRef.current.active) ? "gray" : "#4CAF50",
            color:"white",
            border:"none",
            borderRadius:"5px",
            cursor: (!isCalibrated || eyeCalibRef.current.active) ? "default" : "pointer"
          }}
        >
          📊 Show Accuracy Stats
        </button>
        <button
          onClick={showModelWeights}
          disabled={!isCalibrated || eyeCalibRef.current.active}
          style={{
            position:"absolute",
            top:230,
            left:20,
            zIndex:10,
            padding:"10px 20px",
            backgroundColor: (!isCalibrated || eyeCalibRef.current.active) ? "gray" : "#9C27B0",
            color:"white",
            border:"none",
            borderRadius:"5px",
            cursor: (!isCalibrated || eyeCalibRef.current.active) ? "default" : "pointer"
          }}
        >
          🔍 Show Model Weights
        </button>
        <button
          onClick={runAccuracyTest}
          disabled={!isCalibrated || isTesting}
          style={{
            position: "absolute",
            top: 280,
            left: 20,
            zIndex: 10,
            padding: "10px 20px",
            backgroundColor: (!isCalibrated || isTesting) ? "gray" : "#E91E63",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer"
          }}
        >
          {isTesting ? "Testing..." : "Run Accuracy Test"}
        </button>

        {isTesting && testPointIndex !== -1 && (
          <div style={{
            position: 'fixed',
            left: `${buildGrid()[testPointIndex].u * 100}%`,
            top: `${buildGrid()[testPointIndex].v * 100}%`,
            width: '30px',
            height: '30px',
            backgroundColor: '#00FF00',
            borderRadius: '50%',
            border: '4px solid white',
            transform: 'translate(-50%, -50%)',
            zIndex: 100,
            boxShadow: '0 0 20px #00FF00'
          }} />
        )}

        {testSummary && testSummary.points && (
          <div style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.92)', 
            zIndex: 100,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: 'Arial, sans-serif'
          }}>
            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
                <h2 style={{ color: '#FF9800', margin: '0 0 10px 0' }}>Gaze Accuracy Results</h2>
                <div style={{ display: 'flex', gap: '20px', justifyContent: 'center' }}>
                    <p style={{ color: '#ccc' }}>Avg Error: <span style={{color: '#00FF00', fontSize: '20px', fontWeight: 'bold'}}>{testSummary.avgPixelError}px</span></p>
                    <p style={{ color: '#ccc' }}>Screen Error: <span style={{color: '#00FF00', fontSize: '20px', fontWeight: 'bold'}}>{testSummary.percentageError}%</span></p>
                </div>
            </div>

            {/* The Heatmap Visualizer Box */}
            <div style={{
              position: 'relative',
              width: '85vw',
              height: '65vh',
              background: '#000',
              border: '2px solid #E91E63', 
              borderRadius: '12px',
              boxShadow: '0 0 30px rgba(233, 30, 99, 0.2)'
            }}>
              {testSummary.points.map((pt, idx) => {
                const intensity = Math.min(1, pt.error / 150); 
                const color = `rgb(${intensity * 255}, ${255 - intensity * 255}, 0)`;
                
                return (
                  <div key={idx} style={{
                    position: 'absolute',
                    left: `${pt.u * 100}%`,
                    top: `${pt.v * 100}%`,
                    transform: 'translate(-50%, -50%)',
                  }}>
                    <div style={{
                      width: '40px',
                      height: '40px',
                      backgroundColor: color,
                      borderRadius: '50%',
                      opacity: 0.8,
                      boxShadow: `0 0 15px ${color}`,
                    }} />
                    <span style={{ fontSize: '11px', color: '#fff', display: 'block', textAlign: 'center', marginTop: '5px' }}>
                      {pt.error.toFixed(0)}px
                    </span>
                  </div>
                );
              })}
            </div>

            <button 
              onClick={() => setTestSummary(null)}
              style={{
                marginTop: '30px',
                padding: '12px 40px',
                backgroundColor: '#E91E63', 
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                fontWeight: 'bold'
              }}
            >
              Done
            </button>
          </div>
        )}
        {/* {renderIrisDebugPanel()}
        {renderApertureDebugPanel()}  */}
        {renderPoGTracePanel()}
      </header>
    </div>
  );
}

export default App;
