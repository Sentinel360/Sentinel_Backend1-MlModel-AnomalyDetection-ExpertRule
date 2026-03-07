const functions = require("firebase-functions");
const admin = require("firebase-admin");
const axios = require("axios");
const { GoogleAuth } = require("google-auth-library");
const { defineString } = require("firebase-functions/params");

admin.initializeApp();

// Set once at deploy time: firebase deploy will prompt for ML_API_URL value.
const ML_API_URL_PARAM = defineString("ML_API_URL");
const googleAuth = new GoogleAuth();
let cachedIdTokenClient = null;
let cachedMlApiUrl = "";

// Trip-level policy tuning (keeps instant risk separate from overall trip risk)
const HIGH_CONSECUTIVE_THRESHOLD = 3;
const HIGH_RATIO_THRESHOLD = 0.2;
const MEDIUM_RATIO_THRESHOLD = 0.1;
const MIN_WINDOWS_FOR_RATIO = 10;
const HIGH_COOLDOWN_MS = 60 * 1000;

function getMlApiUrl() {
  if (process.env.ML_API_URL) return process.env.ML_API_URL;
  if (!cachedMlApiUrl) cachedMlApiUrl = ML_API_URL_PARAM.value() || "";
  return cachedMlApiUrl;
}

if (!getMlApiUrl()) {
  console.warn("ML_API_URL is not set. Configure ML_API_URL param during deploy.");
}

function currentStateRef(tripId) {
  return admin.firestore()
    .collection("trips")
    .doc(tripId)
    .collection("current_state")
    .doc("latest");
}

async function getMlApiAuthHeaders() {
  const mlApiUrl = getMlApiUrl();
  if (!mlApiUrl) return {};
  if (!cachedIdTokenClient) {
    cachedIdTokenClient = await googleAuth.getIdTokenClient(mlApiUrl);
  }
  const headers = await cachedIdTokenClient.getRequestHeaders();
  return headers || {};
}

async function callMlApi(path, payload, timeoutMs = 5000) {
  const mlApiUrl = getMlApiUrl();
  if (!mlApiUrl) {
    throw new Error("ML_API_URL is not configured");
  }
  const headers = await getMlApiAuthHeaders();
  return axios.post(`${mlApiUrl}${path}`, payload, {
    timeout: timeoutMs,
    headers,
  });
}

function computeTripPolicy(previousState = {}, instantLevel, eventTsMs) {
  const prevPolicy = previousState.policy || {};
  const totalWindows = Number(prevPolicy.totalWindows || 0) + 1;
  const highWindows = Number(prevPolicy.highWindows || 0) + (instantLevel === "HIGH RISK" ? 1 : 0);
  const consecutiveHigh = instantLevel === "HIGH RISK"
    ? Number(prevPolicy.consecutiveHigh || 0) + 1
    : 0;
  const maxConsecutiveHigh = Math.max(
    Number(prevPolicy.maxConsecutiveHigh || 0),
    consecutiveHigh,
  );
  const highRatio = highWindows / totalWindows;

  const lastHighTimestamp = instantLevel === "HIGH RISK"
    ? eventTsMs
    : Number(prevPolicy.lastHighTimestamp || 0);
  const latchedHighUntil = instantLevel === "HIGH RISK"
    ? eventTsMs + HIGH_COOLDOWN_MS
    : Number(prevPolicy.latchedHighUntil || 0);

  const ratioGateOpen = totalWindows >= MIN_WINDOWS_FOR_RATIO;
  const sustainedUnsafe = consecutiveHigh >= HIGH_CONSECUTIVE_THRESHOLD;
  const frequentUnsafe = ratioGateOpen && highRatio >= HIGH_RATIO_THRESHOLD;
  const overallUnsafe = Boolean(prevPolicy.overallUnsafe) || sustainedUnsafe || frequentUnsafe;

  let overallLevel = "SAFE";
  if (overallUnsafe) {
    overallLevel = "HIGH RISK";
  } else if (consecutiveHigh >= 2 || (ratioGateOpen && highRatio >= MEDIUM_RATIO_THRESHOLD)) {
    overallLevel = "MEDIUM";
  }

  const reasons = [];
  if (sustainedUnsafe) reasons.push("Sustained high risk events");
  if (frequentUnsafe) reasons.push("Frequent high risk ratio");
  if (!reasons.length && overallLevel === "MEDIUM") reasons.push("Elevated risk trend");
  if (!reasons.length) reasons.push("No sustained unsafe trend");

  return {
    totalWindows,
    highWindows,
    highRatio: Number(highRatio.toFixed(4)),
    consecutiveHigh,
    maxConsecutiveHigh,
    lastHighTimestamp,
    latchedHighUntil,
    overallUnsafe,
    overallLevel,
    reason: reasons.join("; "),
  };
}

exports.processSensorData = functions.firestore
  .document("trips/{tripId}/sensor_data/{eventId}")
  .onCreate(async (snap, context) => {
    const tripId = context.params.tripId;
    const sensorData = snap.data() || {};

    console.log(`Processing sensor data for trip: ${tripId}`);

    if (!getMlApiUrl()) {
      await currentStateRef(tripId).set({
        riskScore: 0.0,
        riskLevel: "SAFE",
        riskColor: "green",
        explanation: "ML API URL missing - fallback state",
        updatedAt: admin.firestore.FieldValue.serverTimestamp(),
      }, { merge: true });
      return null;
    }

    try {
      const eventTsMs = Number(sensorData.timestamp || Date.now());
      const latestRef = currentStateRef(tripId);
      const latestSnap = await latestRef.get();
      const previousState = latestSnap.exists ? latestSnap.data() : {};

      const response = await callMlApi(
        "/predict",
        {
          trip_id: tripId,
          gps: sensorData.gps || {},
          acceleration: sensorData.acceleration || {},
          timestamp: eventTsMs,
          source: sensorData.source || "PHONE",
        },
        5000,
      );

      const result = response.data;
      const policy = computeTripPolicy(previousState, result.final_level, eventTsMs);

      await latestRef.set({
        riskScore: result.final_score,
        riskLevel: result.final_level,
        riskColor: result.final_color,
        // Trip-level status (not the same as instant event risk)
        overallRiskLevel: policy.overallLevel,
        overallUnsafe: policy.overallUnsafe,
        policy,
        activeSensor: result.active_sensor,
        explanation: result.explanation,
        modelVersion: result.model_version,
        components: result.components,
        actions: result.actions || [],
        updatedAt: admin.firestore.FieldValue.serverTimestamp(),
      }, { merge: true });

      if (result.final_level === "HIGH RISK") {
        await admin.firestore()
          .collection("trips")
          .doc(tripId)
          .collection("alerts")
          .add({
            timestamp: admin.firestore.FieldValue.serverTimestamp(),
            riskScore: result.final_score,
            reason: result.explanation,
            actions: result.actions || [],
            resolved: false,
          });
      }

      console.log(
        `Risk complete ${tripId}: instant=${result.final_level}, overall=${policy.overallLevel}, ` +
        `consecutiveHigh=${policy.consecutiveHigh}, highRatio=${policy.highRatio}`,
      );
      return null;
    } catch (error) {
      console.error("Error calling ML API:", error.message);
      await currentStateRef(tripId).set({
        riskScore: 0.0,
        riskLevel: "SAFE",
        riskColor: "green",
        explanation: "ML service unavailable - fallback",
        error: error.message,
        updatedAt: admin.firestore.FieldValue.serverTimestamp(),
      }, { merge: true });
      return null;
    }
  });

exports.onTripStart = functions.firestore
  .document("trips/{tripId}")
  .onCreate(async (snap, context) => {
    const tripId = context.params.tripId;
    const tripData = snap.data() || {};

    if (!getMlApiUrl()) return null;

    try {
      await callMlApi(
        "/trip/start",
        {
          trip_id: tripId,
          origin: tripData.origin || { lat: 0, lon: 0 },
          destination: tripData.destination || { lat: 0, lon: 0 },
        },
        5000,
      );
      console.log(`Trip monitoring initialized: ${tripId}`);
    } catch (error) {
      console.error("Error initializing trip on ML API:", error.message);
    }
    return null;
  });

exports.onTripEnd = functions.firestore
  .document("trips/{tripId}")
  .onUpdate(async (change, context) => {
    const tripId = context.params.tripId;
    const newData = change.after.data() || {};
    const oldData = change.before.data() || {};

    if (!(newData.status === "completed" && oldData.status !== "completed")) {
      return null;
    }

    if (!getMlApiUrl()) return null;

    try {
      const response = await callMlApi(
        "/trip/end",
        { trip_id: tripId },
        5000,
      );
      const latestSnap = await currentStateRef(tripId).get();
      const latest = latestSnap.exists ? latestSnap.data() : {};
      const policy = latest.policy || {};

      await admin.firestore()
        .collection("trips")
        .doc(tripId)
        .set(
          {
            summary: response.data.summary || null,
            riskSummary: {
              overallRiskLevel: latest.overallRiskLevel || "SAFE",
              overallUnsafe: Boolean(latest.overallUnsafe),
              totalWindows: Number(policy.totalWindows || 0),
              highWindows: Number(policy.highWindows || 0),
              highRatio: Number(policy.highRatio || 0),
              maxConsecutiveHigh: Number(policy.maxConsecutiveHigh || 0),
              policyReason: policy.reason || "No sustained unsafe trend",
            },
            processedAt: admin.firestore.FieldValue.serverTimestamp(),
          },
          { merge: true },
        );

      console.log(`Trip finalized: ${tripId}`);
    } catch (error) {
      console.error("Error finalizing trip on ML API:", error.message);
    }
    return null;
  });
