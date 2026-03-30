const functions = require("firebase-functions");
const admin = require("firebase-admin");
const axios = require("axios");
const { GoogleAuth } = require("google-auth-library");
admin.initializeApp();

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const googleAuth = new GoogleAuth();
let cachedIdTokenClient = null;

// Set to false in production to enable real SOS notifications
const TEST_MODE = true;

// Trip-level policy tuning (keeps instant risk separate from overall trip risk)
const HIGH_CONSECUTIVE_THRESHOLD = 3;
const HIGH_RATIO_THRESHOLD = 0.2;
const MEDIUM_RATIO_THRESHOLD = 0.1;
const MIN_WINDOWS_FOR_RATIO = 10;
const HIGH_COOLDOWN_MS = 60 * 1000;

// SOS escalation
const ESCALATION_DELAY_MS = 2 * 60 * 1000; // 2 minutes before auto-escalation

function getMlApiUrl() {
  return process.env.ML_API_URL || "";
}

if (!getMlApiUrl()) {
  console.warn("ML_API_URL is not set. Set it in cloud_functions/.env.sentinel360-final");
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function currentStateRef(tripId) {
  return admin.firestore()
    .collection("trips")
    .doc(tripId)
    .collection("current_state")
    .doc("latest");
}

function normalizeLatLon(value) {
  if (!value || typeof value !== "object") return { lat: 0, lon: 0 };
  const lat = value.lat ?? value.latitude ?? value._latitude ?? 0;
  const lon = value.lon ?? value.lng ?? value.longitude ?? value._longitude ?? 0;
  return { lat: Number(lat), lon: Number(lon) };
}

async function getMlApiAuthHeaders() {
  const mlApiUrl = getMlApiUrl();
  if (!mlApiUrl) return {};
  try {
    if (!cachedIdTokenClient) {
      cachedIdTokenClient = await googleAuth.getIdTokenClient(mlApiUrl);
    }
    const headers = await cachedIdTokenClient.getRequestHeaders();
    return headers || {};
  } catch (err) {
    // Running locally or in emulator — skip IAM auth
    console.warn("Could not obtain ID token (expected in emulator):", err.message);
    return {};
  }
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

// ---------------------------------------------------------------------------
// Trip-level policy computation
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 1. processSensorData — Firestore trigger on sensor_data write
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 2. onTripStart — Initialize ML API monitoring + route fetch fallback
// ---------------------------------------------------------------------------

exports.onTripStart = functions.firestore
  .document("trips/{tripId}")
  .onCreate(async (snap, context) => {
    const tripId = context.params.tripId;
    const tripData = snap.data() || {};

    if (!getMlApiUrl()) return null;

    try {
      const origin = normalizeLatLon(tripData.origin ?? tripData.originGeo ?? tripData.startLocation);
      const destination = normalizeLatLon(tripData.destination);

      await callMlApi(
        "/trip/start",
        {
          trip_id: tripId,
          origin,
          destination,
        },
        5000,
      );
      console.log(`Trip monitoring initialized: ${tripId}`);

      // Route fetch fallback — if the mobile app didn't store a polyline,
      // fetch one server-side using Google Directions API
      const mapsKey = process.env.GOOGLE_MAPS_API_KEY || "";
      if (
        mapsKey &&
        origin.lat !== 0 && destination.lat !== 0 &&
        !(tripData.routePolyline && tripData.routePolyline.length > 0)
      ) {
        try {
          const dirUrl = `https://maps.googleapis.com/maps/api/directions/json` +
            `?origin=${origin.lat},${origin.lon}` +
            `&destination=${destination.lat},${destination.lon}` +
            `&key=${mapsKey}`;
          const dirResp = await axios.get(dirUrl, { timeout: 5000 });
          const routes = dirResp.data.routes || [];
          if (routes.length > 0 && routes[0].overview_polyline) {
            // Store the encoded polyline so the mobile app can render it
            await admin.firestore()
              .collection("trips")
              .doc(tripId)
              .set(
                {
                  encodedPolyline: routes[0].overview_polyline.points,
                  routeFetchedBy: "cloud_function",
                },
                { merge: true },
              );
            console.log(`Route polyline stored for trip: ${tripId}`);
          }
        } catch (routeErr) {
          console.warn("Route fetch fallback failed:", routeErr.message);
        }
      }
    } catch (error) {
      console.error("Error initializing trip on ML API:", error.message);
    }
    return null;
  });

// ---------------------------------------------------------------------------
// 3. onTripEnd — Finalize trip with risk summary
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 4. onEmergencyAlert — SOS button handler
// ---------------------------------------------------------------------------

exports.onEmergencyAlert = functions.runWith({ timeoutSeconds: 300, memory: "256MB" }).firestore
  .document("emergency_alerts/{alertId}")
  .onCreate(async (snap, context) => {
    const alertId = context.params.alertId;
    const alertData = snap.data() || {};
    const userId = alertData.userId || "";
    const tripId = alertData.tripId || "";
    const location = alertData.location || {};

    console.log(`Emergency alert received: ${alertId}, user: ${userId}, trip: ${tripId}`);

    try {
      // Mark alert as processing
      await snap.ref.set({ status: "processing", processedAt: admin.firestore.FieldValue.serverTimestamp() }, { merge: true });

      // Fetch user profile for emergency contacts
      let userDoc = {};
      if (userId) {
        const userSnap = await admin.firestore().collection("users").doc(userId).get();
        userDoc = userSnap.exists ? userSnap.data() : {};
      }

      const emergencyContacts = userDoc.emergencyContacts || [];
      const userName = userDoc.displayName || userDoc.name || "A Sentinel360 user";

      // Build notification payload
      const lat = location.lat ?? location.latitude ?? location._latitude ?? 0;
      const lon = location.lon ?? location.lng ?? location.longitude ?? location._longitude ?? 0;
      const mapsLink = `https://maps.google.com/?q=${lat},${lon}`;
      const notificationBody = `${userName} triggered an SOS alert. Location: ${mapsLink}`;

      if (TEST_MODE) {
        console.log(`[TEST MODE] Would notify ${emergencyContacts.length} contacts: ${notificationBody}`);
        console.log(`[TEST MODE] Emergency contacts:`, JSON.stringify(emergencyContacts));
      } else {
        // Send FCM push to emergency contacts who have the app
        for (const contact of emergencyContacts) {
          if (contact.fcmToken) {
            try {
              await admin.messaging().send({
                token: contact.fcmToken,
                notification: {
                  title: "SOS Alert",
                  body: notificationBody,
                },
                data: {
                  type: "SOS",
                  alertId,
                  tripId,
                  lat: String(lat),
                  lon: String(lon),
                },
              });
            } catch (fcmErr) {
              console.warn(`FCM send failed for contact ${contact.name}:`, fcmErr.message);
            }
          }
        }
      }

      // Write admin notification
      await admin.firestore().collection("admin_notifications").add({
        type: "SOS",
        alertId,
        userId,
        tripId,
        userName,
        location: { lat, lon },
        mapsLink,
        emergencyContactCount: emergencyContacts.length,
        status: "pending_review",
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
      });

      // Create escalation timer — if not resolved within ESCALATION_DELAY_MS,
      // onEscalationTimerCreate will handle further escalation
      await admin.firestore().collection("escalation_timers").add({
        alertId,
        userId,
        tripId,
        escalateAt: new Date(Date.now() + ESCALATION_DELAY_MS),
        status: "pending",
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
      });

      // Update alert status
      await snap.ref.set({ status: "contacts_notified" }, { merge: true });

      console.log(`Emergency alert processed: ${alertId}`);
    } catch (error) {
      console.error("Error processing emergency alert:", error.message);
      await snap.ref.set({ status: "error", error: error.message }, { merge: true });
    }
    return null;
  });

// ---------------------------------------------------------------------------
// 5. onEscalationTimerCreate — Auto-escalation after timeout
// ---------------------------------------------------------------------------

exports.onEscalationTimerCreate = functions.runWith({ timeoutSeconds: 300, memory: "256MB" }).firestore
  .document("escalation_timers/{timerId}")
  .onCreate(async (snap, context) => {
    const timerId = context.params.timerId;
    const timerData = snap.data() || {};
    const alertId = timerData.alertId || "";
    const userId = timerData.userId || "";
    const tripId = timerData.tripId || "";

    console.log(`Escalation timer created: ${timerId} for alert ${alertId}`);

    // Wait for escalation delay (Cloud Functions Gen1 max timeout = 540s = 9min)
    // 2 minutes is well within limits
    const delayMs = ESCALATION_DELAY_MS;
    await new Promise((resolve) => setTimeout(resolve, delayMs));

    try {
      // Re-read the alert to check if it was resolved
      const alertSnap = await admin.firestore().collection("emergency_alerts").doc(alertId).get();
      const alertData = alertSnap.exists ? alertSnap.data() : {};

      if (alertData.status === "resolved" || alertData.status === "cancelled") {
        console.log(`Alert ${alertId} already resolved/cancelled — skipping escalation`);
        await snap.ref.set({ status: "cancelled", reason: "alert_resolved" }, { merge: true });
        return null;
      }

      // Check trip document for escalation attempts
      let tripDoc = {};
      if (tripId) {
        const tripSnap = await admin.firestore().collection("trips").doc(tripId).get();
        tripDoc = tripSnap.exists ? tripSnap.data() : {};
      }
      const attempts = Number(tripDoc.escalationAttempts || 0) + 1;

      if (TEST_MODE) {
        console.log(`[TEST MODE] Escalation #${attempts} for alert ${alertId}. Would contact authorities.`);
      } else {
        // In production: send SMS to authorities, make further FCM pushes, etc.
        console.log(`Escalation #${attempts} for alert ${alertId}. Contacting authorities.`);
      }

      // Update escalation timer
      await snap.ref.set({
        status: "escalated",
        escalatedAt: admin.firestore.FieldValue.serverTimestamp(),
        attemptNumber: attempts,
      }, { merge: true });

      // Update the alert
      await admin.firestore().collection("emergency_alerts").doc(alertId).set({
        status: "escalated",
        escalationAttempts: attempts,
        lastEscalatedAt: admin.firestore.FieldValue.serverTimestamp(),
      }, { merge: true });

      // Update trip document
      if (tripId) {
        await admin.firestore().collection("trips").doc(tripId).set({
          escalationAttempts: attempts,
        }, { merge: true });
      }

      // Write another admin notification for the escalation
      await admin.firestore().collection("admin_notifications").add({
        type: "ESCALATION",
        alertId,
        userId,
        tripId,
        attemptNumber: attempts,
        status: "pending_review",
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
      });

      console.log(`Escalation complete for alert ${alertId}, attempt #${attempts}`);
    } catch (error) {
      console.error("Error during escalation:", error.message);
      await snap.ref.set({ status: "error", error: error.message }, { merge: true });
    }
    return null;
  });
