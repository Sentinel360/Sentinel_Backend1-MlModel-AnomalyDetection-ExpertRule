const functions = require("firebase-functions");
const admin = require("firebase-admin");
const axios = require("axios");
const { GoogleAuth } = require("google-auth-library");
const { defineString } = require("firebase-functions/params");

admin.initializeApp();

// Set once at deploy time: firebase deploy will prompt for ML_API_URL value.
const ML_API_URL_PARAM = defineString("ML_API_URL");
// Optional: must match Cloud Run ML_API_KEY when set (sent as X-API-Key).
const ML_API_KEY_PARAM = defineString("ML_API_KEY", { default: "" });
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

function getMlApiKey() {
  if (process.env.ML_API_KEY) return process.env.ML_API_KEY;
  return ML_API_KEY_PARAM.value() || "";
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

function normalizeLatLon(value) {
  if (!value || typeof value !== "object") return { lat: 0, lon: 0 };
  const lat = value.lat ?? value.latitude ?? value._latitude ?? 0;
  const lon = value.lon ?? value.lng ?? value.longitude ?? value._longitude ?? 0;
  return { lat: Number(lat), lon: Number(lon) };
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
  const headers = { ...(await getMlApiAuthHeaders()) };
  const apiKey = getMlApiKey();
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
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
          origin: normalizeLatLon(tripData.origin),
          destination: normalizeLatLon(tripData.destination),
        },
        5000,
      );
      console.log(`Trip monitoring initialized: ${tripId}`);
    } catch (error) {
      console.error("Error initializing trip on ML API:", error.message);
    }

    // Ensure expected_route is set for route anomaly detection
    const tripRef = admin.firestore().collection("trips").doc(tripId);
    const freshTrip = await tripRef.get();
    const freshData = freshTrip.data() || {};

    if (!freshData.expected_route && !freshData.routePolyline) {
      // Trip has no route data yet — try to fetch from Google Directions
      const origin = normalizeLatLon(tripData.origin);
      const dest = normalizeLatLon(tripData.destination);

      if (origin.lat && origin.lon && dest.lat && dest.lon) {
        const MAPS_API_KEY = process.env.GOOGLE_MAPS_API_KEY || "";
        if (MAPS_API_KEY) {
          try {
            const directionsResp = await axios.get(
              `https://maps.googleapis.com/maps/api/directions/json` +
              `?origin=${origin.lat},${origin.lon}` +
              `&destination=${dest.lat},${dest.lon}` +
              `&key=${MAPS_API_KEY}`,
              { timeout: 5000 }
            );

            if (directionsResp.data.status === "OK") {
              const encodedPolyline = directionsResp.data.routes[0].overview_polyline.points;
              await tripRef.update({
                expected_route: {
                  polyline: encodedPolyline,
                  fetchedBy: "cloud_function",
                  fetchedAt: admin.firestore.FieldValue.serverTimestamp(),
                },
              });
              console.log(`Route fetched and saved for trip ${tripId}`);
            }
          } catch (routeErr) {
            console.warn(`Could not fetch route for trip ${tripId}: ${routeErr.message}`);
          }
        }
      }
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

// ── SOS Escalation Pipeline ──────────────────────────────────────────────────
// Triggers when a new emergency_alert document is created.
// In production: sends FCM push to emergency contacts, sends SMS via Africa's Talking.
// In test mode: logs what would be sent.

const TEST_MODE = true; // Set to false for production deployment

exports.onEmergencyAlert = functions.firestore
  .document("emergency_alerts/{alertId}")
  .onCreate(async (snap, context) => {
    const alertId = context.params.alertId;
    const alertData = snap.data() || {};
    const userId = alertData.userId;
    const tripId = alertData.tripId;
    const alertType = alertData.type || "SOS_MANUAL";

    console.log(`[SOS Escalation] Processing alert ${alertId} for user ${userId}, trip ${tripId}`);

    try {
      // 1. Get user data and emergency contacts
      const userDoc = await admin.firestore().collection("users").doc(userId).get();
      const userData = userDoc.exists ? userDoc.data() : {};
      const userName = userData.displayName || userData.name || "Sentinel360 User";
      const emergencyContacts = userData.emergencyContacts || [];

      // 2. Get trip data for location context
      let locationText = "Location unavailable";
      let locationLink = "";
      if (tripId) {
        const tripDoc = await admin.firestore().collection("trips").doc(tripId).get();
        if (tripDoc.exists) {
          const tripData = tripDoc.data() || {};
          const origin = tripData.origin || tripData.originGeo;
          const dest = tripData.destination || tripData.destinationGeo;
          const destName = tripData.destinationName || "Unknown destination";

          // Get latest sensor position
          const latestSensor = await admin.firestore()
            .collection("trips").doc(tripId)
            .collection("sensor_data")
            .orderBy("timestamp", "desc")
            .limit(1)
            .get();

          let lat = 0, lon = 0;
          if (!latestSensor.empty) {
            const sensorData = latestSensor.docs[0].data();
            const gps = sensorData.gps || {};
            lat = gps.lat || 0;
            lon = gps.lon || 0;
          } else if (origin) {
            lat = origin.lat || origin._latitude || origin.latitude || 0;
            lon = origin.lon || origin._longitude || origin.longitude || 0;
          }

          if (lat && lon) {
            locationText = `Lat: ${lat.toFixed(6)}, Lon: ${lon.toFixed(6)}`;
            locationLink = `https://maps.google.com/?q=${lat},${lon}`;
          }
          locationText += ` (heading to ${destName})`;
        }
      }

      // 3. Compose emergency message
      const message = `EMERGENCY ALERT from ${userName}!\n` +
        `Type: ${alertType}\n` +
        `${locationText}\n` +
        (locationLink ? `Track location: ${locationLink}\n` : "") +
        `Time: ${new Date().toLocaleString("en-GH", { timeZone: "Africa/Accra" })}`;

      // 4. Send notifications to each emergency contact
      const notifications = [];

      for (const contact of emergencyContacts) {
        const contactName = contact.name || "Emergency Contact";
        const contactPhone = contact.phone || contact.phoneNumber;
        const contactFcmToken = contact.fcmToken;

        if (TEST_MODE) {
          console.log(`[SOS TEST MODE] Would notify ${contactName}:`);
          console.log(`  Phone: ${contactPhone}`);
          console.log(`  FCM Token: ${contactFcmToken ? "present" : "none"}`);
          console.log(`  Message: ${message}`);
          notifications.push({
            contact: contactName,
            method: "test_mode",
            status: "simulated",
          });
          continue;
        }

        // FCM Push Notification (if contact has the app)
        if (contactFcmToken) {
          try {
            await admin.messaging().send({
              token: contactFcmToken,
              notification: {
                title: `SOS Alert from ${userName}`,
                body: `${userName} triggered an emergency alert. Tap to view location.`,
              },
              data: {
                type: "SOS_ALERT",
                alertId: alertId,
                tripId: tripId || "",
                location: locationLink || "",
              },
              android: {
                priority: "high",
                notification: {
                  channelId: "emergency_alerts",
                  priority: "max",
                  sound: "alarm",
                },
              },
            });
            notifications.push({ contact: contactName, method: "fcm", status: "sent" });
          } catch (fcmErr) {
            console.error(`[SOS] FCM failed for ${contactName}: ${fcmErr.message}`);
            notifications.push({ contact: contactName, method: "fcm", status: "failed", error: fcmErr.message });
          }
        }

        // SMS via Africa's Talking (Ghana-optimized)
        // To enable: npm install africastalking, set AT_API_KEY and AT_USERNAME in env
        if (contactPhone) {
          try {
            // Africa's Talking integration placeholder
            // In production, uncomment and configure:
            // const AfricasTalking = require("africastalking");
            // const at = AfricasTalking({ apiKey: process.env.AT_API_KEY, username: process.env.AT_USERNAME });
            // const sms = at.SMS;
            // await sms.send({ to: [contactPhone], message: message, from: "Sentinel360" });

            console.log(`[SOS] SMS would be sent to ${contactPhone}`);
            notifications.push({ contact: contactName, method: "sms", status: "pending_integration" });
          } catch (smsErr) {
            console.error(`[SOS] SMS failed for ${contactName}: ${smsErr.message}`);
            notifications.push({ contact: contactName, method: "sms", status: "failed", error: smsErr.message });
          }
        }
      }

      // 5. Update the alert document with notification results
      await snap.ref.update({
        notificationsSent: notifications,
        escalationStatus: "notified",
        processedAt: admin.firestore.FieldValue.serverTimestamp(),
      });

      // 6. Start auto-escalation timer (2 minutes)
      // Create an escalation document that the onEscalationCheck function monitors
      await admin.firestore().collection("escalation_timers").doc(alertId).set({
        alertId,
        userId,
        tripId: tripId || null,
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
        escalateAfter: admin.firestore.Timestamp.fromMillis(Date.now() + 2 * 60 * 1000),
        status: "pending",
        contactsNotified: notifications.length,
      });

      console.log(`[SOS Escalation] Alert ${alertId} processed: ${notifications.length} contacts notified`);
      return null;
    } catch (error) {
      console.error(`[SOS Escalation] Error processing alert ${alertId}: ${error.message}`);
      await snap.ref.update({
        escalationStatus: "error",
        escalationError: error.message,
        processedAt: admin.firestore.FieldValue.serverTimestamp(),
      });
      return null;
    }
  });

// ── Auto-Escalation Check ────────────────────────────────────────────────────
// Runs every minute via Cloud Scheduler (configure in Firebase console).
// Checks for unresolved escalation timers that have expired.
// In a real system this would be a scheduled function; here it's also triggerable
// by writing to escalation_timers.

exports.onEscalationTimerCreate = functions.firestore
  .document("escalation_timers/{timerId}")
  .onUpdate(async (change, context) => {
    const timerId = context.params.timerId;
    const newData = change.after.data() || {};

    // Only process when status changes to "check"
    if (newData.status !== "check") return null;

    const alertId = newData.alertId;
    const userId = newData.userId;
    const tripId = newData.tripId;

    try {
      // Check if the alert has been resolved
      const alertDoc = await admin.firestore().collection("emergency_alerts").doc(alertId).get();
      const alertData = alertDoc.exists ? alertDoc.data() : {};

      if (alertData.resolved) {
        await change.after.ref.update({ status: "resolved" });
        console.log(`[Escalation] Alert ${alertId} already resolved, no escalation needed`);
        return null;
      }

      // Check if any contact acknowledged
      const userDoc = await admin.firestore().collection("users").doc(userId).get();
      const userData = userDoc.exists ? userDoc.data() : {};

      if (TEST_MODE) {
        console.log(`[Escalation TEST MODE] Would escalate alert ${alertId} to authorities:`);
        console.log(`  User: ${userData.displayName || "Unknown"}`);
        console.log(`  Trip: ${tripId}`);
        console.log(`  Action: Notify local authorities / admin dashboard`);
      }

      // Mark as escalated
      await change.after.ref.update({
        status: "escalated",
        escalatedAt: admin.firestore.FieldValue.serverTimestamp(),
      });

      // Update the original alert
      await admin.firestore().collection("emergency_alerts").doc(alertId).update({
        escalationStatus: "escalated_to_authorities",
        escalatedAt: admin.firestore.FieldValue.serverTimestamp(),
      });

      // Create a high-priority admin notification
      await admin.firestore().collection("admin_notifications").add({
        type: "SOS_ESCALATION",
        alertId,
        userId,
        tripId: tripId || null,
        message: `Unresolved SOS from ${userData.displayName || "Unknown user"} — no contact response after 2 minutes`,
        priority: "critical",
        read: false,
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
      });

      console.log(`[Escalation] Alert ${alertId} escalated to admin dashboard`);
      return null;
    } catch (error) {
      console.error(`[Escalation] Error: ${error.message}`);
      return null;
    }
  });
