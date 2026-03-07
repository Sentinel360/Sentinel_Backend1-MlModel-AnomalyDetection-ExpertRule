# Flutter ↔ Firestore Data Schema (Sentinel360)

This schema is the source of truth for app integration with the deployed backend.

## 1) Trip Document

Path: `trips/{tripId}`

Create this at trip start.

```json
{
  "status": "active",
  "origin": { "lat": 5.6519, "lon": -0.1873 },
  "destination": { "lat": 5.6052, "lon": -0.1668 },
  "startedAt": 1760000000000,
  "driver_id": "driver_123",
  "vehicle_type": "sedan"
}
```

### Required fields
- `status` (`active` at start, `completed` at end)
- `origin.lat`, `origin.lon`
- `destination.lat`, `destination.lon`

### Optional fields
- `driver_id`
- `vehicle_type`
- `startedAt`

---

## 2) Sensor Event Document

Path: `trips/{tripId}/sensor_data/{eventId}`

Write this continuously during a trip (recommended: ~1 Hz).

```json
{
  "trip_id": "trip_abc_001",
  "timestamp": 1760000600000,
  "source": "PHONE",
  "gps": {
    "lat": 5.6400,
    "lon": -0.1800,
    "speed": 42.5,
    "accuracy": 6.2,
    "heading": 132.0,
    "altitude": 71.0
  },
  "acceleration": {
    "x": 0.8,
    "y": 0.2,
    "z": 9.8
  },
  "gyro": {
    "x": 0.02,
    "y": 0.03,
    "z": 0.01
  },
  "activity": "in_vehicle"
}
```

### Required fields for successful prediction
- `trip_id`
- `timestamp` (milliseconds since epoch)
- `gps.lat`
- `gps.lon`
- `gps.speed` (km/h)
- `acceleration.x`

### Recommended fields for better model quality
- `acceleration.y`, `acceleration.z`
- `gps.accuracy`, `gps.heading`, `gps.altitude`
- `gyro.x`, `gyro.y`, `gyro.z`
- `activity`
- `source` (`PHONE` or `IOT`)

---

## 3) Live Output State Document

Path: `trips/{tripId}/current_state/latest`

This is written by Cloud Functions after each sensor event.

```json
{
  "riskScore": 0.64,
  "riskLevel": "MEDIUM",
  "riskColor": "orange",
  "overallRiskLevel": "MEDIUM",
  "overallUnsafe": false,
  "activeSensor": "PHONE",
  "explanation": "Driving behaviour: MEDIUM ...",
  "modelVersion": "1.0.0",
  "components": {},
  "actions": [],
  "policy": {
    "totalWindows": 18,
    "highWindows": 2,
    "highRatio": 0.1111,
    "consecutiveHigh": 0,
    "maxConsecutiveHigh": 2,
    "lastHighTimestamp": 1760000540000,
    "latchedHighUntil": 1760000600000,
    "overallUnsafe": false,
    "overallLevel": "MEDIUM",
    "reason": "Elevated risk trend"
  },
  "updatedAt": "Firestore server timestamp"
}
```

### Semantics
- `riskLevel` / `riskColor`: **instant** risk now
- `overallRiskLevel` / `overallUnsafe`: **trip trend** risk policy

---

## 4) Alerts Collection

Path: `trips/{tripId}/alerts/{alertId}`

Created when instant level is `HIGH RISK`.

```json
{
  "timestamp": "Firestore server timestamp",
  "riskScore": 0.84,
  "reason": "Driving behaviour: HIGH RISK ...",
  "actions": ["ALERT_USER"],
  "resolved": false
}
```

---

## 5) Trip End Behavior

When trip status becomes `completed`, function finalizes and writes summary fields back to:

Path: `trips/{tripId}`

- `summary`
- `riskSummary.overallRiskLevel`
- `riskSummary.overallUnsafe`
- `riskSummary.totalWindows`
- `riskSummary.highWindows`
- `riskSummary.highRatio`
- `riskSummary.maxConsecutiveHigh`
- `riskSummary.policyReason`
- `processedAt`
