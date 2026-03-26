require("dotenv").config();
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const { google } = require("googleapis");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");

const app = express();
const PORT = process.env.PORT || 8000;
const APP_DEPLOY_MARKER = "insightface-worker-v2";

app.use(cors({ origin: "*" }));
app.use(bodyParser.json({ limit: "10mb" }));
app.use(express.static(__dirname));

const auth = new google.auth.GoogleAuth({
    credentials: {
        client_email: process.env.GOOGLE_SERVICE_ACCOUNT_EMAIL,
        private_key: (process.env.GOOGLE_PRIVATE_KEY || "").replace(/\\n/g, "\n"),
    },
    scopes: ["https://www.googleapis.com/auth/spreadsheets"],
});
const sheets = google.sheets({ version: "v4", auth });

let isVerifying = false;
const verificationQueue = [];

function processNextInQueue() {
    if (isVerifying || verificationQueue.length === 0) return;
    isVerifying = true;
    
    const { registerNumber, faceImage, resolve, reject } = verificationQueue.shift();
    
    const tmpId = crypto.randomUUID();
    const payload = faceImage.split(",")[1] || faceImage;
    const tmpPath = path.join("/tmp", `${tmpId}.jpg`);
    
    try {
        fs.writeFileSync(tmpPath, Buffer.from(payload, "base64"));
    } catch(err) {
        isVerifying = false;
        reject(new Error("Failed to write temp image file"));
        processNextInQueue();
        return;
    }
    
    const scriptPath = path.join(__dirname, "face_verification.py");
    const proc = spawn("python3", [scriptPath, registerNumber, tmpPath], {
        cwd: __dirname
    });
    
    let stdoutBuffer = "";
    let stderrBuffer = "";
    
    proc.stdout.on("data", chunk => stdoutBuffer += chunk.toString());
    proc.stderr.on("data", chunk => stderrBuffer += chunk.toString());
    
    proc.on("close", (code) => {
        try { fs.unlinkSync(tmpPath); } catch(e) {}
        isVerifying = false;
        
        if (code !== 0) {
            console.error("[face-verifier error]:", stderrBuffer);
            reject(new Error("Face verifier failed. Server might be under heavy load."));
            processNextInQueue();
            return;
        }
        
        try {
            const result = JSON.parse(stdoutBuffer.trim());
            if (result.error) {
                reject(new Error(result.error));
            } else {
                resolve(result);
            }
        } catch(err) {
            console.error("Parse error:", err, stdoutBuffer);
            reject(new Error("Invalid response from face verifier"));
        }
        
        processNextInQueue();
    });
}

function verifyFaceWithWorker(registerNumber, faceImage) {
    return new Promise((resolve, reject) => {
        verificationQueue.push({ registerNumber, faceImage, resolve, reject });
        processNextInQueue();
    });
}

function formatVerifierError(error) {
    const message = (error && error.message) || "Face verification failed.";
    return message.replace(/\s+/g, " ").trim();
}

app.post("/verify-face", async (req, res) => {
    const { registerNumber, faceImage } = req.body;

    if (!registerNumber || !faceImage) {
        return res.status(400).json({
            success: false,
            verified: false,
            message: "Missing register number or face image",
        });
    }

    try {
        const result = await verifyFaceWithWorker(registerNumber, faceImage);
        res.json({
            success: true,
            verified: !!result.verified,
            message: result.message,
            confidence: result.confidence,
            threshold: result.threshold,
        });
    } catch (error) {
        console.error("Face verification error:", error);
        res.status(500).json({
            success: false,
            verified: false,
            message: formatVerifierError(error),
        });
    }
});

app.post("/register", async (req, res) => {
    const { registerNumber, faceImage } = req.body;

    if (!registerNumber) {
        return res.status(400).json({ success: false, message: "Missing register number" });
    }

    // Optional: if caller includes a face image here, verify using the same worker.
    if (faceImage) {
        try {
            const verificationResult = await verifyFaceWithWorker(registerNumber, faceImage);
            if (!verificationResult.verified) {
                return res.status(400).json({
                    success: false,
                    message: verificationResult.message || "Face verification failed. Please ensure your face matches the registration number.",
                    verified: false,
                    confidence: verificationResult.confidence,
                });
            }
        } catch (error) {
            console.error("Face verification error:", error);
            return res.status(500).json({
                success: false,
                message: formatVerifierError(error),
            });
        }
    }

    const generatedCode = `CODE-${Math.floor(Math.random() * 10000)}`;

    try {
        await sheets.spreadsheets.values.append({
            spreadsheetId: process.env.SHEET_ID,
            range: "Sheet1!A:B",
            valueInputOption: "RAW",
            requestBody: { values: [[registerNumber, generatedCode]] },
        });

        res.json({ success: true, code: generatedCode });
    } catch (error) {
        console.error("Google Sheets error:", error);
        res.status(500).json({ success: false, message: "Failed to save to Google Sheets" });
    }
});

app.listen(PORT, () => {
    console.log(`✅ Server running on port ${PORT}`);
});

app.get("/healthz", (req, res) => {
    res.json({
        ok: true,
        deployMarker: APP_DEPLOY_MARKER,
        verifierReady: true,
        verifierLastError: null,
    });
});

app.get("/", (req, res) => {
    res.send(`✅ Server is running (${APP_DEPLOY_MARKER}). Use POST /register to register.`);
});
