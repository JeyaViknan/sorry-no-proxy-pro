require("dotenv").config();
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const { google } = require("googleapis");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 8000;

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

const verifierState = {
    process: null,
    buffer: "",
    nextId: 1,
    pending: new Map(),
};

function rejectAllPending(message) {
    for (const { reject, timer } of verifierState.pending.values()) {
        clearTimeout(timer);
        reject(new Error(message));
    }
    verifierState.pending.clear();
}

function handleVerifierOutput(chunk) {
    verifierState.buffer += chunk.toString();
    const lines = verifierState.buffer.split("\n");
    verifierState.buffer = lines.pop() || "";

    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
            continue;
        }

        let payload;
        try {
            payload = JSON.parse(trimmed);
        } catch (err) {
            console.error("[face-verifier] non-json stdout:", trimmed);
            continue;
        }

        const requestId = payload.id;
        const pending = verifierState.pending.get(requestId);
        if (!pending) {
            continue;
        }

        clearTimeout(pending.timer);
        verifierState.pending.delete(requestId);
        pending.resolve(payload);
    }
}

function startVerifierProcess() {
    const scriptPath = path.join(__dirname, "face_verification.py");
    const proc = spawn("python3", ["-u", scriptPath, "--serve"], {
        cwd: __dirname,
        stdio: ["pipe", "pipe", "pipe"],
    });

    verifierState.process = proc;
    verifierState.buffer = "";

    proc.stdout.on("data", handleVerifierOutput);

    proc.stderr.on("data", (chunk) => {
        console.log(`[face-verifier] ${chunk.toString().trim()}`);
    });

    proc.on("error", (err) => {
        console.error("[face-verifier] process error:", err);
    });

    proc.on("exit", (code, signal) => {
        console.error(`[face-verifier] exited (code=${code}, signal=${signal})`);
        verifierState.process = null;
        rejectAllPending("Face verifier is unavailable");

        // Auto-restart worker so service remains available.
        setTimeout(() => {
            startVerifierProcess();
        }, 1000);
    });
}

function verifyFaceWithWorker(registerNumber, faceImage) {
    return new Promise((resolve, reject) => {
        if (!verifierState.process || verifierState.process.killed) {
            startVerifierProcess();
        }

        const requestId = verifierState.nextId++;
        const timer = setTimeout(() => {
            verifierState.pending.delete(requestId);
            reject(new Error("Face verification request timed out"));
        }, 10000);

        verifierState.pending.set(requestId, { resolve, reject, timer });

        try {
            const payload = JSON.stringify({
                id: requestId,
                registerNumber,
                faceImage,
            });
            verifierState.process.stdin.write(`${payload}\n`);
        } catch (err) {
            clearTimeout(timer);
            verifierState.pending.delete(requestId);
            reject(err);
        }
    });
}

// Start verifier worker on server start (preloads embeddings and model once).
startVerifierProcess();

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
            message: "Face verification failed. Please try again.",
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
                message: "Face verification failed. Please try again.",
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

app.get("/", (req, res) => {
    res.send("✅ Server is running! Use POST /register to register.");
});
