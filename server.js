require("dotenv").config();
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const { google } = require("googleapis");
const { exec } = require("child_process");
const { promisify } = require("util");
const path = require("path");

const execAsync = promisify(exec);

const app = express();
const PORT = process.env.PORT || 8000;

app.use(cors({ origin: "*" })); 
app.use(bodyParser.json({ limit: "10mb" })); // Increase limit for base64 images

// Serve static files (HTML, CSS, JS)
app.use(express.static(__dirname));


const auth = new google.auth.GoogleAuth({
    credentials: {
        client_email: process.env.GOOGLE_SERVICE_ACCOUNT_EMAIL,
        private_key: (process.env.GOOGLE_PRIVATE_KEY || "").replace(/\\n/g, '\n'),
    },
    scopes: ["https://www.googleapis.com/auth/spreadsheets"],
});
const sheets = google.sheets({ version: "v4", auth });

// Face verification endpoint
app.post("/verify-face", async (req, res) => {
    const { registerNumber, faceImage } = req.body;

    if (!registerNumber || !faceImage) {
        return res.status(400).json({ 
            success: false, 
            verified: false,
            message: "Missing register number or face image" 
        });
    }

    try {
        // Escape the base64 image for command line
        const escapedImage = faceImage.replace(/"/g, '\\"');
        const escapedRegno = registerNumber.replace(/"/g, '\\"');
        
        // Call Python script for face verification
        const scriptPath = path.join(__dirname, "face_verification.py");
        const command = `python3 "${scriptPath}" "${escapedRegno}" "${escapedImage}"`;
        
        const { stdout, stderr } = await execAsync(command);
        
        if (stderr && !stderr.includes("Loading")) {
            console.error("Python script error:", stderr);
        }
        
        const result = JSON.parse(stdout);
        res.json({
            success: true,
            verified: result.verified,
            message: result.message,
            confidence: result.confidence
        });
    } catch (error) {
        console.error("Face verification error:", error);
        res.status(500).json({ 
            success: false, 
            verified: false,
            message: "Face verification failed. Please try again." 
        });
    }
});

app.post("/register", async (req, res) => {
    const { registerNumber, faceImage } = req.body;

    if (!registerNumber) {
        return res.status(400).json({ success: false, message: "Missing register number" });
    }

    // Verify face if image is provided
    if (faceImage) {
        try {
            const escapedImage = faceImage.replace(/"/g, '\\"');
            const escapedRegno = registerNumber.replace(/"/g, '\\"');
            const scriptPath = path.join(__dirname, "face_verification.py");
            const command = `python3 "${scriptPath}" "${escapedRegno}" "${escapedImage}"`;
            
            const { stdout, stderr } = await execAsync(command);
            
            if (stderr && !stderr.includes("Loading")) {
                console.error("Python script error:", stderr);
            }
            
            const verificationResult = JSON.parse(stdout);
            
            if (!verificationResult.verified) {
                return res.status(400).json({ 
                    success: false, 
                    message: verificationResult.message || "Face verification failed. Please ensure your face matches the registration number.",
                    verified: false,
                    confidence: verificationResult.confidence
                });
            }
        } catch (error) {
            console.error("Face verification error:", error);
            return res.status(500).json({ 
                success: false, 
                message: "Face verification failed. Please try again." 
            });
        }
    }

    // If face verification passed (or not required), proceed with registration
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