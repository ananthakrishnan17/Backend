/**
 * ============================================================
 * VOICE-TO-BILL POS — Backend Server (Local Whisper Version)
 * Stack : Node.js + Local Faster-Whisper + Google Gemini
 * ============================================================
 */

import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenerativeAI } from "@google/generative-ai";
import cors from "cors";
import { spawn } from "child_process"; // Python script-ah run panna ithu venum

// ─── ESM __dirname shim ───────────────────────────────────────
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ─── Environment Config ───────────────────────────────────────
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "AIzaSyAFwM7K4GdQBKpiy4f0p6c-xFF917BcT6U";
const PORT = process.env.PORT || 3000;

// ─── SDK Initialisation ───────────────────────────────────────
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// ─── Express App ──────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json());

// ─── Local Whisper Transcription Function ───────────────────
/**
 * Intha function unga transcribe.py script-ah call panni 
 * audio-va text-ah mathi tharum.
 */
const getLocalTranscription = (filePath) => {
  return new Promise((resolve, reject) => {
    // Unga system-la 'python' nu type panna work aagalana 'python3' nu mathunga
    const pythonProcess = spawn('python', ["transcribe.py", filePath]);

    let result = "";
    let error = "";

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve(result.trim());
      } else {
        console.error("Whisper Error:", error);
        reject("Local Whisper failed to process audio.");
      }
    });
  });
};

// ─── Multer: audio upload to /tmp ────────────────────────────
const upload = multer({
  dest: path.join(__dirname, "tmp"),
  limits: { fileSize: 25 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const allowed = ["audio/m4a", "audio/wav", "audio/mpeg", "audio/webm", "audio/mp4", "audio/ogg", "audio/x-m4a"];
    if (allowed.includes(file.mimetype) || file.originalname.match(/\.(m4a|wav|mp3|webm|ogg)$/i)) {
      cb(null, true);
    } else {
      cb(new Error("Unsupported audio format. Use m4a, wav, mp3, or webm."));
    }
  },
});

// ─── INVENTORY ────────────────────────────────────────────────
const INVENTORY = [
  { id: 1,  name: "Tea",           price: 12,  aliases: ["chai", "tae", "te", "chaa", "cha"] },
  { id: 2,  name: "Coffee",        price: 20,  aliases: ["kaapi", "kopi", "coffe", "cofee", "kape"] },
  { id: 3,  name: "Samosa",        price: 15,  aliases: ["samusa", "samosa", "samoosa", "chamosa", "samose"] },
  { id: 4,  name: "Vada",          price: 10,  aliases: ["vada", "wada", "vadai", "medhu vada"] },
  { id: 5,  name: "Idli",          price: 8,   aliases: ["idly", "idli", "iddli", "ittly"] },
  { id: 6,  name: "Dosa",          price: 30,  aliases: ["dosai", "dosa", "dhosa", "tosa"] },
  { id: 7,  name: "Pongal",        price: 25,  aliases: ["pongal", "pongall", "khara pongal"] },
  { id: 8,  name: "Bajji",         price: 10,  aliases: ["bajji", "bhaji", "pakoda", "pakora", "bajjji"] },
  { id: 9,  name: "Parotta",       price: 20,  aliases: ["parotta", "parota", "porotta", "parotha"] },
  { id: 10, name: "Lassi",         price: 25,  aliases: ["lassi", "lasi", "lassie"] },
  { id: 11, name: "Juice",         price: 30,  aliases: ["juice", "joos", "jos"] },
  { id: 12, name: "Water Bottle",  price: 20,  aliases: ["water", "paani", "thanni", "waterbottle", "mineral water"] },
  { id: 13, name: "Biscuit",       price: 5,   aliases: ["biscuit", "biskit", "biscuits", "biscuite"] },
  { id: 14, name: "Bread",         price: 35,  aliases: ["bread", "bred", "pav", "pavv"] },
  { id: 15, name: "Egg",           price: 7,   aliases: ["egg", "muttai", "eggs", "egge"] },
];

// ─── FUZZY MATCHING ENGINE ────────────────────────────────────
function levenshtein(a, b) {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) => Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1] ? dp[i - 1][j - 1] : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return dp[m][n];
}

function similarity(a, b) {
  const dist = levenshtein(a.toLowerCase(), b.toLowerCase());
  return 1 - dist / Math.max(a.length, b.length);
}

const FUZZY_THRESHOLD = 0.60;

function fuzzyMatch(rawName) {
  const query = rawName.toLowerCase().trim();
  let bestItem = null;
  let bestScore = 0;

  for (const item of INVENTORY) {
    const candidates = [item.name, ...item.aliases];
    for (const candidate of candidates) {
      const c = candidate.toLowerCase();
      if (c === query) return { item, score: 1.0 };
      const subScore = (c.includes(query) || query.includes(c)) ? 0.9 : 0;
      const levScore = similarity(query, c);
      const finalScore = Math.max(subScore, levScore);
      if (finalScore > bestScore) {
        bestScore = finalScore;
        bestItem = item;
      }
    }
  }
  return bestScore >= FUZZY_THRESHOLD ? { item: bestItem, score: bestScore } : null;
}

// ─── GEMINI SYSTEM PROMPT ─────────────────────────────────────
const GEMINI_SYSTEM_PROMPT = `Extract items and quantities from Tanglish. Tamil numbers: Onnu=1, Rendu=2, Moonu=3, Naalu=4, Anju=5. Return JSON array: [{"name":"Item","quantity":X}]`;

// ─── CORE ROUTE: /process-voice ──────────────────────────────
app.post("/process-voice", upload.single("audio"), async (req, res) => {
  const tmpFilePath = req.file?.path;

  try {
    if (!req.file) return res.status(400).json({ success: false, error: "No audio file uploaded." });

    const ext = path.extname(req.file.originalname) || ".m4a";
    const namedPath = `${tmpFilePath}${ext}`;
    fs.renameSync(tmpFilePath, namedPath);

    // ── Step 2: LOCAL WHISPER (No API Cost!) ───────────────
    console.log("[2/4] Running Local Faster-Whisper...");
    const rawText = await getLocalTranscription(namedPath);
    console.log(`[2/4] Whisper Output: "${rawText}"`);

    if (!rawText) {
      if (fs.existsSync(namedPath)) fs.unlinkSync(namedPath);
      return res.status(200).json({ success: false, error: "Empty transcription.", transcript: "", items: [], total: 0 });
    }

    // ── Step 3: Gemini entity extraction ───────────────────
    console.log("[3/4] Sending to Gemini...");
    const geminiResponse = await geminiModel.generateContent(`${GEMINI_SYSTEM_PROMPT}\n\nOrder text: "${rawText}"`);
    const geminiText = geminiResponse.response.text().trim();
    
    let extractedItems = [];
    try {
      const cleaned = geminiText.replace(/```json|```/g, "").trim();
      extractedItems = JSON.parse(cleaned);
    } catch (parseErr) {
      return res.status(200).json({ success: false, error: "AI parse failed", transcript: rawText, items: [] });
    }

    // ── Step 4: Fuzzy match + build bill ───────────────────
    const billItems = [];
    for (const extracted of extractedItems) {
      const qty = Math.max(1, parseInt(extracted.quantity) || 1);
      const match = fuzzyMatch(extracted.name);

      if (match) {
        const existing = billItems.find((b) => b.id === match.item.id);
        if (existing) {
          existing.quantity += qty;
          existing.subtotal = existing.quantity * existing.unitPrice;
        } else {
          billItems.push({
            id: match.item.id,
            name: match.item.name,
            quantity: qty,
            unitPrice: match.item.price,
            subtotal: qty * match.item.price,
          });
        }
      }
    }

    const grandTotal = billItems.reduce((sum, i) => sum + i.subtotal, 0);
    if (fs.existsSync(namedPath)) fs.unlinkSync(namedPath);

    return res.status(200).json({ success: true, transcript: rawText, items: billItems, total: grandTotal });

  } catch (err) {
    if (tmpFilePath && fs.existsSync(tmpFilePath)) fs.unlinkSync(tmpFilePath);
    res.status(500).json({ success: false, error: err.message });
  }
});
// ─── Health Check ─────────────────────────────────────────────
app.get("/health", (_req, res) => {
  res.json({ 
    status: "ok", 
    inventory_count: INVENTORY.length, 
    uptime: process.uptime() 
  });
});

// ─── Inventory List (Flutter-kaga) ───────────────────────────
app.get("/inventory", (_req, res) => {
  res.json({ 
    success: true, 
    items: INVENTORY 
  });
});
app.listen(PORT, () => console.log(`🚀 Server running on http://localhost:${PORT}`));