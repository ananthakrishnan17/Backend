/**
 * ============================================================
 * VOICE-TO-BILL POS — Backend Server (Fixed & Final)
 * ============================================================
 */

import express from "express";
import multer from "multer";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { GoogleGenerativeAI } from "@google/generative-ai";
import cors from "cors";
import { spawn } from "child_process";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ─── Environment Config ───────────────────────────────────────
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "AIzaSyAFwM7K4GdQBKpiy4f0p6c-xFF917BcT6U";
const PORT = process.env.PORT || 3000;

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ 
    model: "gemini-1.5-flash"
}, { apiVersion: 'v1' });
const app = express();
app.use(cors());
app.use(express.json());

// ─── Local Whisper Function (UTF-8 Fixed) ─────────────────────
const getLocalTranscription = (filePath) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', ["transcribe.py", filePath], {
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
    });

    let result = "";
    let error = "";

    pythonProcess.stdout.on('data', (data) => { result += data.toString('utf8'); });
    pythonProcess.stderr.on('data', (data) => { error += data.toString('utf8'); });

    pythonProcess.on('close', (code) => {
      if (code === 0) resolve(result.trim());
      else reject(error || "Local Whisper failed.");
    });
  });
};

const upload = multer({ dest: path.join(__dirname, "tmp") });

// ─── INVENTORY ────────────────────────────────────────────────
const INVENTORY = [
  { id: 1,  name: "Tea",           price: 12,  aliases: ["chai", "tae", "te", "chaa", "cha", "டீ", "தீ"] },
  { id: 2,  name: "Coffee",        price: 20,  aliases: ["kaapi", "kopi", "coffe", "cofee", "kape", "காபி"] },
  { id: 3,  name: "Samosa",        price: 15,  aliases: ["samusa", "samosa", "samoosa", "chamosa", "samose", "சமோசா"] },
  { id: 4,  name: "Vada",          price: 10,  aliases: ["vada", "wada", "vadai", "medhu vada", "வடை"] },
  { id: 5,  name: "Idli",          price: 8,   aliases: ["idly", "idli", "iddli", "ittly", "இட்லி"] },
  { id: 15, name: "Egg",           price: 7,   aliases: ["egg", "muttai", "eggs", "egge", "முட்டை"] },
];

// ─── FUZZY MATCHING ───────────────────────────────────────────
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

function fuzzyMatch(rawName) {
  const query = rawName.toLowerCase().trim();
  let bestItem = null, bestScore = 0;
  for (const item of INVENTORY) {
    for (const candidate of [item.name, ...item.aliases]) {
      const c = candidate.toLowerCase();
      const score = 1 - levenshtein(query, c) / Math.max(query.length, c.length);
      if (score > bestScore) { bestScore = score; bestItem = item; }
    }
  }
  return bestScore >= 0.6 ? { item: bestItem } : null;
}

// ─── SYSTEM PROMPT (Tamil/Tanglish Mapping) ───────────────────
const GEMINI_SYSTEM_PROMPT = `
Extract items and quantities from the text. 
Input can be English, Tanglish, or Tamil Script.
Inventory: ${JSON.stringify(INVENTORY.map(i => i.name))}

Tamil Numbers: Onnu/Ondru=1, Rendu/Irandu=2, Moonu=3, Naalu=4, Anju=5.

TASK:
1. Identify item and quantity.
2. Return ONLY a JSON array: [{"name":"English_Item_Name", "quantity": X}]
3. If Tamil script is used (e.g., 'தீ'), map it to English (e.g., 'Tea').
`;

// ─── CORE ROUTE ──────────────────────────────────────────────
app.post("/process-voice", upload.single("audio"), async (req, res) => {
  const tmpFilePath = req.file?.path;
  if (!req.file) return res.status(400).json({ success: false, error: "No audio file" });

  try {
    const namedPath = `${tmpFilePath}.m4a`;
    fs.renameSync(tmpFilePath, namedPath);

    console.log("[2/4] Running Local Faster-Whisper...");
    const rawText = await getLocalTranscription(namedPath);
    console.log(`[2/4] Whisper Output: "${rawText}"`);

    if (!rawText || rawText.toLowerCase().includes("you")) {
       if (fs.existsSync(namedPath)) fs.unlinkSync(namedPath);
       return res.json({ success: false, error: "Try speaking again.", transcript: rawText, items: [] });
    }

    console.log("[3/4] Sending to Gemini...");
    const geminiResponse = await geminiModel.generateContent(`${GEMINI_SYSTEM_PROMPT}\n\nOrder: "${rawText}"`);
    const geminiText = geminiResponse.response.text().trim();
    console.log("[3/4] Gemini Raw:", geminiText);

    let extractedItems = [];
    try {
      const cleaned = geminiText.replace(/```json|```/g, "").trim();
      extractedItems = JSON.parse(cleaned);
    } catch (e) { extractedItems = []; }

    const billItems = [];
    for (const ext of extractedItems) {
      const match = fuzzyMatch(ext.name);
      if (match) {
        billItems.push({
          id: match.item.id,
          name: match.item.name,
          quantity: parseInt(ext.quantity) || 1,
          unitPrice: match.item.price,
          subtotal: (parseInt(ext.quantity) || 1) * match.item.price,
        });
      }
    }

    const total = billItems.reduce((sum, i) => sum + i.subtotal, 0);
    if (fs.existsSync(namedPath)) fs.unlinkSync(namedPath);

    console.log(`[4/4] Final Bill: ₹${total}`);
    return res.json({ success: true, transcript: rawText, items: billItems, total: total });

  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, error: err.message });
  }
});

app.get("/inventory", (req, res) => res.json({ success: true, items: INVENTORY }));

// 🔧 CRITICAL: Change 127.0.0.1 to 0.0.0.0 for mobile access
app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on http://192.168.1.7:${PORT}`);
});