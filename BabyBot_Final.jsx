import { useState, useRef, useEffect, useCallback } from "react";

const SYSTEM_PROMPT = `Tu es BabyBot, un assistant IA spécialisé dans la parentalité.
RÈGLES: Réponds en français. Réponses SPÉCIFIQUES. Si le parent creuse, APPROFONDIS. Tiens compte de l'âge. Pose des questions de clarification. Si tu ne sais pas, recommande un pédiatre. Jamais de diagnostic. Max 200 mots. Termine par une ouverture.
DOMAINES: pleurs, sommeil, alimentation, développement, santé, bien-être parental, urgences.`;

const STORE_KEY = "babybot-v3";

async function loadData() {
  try {
    const r = await window.storage.get(STORE_KEY);
    return r ? JSON.parse(r.value) : null;
  } catch (e) {
    return null;
  }
}

async function saveData(d) {
  try {
    await window.storage.set(STORE_KEY, JSON.stringify(d));
  } catch (e) {
    console.warn("save failed", e);
  }
}

function audioBufferToWav(audioBuffer) {
  const sr = audioBuffer.sampleRate;
  const samples = audioBuffer.getChannelData(0);
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

async function convertToWav(blob) {
  const ctx = new AudioContext();
  try {
    const arrBuf = await blob.arrayBuffer();
    const decoded = await ctx.decodeAudioData(arrBuf);
    return audioBufferToWav(decoded);
  } finally {
    ctx.close();
  }
}

const COLORS = ["#7c3aed", "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899"];
const CRY_PALETTE = {
  faim: { color: "#f59e0b", icon: "🍼" },
  fatigue: { color: "#8b5cf6", icon: "😴" },
  douleur: { color: "#ef4444", icon: "😢" },
  peur: { color: "#3b82f6", icon: "😨" },
  inconfort: { color: "#06b6d4", icon: "😣" },
};

function AudioAnalyzer({ onResult, apiBase }) {
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [level, setLevel] = useState(0);
  const [duration, setDuration] = useState(0);
  const [infoText, setInfoText] = useState("");
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const animRef = useRef(null);
  const audioCtxRef = useRef(null);
  const fileInputRef = useRef(null);

  const cleanup = useCallback(() => {
    if (animRef.current) cancelAnimationFrame(animRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
    }
  }, []);

  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);

  function monitorLevel(stream) {
    const ctx = new AudioContext();
    audioCtxRef.current = ctx;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const dataArr = new Uint8Array(analyser.frequencyBinCount);
    function tick() {
      analyser.getByteFrequencyData(dataArr);
      const avg = dataArr.reduce((a, b) => a + b, 0) / dataArr.length / 255;
      setLevel(avg);
      animRef.current = requestAnimationFrame(tick);
    }
    tick();
  }

  async function analyzeLocally(wavBlob) {
    setInfoText("Analyse heuristique locale...");
    const ctx = new AudioContext();
    try {
      const decoded = await ctx.decodeAudioData(await wavBlob.arrayBuffer());
      const ch = decoded.getChannelData(0);
      let sumSq = 0;
      for (let i = 0; i < ch.length; i++) sumSq += ch[i] * ch[i];
      const rms = Math.sqrt(sumSq / ch.length);
      let zeroCross = 0;
      for (let i = 1; i < ch.length; i++) {
        if ((ch[i] >= 0) !== (ch[i - 1] >= 0)) zeroCross++;
      }
      const zcRate = zeroCross / decoded.duration;

      let res;
      if (rms < 0.01) {
        res = { type: "Silence", confidence: 0, color: "#64748b", icon: "🔇", advice: "Aucun son significatif détecté.", src: "local" };
      } else if (rms < 0.05) {
        res = { type: "Bruit ambiant", confidence: 0.2, color: "#94a3b8", icon: "🔉", advice: "Son de fond sans pleurs identifiés.", src: "local" };
      } else if (zcRate > 3000 && rms > 0.08) {
        res = { type: "Pleurs détectés", confidence: 0.6, color: "#f59e0b", icon: "😢", advice: "Signal compatible avec des pleurs. Le serveur ML donnerait plus de détails.", src: "local" };
      } else {
        res = { type: "Son non identifié", confidence: 0.25, color: "#8b5cf6", icon: "🔊", advice: "Son capté (" + (rms * 100).toFixed(0) + "%) mais non classifiable sans modèle ML.", src: "local" };
      }
      setResult(res);
      setStatus("result");
      setInfoText("Analyse locale terminée");
      if (onResult) onResult(res);
    } catch (err) {
      setResult({ type: "Erreur", confidence: 0, color: "#ef4444", icon: "⚠️", advice: err.message, src: "erreur" });
      setStatus("result");
    } finally {
      ctx.close();
    }
  }

  async function sendToServer(wavBlob, filename) {
    setStatus("analyzing");
    setInfoText("Envoi au serveur...");
    setResult(null);

    if (apiBase) {
      try {
        const fd = new FormData();
        fd.append("audio", wavBlob, filename);
        fd.append("parent_id", "web-user");
        const resp = await fetch(apiBase + "/api/analyze", { method: "POST", body: fd });
        if (resp.ok) {
          const data = await resp.json();
          const p = CRY_PALETTE[data.cry_type] || { color: "#94a3b8", icon: "🔊" };
          const res = {
            type: data.cry_type.charAt(0).toUpperCase() + data.cry_type.slice(1),
            confidence: data.confidence,
            color: p.color,
            icon: p.icon,
            advice: data.advice,
            src: "serveur (ML)",
          };
          setResult(res);
          setStatus("result");
          setInfoText("Analyse serveur terminée");
          if (onResult) onResult(res);
          return;
        }
      } catch (e) {
        setInfoText("Serveur indisponible, analyse locale...");
      }
    }
    await analyzeLocally(wavBlob);
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      streamRef.current = stream;
      chunksRef.current = [];
      setStatus("recording");
      setDuration(0);
      setResult(null);
      setInfoText("Enregistrement... rapprochez le micro.");
      monitorLevel(stream);

      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        cleanup();
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        try {
          setInfoText("Conversion WAV...");
          const wav = await convertToWav(blob);
          await sendToServer(wav, "micro.wav");
        } catch (err) {
          setResult({ type: "Erreur", confidence: 0, color: "#ef4444", icon: "⚠️", advice: err.message, src: "erreur" });
          setStatus("result");
        }
      };
      recorder.start();
      let sec = 0;
      timerRef.current = setInterval(() => {
        sec++;
        setDuration(sec);
      }, 1000);
    } catch (err) {
      setStatus("idle");
      setInfoText(err.name === "NotAllowedError" ? "🚫 Accès micro refusé" : "Erreur: " + err.message);
    }
  }

  function stopRecording() {
    if (timerRef.current) clearInterval(timerRef.current);
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
  }

  async function handleFileUpload(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    e.target.value = "";
    setResult(null);
    setInfoText("Fichier: " + file.name);
    try {
      const wav = file.type === "audio/wav" || file.name.endsWith(".wav") ? file : await convertToWav(file);
      await sendToServer(wav, file.name.replace(/\.[^.]+$/, ".wav"));
    } catch (err) {
      setResult({ type: "Erreur", confidence: 0, color: "#ef4444", icon: "⚠️", advice: err.message, src: "erreur" });
      setStatus("result");
    }
  }

  const isAnalyzing = status === "analyzing";
  const isRecording = status === "recording";

  return (
    <div style={{ background: "#141c2e", borderRadius: 14, padding: 18, border: "1px solid #1e2d45" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 20 }}>🎤</span>
          <span style={{ color: "#e2e8f0", fontWeight: 700, fontSize: 15 }}>Analyse des Pleurs</span>
        </div>
        {isRecording && (
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#ef4444", animation: "pulse 1s infinite" }} />
            <span style={{ color: "#ef4444", fontSize: 11, fontWeight: 600 }}>REC {duration}s</span>
          </div>
        )}
      </div>

      {isRecording && (
        <div style={{ display: "flex", alignItems: "end", justifyContent: "center", gap: 2, height: 36, marginBottom: 12 }}>
          {Array.from({ length: 16 }).map((_, i) => (
            <div key={i} style={{ width: 5, borderRadius: 2, transition: "height 0.08s",
              height: Math.max(3, level * 36 * (0.5 + Math.sin(i * 0.7 + Date.now() * 0.004) * 0.5)),
              background: level > 0.3 ? "#ef4444" : level > 0.1 ? "#f59e0b" : "#3b82f6" }} />
          ))}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 8 }}>
        {isRecording ? (
          <button onClick={stopRecording} style={{ gridColumn: "1/3", padding: 13, borderRadius: 11, border: "2px solid #ef4444", background: "#ef444415", color: "#ef4444", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
            ⏹️ Arrêter ({duration}s)
          </button>
        ) : (
          <>
            <button onClick={startRecording} disabled={isAnalyzing} style={{ padding: 13, borderRadius: 11, border: "none", background: isAnalyzing ? "#1e2d45" : "linear-gradient(135deg, #7c3aed, #3b82f6)", color: "#fff", fontSize: 13, fontWeight: 600, cursor: isAnalyzing ? "default" : "pointer" }}>
              🎙️ Enregistrer au micro
            </button>
            <button onClick={() => fileInputRef.current && fileInputRef.current.click()} disabled={isAnalyzing} style={{ padding: 13, borderRadius: 11, border: "none", background: isAnalyzing ? "#1e2d45" : "linear-gradient(135deg, #8b5cf6, #06b6d4)", color: "#fff", fontSize: 13, fontWeight: 600, cursor: isAnalyzing ? "default" : "pointer" }}>
              📁 Choisir un fichier
            </button>
          </>
        )}
      </div>
      <input ref={fileInputRef} type="file" accept="audio/*" onChange={handleFileUpload} style={{ display: "none" }} />

      {infoText && (
        <p style={{ color: "#64748b", fontSize: 11, margin: "4px 0 0" }}>
          {isAnalyzing && <span style={{ display: "inline-block", width: 12, height: 12, border: "2px solid #7c3aed44", borderTop: "2px solid #7c3aed", borderRadius: "50%", animation: "spin 0.7s linear infinite", verticalAlign: "middle", marginRight: 6 }} />}
          {infoText}
        </p>
      )}

      {result && status === "result" && (
        <div style={{ marginTop: 14, background: "#0a0f1e", borderRadius: 11, padding: 14, borderLeft: "4px solid " + result.color }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <span style={{ fontSize: 26 }}>{result.icon}</span>
            <div>
              <div style={{ color: result.color, fontWeight: 700, fontSize: 16 }}>{result.type}</div>
              <div style={{ color: "#475569", fontSize: 10 }}>
                {result.confidence > 0 ? "Confiance: " + (result.confidence * 100).toFixed(0) + "%" : ""} · {result.src}
              </div>
            </div>
          </div>
          {result.confidence > 0 && (
            <div style={{ height: 4, borderRadius: 2, background: "#1e2d45", overflow: "hidden", marginBottom: 8 }}>
              <div style={{ height: "100%", width: result.confidence * 100 + "%", background: result.color, borderRadius: 2 }} />
            </div>
          )}
          <p style={{ color: "#cbd5e1", fontSize: 12, lineHeight: 1.55, margin: 0 }}>{result.advice}</p>
        </div>
      )}
    </div>
  );
}

export default function BabyBotApp() {
  const [profiles, setProfiles] = useState([]);
  const [activePid, setActivePid] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [tab, setTab] = useState("chat");
  const [stats, setStats] = useState({ positive: 0, negative: 0, total: 0, feedbacks: [] });
  const [screen, setScreen] = useState("loading");
  const [apiBase, setApiBase] = useState("");
  const [editP, setEditP] = useState(null);
  const msgEndRef = useRef(null);
  const inputRef = useRef(null);
  const convRef = useRef([]);

  const activeProfile = profiles.find((p) => p.id === activePid) || null;

  useEffect(() => {
    loadData().then((d) => {
      if (d) {
        if (d.profiles) setProfiles(d.profiles);
        if (d.stats) setStats(d.stats);
        if (d.apiBase) setApiBase(d.apiBase);
      }
      setScreen("home");
    });
  }, []);

  useEffect(() => {
    if (screen !== "loading") {
      saveData({ profiles, stats, apiBase });
    }
  }, [profiles, stats, apiBase, screen]);

  useEffect(() => {
    if (msgEndRef.current) {
      msgEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isTyping]);

  async function callClaude(userMsg) {
    const history = convRef.current.slice(-14).map((m) => ({ role: m.role, content: m.content }));
    history.push({ role: "user", content: userMsg });
    let sys = SYSTEM_PROMPT;
    if (activeProfile && (activeProfile.name || activeProfile.age)) {
      sys += "\n\nENFANT: " + (activeProfile.name || "non précisé") + (activeProfile.age ? ", " + activeProfile.age : "") + ".";
    }
    const resp = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "claude-sonnet-4-20250514", max_tokens: 1000, system: sys, messages: history }),
    });
    if (!resp.ok) throw new Error("API " + resp.status);
    const data = await resp.json();
    const texts = (data.content || []).filter((b) => b.type === "text").map((b) => b.text);
    return texts.join("\n") || "";
  }

  async function sendMessage(text) {
    const msg = text || input.trim();
    if (!msg || isTyping) return;
    setInput("");
    const userMsg = { id: Date.now(), role: "user", content: msg, feedback: null };
    setMessages((prev) => [...prev, userMsg]);
    convRef.current.push({ role: "user", content: msg });
    setIsTyping(true);
    try {
      const response = await callClaude(msg);
      const botMsg = { id: Date.now() + 1, role: "assistant", content: response, feedback: null };
      setMessages((prev) => [...prev, botMsg]);
      convRef.current.push({ role: "assistant", content: response });
    } catch (err) {
      setMessages((prev) => [...prev, { id: Date.now() + 1, role: "assistant", content: "Problème technique, réessayez. 🙏", feedback: null }]);
    }
    setIsTyping(false);
    setTimeout(() => { if (inputRef.current) inputRef.current.focus(); }, 100);
  }

  function handleFeedback(msgId, type) {
    setMessages((prev) => prev.map((m) => (m.id === msgId ? { ...m, feedback: type } : m)));
    setStats((prev) => ({
      positive: prev.positive + (type === "up" ? 1 : 0),
      negative: prev.negative + (type === "down" ? 1 : 0),
      total: prev.total + 1,
      feedbacks: [...(prev.feedbacks || []), { type, time: new Date().toISOString(), child: activeProfile ? activeProfile.name : "" }],
    }));
  }

  function openProfile(p) {
    setActivePid(p.id);
    setTab("chat");
    const greeting = p.name
      ? "Bonjour ! Parlons de " + p.name + (p.age ? " (" + p.age + ")" : "") + " — qu'est-ce qui vous préoccupe ?"
      : "Bonjour ! Comment puis-je vous aider ?";
    setMessages([{ id: 1, role: "assistant", content: greeting, feedback: null }]);
    convRef.current = [{ role: "assistant", content: greeting }];
    setScreen("chat");
  }

  function saveProfile() {
    if (!editP) return;
    setProfiles((prev) => {
      const idx = prev.findIndex((x) => x.id === editP.id);
      if (idx >= 0) { const n = [...prev]; n[idx] = editP; return n; }
      return [...prev, editP];
    });
    setEditP(null);
  }

  function deleteProfile(id) {
    setProfiles((prev) => prev.filter((x) => x.id !== id));
    if (activePid === id) setActivePid(null);
  }

  async function resetAll() {
    setProfiles([]);
    setStats({ positive: 0, negative: 0, total: 0, feedbacks: [] });
    setApiBase("");
    setMessages([]);
    convRef.current = [];
    setActivePid(null);
    setScreen("home");
    try { await window.storage.delete(STORE_KEY); } catch (e) { /* ok */ }
  }

  function renderMd(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/\n/g, "<br/>");
  }

  function handleAudioResult(r) {
    if (r && r.type !== "Silence" && r.type !== "Erreur") {
      const confStr = r.confidence > 0 ? " (" + (r.confidence * 100).toFixed(0) + "%)" : "";
      setMessages((prev) => [...prev, {
        id: Date.now(), role: "assistant",
        content: "🔔 **" + r.type + "**" + confStr + "\n\n" + r.advice,
        feedback: null,
      }]);
    }
  }

  const globalCSS = `
    @keyframes spin { to { transform: rotate(360deg) } }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(5px) } to { opacity: 1; transform: translateY(0) } }
    @keyframes pulse { 0%, 100% { opacity: 0.4 } 50% { opacity: 1 } }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(15px) } to { opacity: 1; transform: translateY(0) } }
    *::-webkit-scrollbar { width: 5px }
    *::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 3px }
  `;

  if (screen === "loading") {
    return (
      <div style={{ height: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#0a0f1e", fontFamily: "'Segoe UI', system-ui, sans-serif" }}>
        <style>{globalCSS}</style>
        <div style={{ color: "#64748b" }}>Chargement...</div>
      </div>
    );
  }

  // ═══════════════════════════════════════════
  // HOME SCREEN
  // ═══════════════════════════════════════════
  if (screen === "home") {
    return (
      <div style={{ minHeight: "100vh", background: "#0a0f1e", fontFamily: "'Segoe UI', system-ui, sans-serif", padding: 20 }}>
        <style>{globalCSS}</style>
        <div style={{ maxWidth: 480, margin: "0 auto", animation: "fadeUp 0.4s" }}>
          <div style={{ textAlign: "center", marginBottom: 28, marginTop: 20 }}>
            <div style={{ fontSize: 52, marginBottom: 8 }}>👶</div>
            <h1 style={{ color: "#e2e8f0", fontSize: 28, fontWeight: 800, margin: 0, fontFamily: "Georgia, serif" }}>BabyBot</h1>
            <p style={{ color: "#64748b", fontSize: 13, margin: "6px 0 0" }}>Assistant parental intelligent</p>
            {stats.total > 0 && <p style={{ color: "#7c3aed", fontSize: 11, margin: "8px 0 0" }}>📊 {stats.total} feedbacks sauvegardés</p>}
          </div>

          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
            <span style={{ color: "#94a3b8", fontSize: 12, fontWeight: 600, letterSpacing: 0.5 }}>MES ENFANTS</span>
            <button onClick={() => setEditP({ id: Date.now().toString(), name: "", age: "", createdAt: new Date().toISOString() })}
              style={{ padding: "5px 14px", borderRadius: 8, border: "1px solid #1e2d45", background: "#141c2e", color: "#7c3aed", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>
              + Ajouter
            </button>
          </div>

          {profiles.length === 0 && !editP && (
            <div style={{ background: "#141c2e", borderRadius: 14, padding: 24, textAlign: "center", border: "1px dashed #1e2d45", marginBottom: 12 }}>
              <div style={{ fontSize: 36, marginBottom: 8 }}>🍼</div>
              <p style={{ color: "#94a3b8", fontSize: 13, margin: 0 }}>Ajoutez votre premier enfant pour commencer !</p>
            </div>
          )}

          {editP && (
            <div style={{ background: "#141c2e", borderRadius: 14, padding: 18, border: "1px solid #7c3aed40", marginBottom: 12, animation: "fadeIn 0.2s" }}>
              <div style={{ display: "flex", gap: 10, marginBottom: 10 }}>
                <div style={{ flex: 1 }}>
                  <label style={{ color: "#94a3b8", fontSize: 10, fontWeight: 600, display: "block", marginBottom: 4 }}>Prénom</label>
                  <input value={editP.name} onChange={(e) => setEditP({ ...editP, name: e.target.value })} placeholder="Emma"
                    style={{ width: "100%", padding: "8px 12px", borderRadius: 8, border: "1px solid #1e2d45", background: "#0a0f1e", color: "#e2e8f0", fontSize: 13, outline: "none", boxSizing: "border-box" }} />
                </div>
                <div style={{ flex: 1 }}>
                  <label style={{ color: "#94a3b8", fontSize: 10, fontWeight: 600, display: "block", marginBottom: 4 }}>Âge</label>
                  <input value={editP.age} onChange={(e) => setEditP({ ...editP, age: e.target.value })} placeholder="4 mois"
                    style={{ width: "100%", padding: "8px 12px", borderRadius: 8, border: "1px solid #1e2d45", background: "#0a0f1e", color: "#e2e8f0", fontSize: 13, outline: "none", boxSizing: "border-box" }} />
                </div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={saveProfile} style={{ flex: 1, padding: 10, borderRadius: 9, border: "none", background: "linear-gradient(135deg, #7c3aed, #3b82f6)", color: "#fff", fontSize: 13, fontWeight: 600, cursor: "pointer" }}>Enregistrer</button>
                <button onClick={() => setEditP(null)} style={{ padding: 10, borderRadius: 9, border: "1px solid #1e2d45", background: "transparent", color: "#64748b", fontSize: 13, cursor: "pointer" }}>Annuler</button>
              </div>
            </div>
          )}

          <div style={{ display: "grid", gap: 10, marginBottom: 16 }}>
            {profiles.map((p, i) => (
              <div key={p.id} onClick={() => openProfile(p)}
                style={{ background: "#141c2e", borderRadius: 14, padding: 16, border: "1px solid #1e2d45", display: "flex", alignItems: "center", gap: 14, cursor: "pointer", animation: "fadeIn 0.2s" }}>
                <div style={{ width: 48, height: 48, borderRadius: 12, background: COLORS[i % COLORS.length] + "20", color: COLORS[i % COLORS.length], display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, fontWeight: 700, flexShrink: 0 }}>
                  {p.name ? p.name[0].toUpperCase() : "?"}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ color: "#e2e8f0", fontWeight: 700, fontSize: 15 }}>{p.name || "Sans nom"}</div>
                  <div style={{ color: "#64748b", fontSize: 11 }}>{p.age || "Âge non précisé"}</div>
                </div>
                <div style={{ display: "flex", gap: 4 }}>
                  <button onClick={(e) => { e.stopPropagation(); setEditP({ ...p }); }} style={{ padding: "5px 8px", borderRadius: 6, border: "1px solid #1e2d45", background: "transparent", color: "#64748b", fontSize: 11, cursor: "pointer" }}>✏️</button>
                  <button onClick={(e) => { e.stopPropagation(); if (confirm("Supprimer " + (p.name || "ce profil") + " ?")) deleteProfile(p.id); }} style={{ padding: "5px 8px", borderRadius: 6, border: "1px solid #1e2d45", background: "transparent", color: "#ef444480", fontSize: 11, cursor: "pointer" }}>🗑️</button>
                </div>
              </div>
            ))}
          </div>

          <button onClick={() => openProfile({ id: "anon", name: "", age: "" })}
            style={{ width: "100%", padding: 13, borderRadius: 11, border: "1px solid #1e2d45", background: "transparent", color: "#94a3b8", fontSize: 13, cursor: "pointer", marginBottom: 16 }}>
            💬 Discussion rapide sans profil
          </button>

          <div style={{ background: "#141c2e", borderRadius: 12, padding: 14, border: "1px solid #1e2d45", marginBottom: 12 }}>
            <label style={{ color: "#94a3b8", fontSize: 10, fontWeight: 600, display: "block", marginBottom: 5 }}>URL Serveur BabyCry (optionnel)</label>
            <input value={apiBase} onChange={(e) => setApiBase(e.target.value.replace(/\/$/, ""))} placeholder="http://localhost:8000"
              style={{ width: "100%", padding: "8px 12px", borderRadius: 8, border: "1px solid #1e2d45", background: "#0a0f1e", color: "#e2e8f0", fontSize: 12, outline: "none", boxSizing: "border-box" }} />
          </div>

          <button onClick={() => { if (confirm("Supprimer TOUTES les données ?")) resetAll(); }}
            style={{ width: "100%", padding: 10, borderRadius: 9, border: "1px solid #ef444430", background: "transparent", color: "#ef444480", fontSize: 11, cursor: "pointer" }}>
            🔄 Réinitialiser toutes les données
          </button>
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════
  // CHAT SCREEN
  // ═══════════════════════════════════════════
  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "#0a0f1e", fontFamily: "'Segoe UI', system-ui, sans-serif" }}>
      <style>{globalCSS}</style>

      <div style={{ background: "#0f1729", padding: "10px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0, borderBottom: "1px solid #1e2d45" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button onClick={() => { setScreen("home"); setMessages([]); convRef.current = []; }}
            style={{ background: "transparent", border: "none", color: "#64748b", fontSize: 18, cursor: "pointer", padding: "2px 6px" }}>←</button>
          <span style={{ fontSize: 20 }}>👶</span>
          <div>
            <span style={{ color: "#e2e8f0", fontWeight: 700, fontSize: 14 }}>BabyBot</span>
            {activeProfile && activeProfile.name && <span style={{ color: "#7c3aed", fontSize: 11, marginLeft: 6 }}>{activeProfile.name}{activeProfile.age ? ", " + activeProfile.age : ""}</span>}
          </div>
        </div>
        <div style={{ display: "flex", gap: 3 }}>
          {[{ id: "chat", label: "💬" }, { id: "analyze", label: "🎤" }, { id: "stats", label: "📊" }].map((t) => (
            <button key={t.id} onClick={() => setTab(t.id)}
              style={{ padding: "5px 10px", borderRadius: 7, border: "none", fontSize: 13, cursor: "pointer", background: tab === t.id ? "#1e2d45" : "transparent", color: tab === t.id ? "#e2e8f0" : "#64748b" }}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {tab === "stats" ? (
        <div style={{ flex: 1, padding: 14, overflowY: "auto" }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 14 }}>
            {[
              { label: "Total", value: stats.total, color: "#3b82f6", icon: "📋" },
              { label: "👍", value: stats.positive, color: "#10b981", icon: "👍" },
              { label: "👎", value: stats.negative, color: "#ef4444", icon: "👎" },
            ].map((s) => (
              <div key={s.label} style={{ background: "#141c2e", borderRadius: 11, padding: 14, textAlign: "center", borderTop: "3px solid " + s.color, border: "1px solid #1e2d45" }}>
                <div style={{ fontSize: 20 }}>{s.icon}</div>
                <div style={{ color: s.color, fontSize: 24, fontWeight: 700, margin: "4px 0" }}>{s.value}</div>
                <div style={{ color: "#64748b", fontSize: 10 }}>{s.label}</div>
              </div>
            ))}
          </div>
          <div style={{ background: "#141c2e", borderRadius: 11, padding: 14, border: "1px solid #1e2d45", marginBottom: 10 }}>
            <div style={{ color: "#f59e0b", fontSize: 12, fontWeight: 600, marginBottom: 6 }}>🔄 Pipeline LoRA — {stats.total}/100</div>
            <div style={{ height: 5, background: "#1e2d45", borderRadius: 3, overflow: "hidden" }}>
              <div style={{ height: "100%", width: Math.min(stats.total, 100) + "%", background: "linear-gradient(90deg, #f59e0b, #10b981)", borderRadius: 3 }} />
            </div>
          </div>
          <div style={{ background: "#141c2e", borderRadius: 11, padding: 14, border: "1px solid #1e2d45", marginBottom: 10 }}>
            <div style={{ color: "#10b981", fontSize: 12, fontWeight: 600, marginBottom: 4 }}>💾 Persistance cross-device</div>
            <p style={{ color: "#64748b", fontSize: 11, margin: 0 }}>Feedbacks sauvegardés et retrouvés même depuis un autre appareil.</p>
          </div>
          {stats.feedbacks && stats.feedbacks.length > 0 && (
            <div style={{ background: "#141c2e", borderRadius: 11, padding: 14, border: "1px solid #1e2d45" }}>
              <div style={{ color: "#94a3b8", fontSize: 11, fontWeight: 600, marginBottom: 6 }}>Derniers feedbacks</div>
              {stats.feedbacks.slice(-10).reverse().map((fb, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 0", borderBottom: "1px solid #1e2d4530" }}>
                  <span style={{ fontSize: 13 }}>{fb.type === "up" ? "👍" : "👎"}</span>
                  {fb.child && <span style={{ color: "#7c3aed", fontSize: 9 }}>{fb.child}</span>}
                  <span style={{ color: "#64748b", fontSize: 9 }}>{new Date(fb.time).toLocaleString("fr-FR", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : tab === "analyze" ? (
        <div style={{ flex: 1, padding: 14, overflowY: "auto" }}>
          <AudioAnalyzer onResult={handleAudioResult} apiBase={apiBase} />
        </div>
      ) : (
        <>
          <div style={{ flex: 1, overflowY: "auto", padding: 14 }}>
            {messages.map((m) => (
              <div key={m.id} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start", marginBottom: 12, animation: "fadeIn 0.25s" }}>
                <div style={{ maxWidth: "84%" }}>
                  {m.role === "assistant" && (
                    <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 3 }}>
                      <span style={{ fontSize: 12 }}>👶</span>
                      <span style={{ color: "#7c3aed", fontSize: 10, fontWeight: 600 }}>BabyBot</span>
                    </div>
                  )}
                  <div
                    style={{
                      padding: "10px 14px", borderRadius: 13,
                      background: m.role === "user" ? "linear-gradient(135deg, #3b82f6, #2563eb)" : "#141c2e",
                      border: m.role === "assistant" ? "1px solid #1e2d45" : "none",
                      color: "#e2e8f0", fontSize: 12.5, lineHeight: 1.6,
                      borderBottomRightRadius: m.role === "user" ? 3 : 13,
                      borderBottomLeftRadius: m.role === "assistant" ? 3 : 13,
                    }}
                    dangerouslySetInnerHTML={{ __html: renderMd(m.content) }}
                  />
                  {m.role === "assistant" && (
                    <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
                      {["up", "down"].map((type) => (
                        <button key={type} onClick={() => handleFeedback(m.id, type)}
                          style={{
                            padding: "3px 9px", borderRadius: 6, fontSize: 12, cursor: "pointer",
                            border: "1px solid " + (m.feedback === type ? (type === "up" ? "#10b98140" : "#ef444440") : "#1e2d45"),
                            background: m.feedback === type ? (type === "up" ? "#10b98115" : "#ef444415") : "#0a0f1e",
                            color: m.feedback === type ? (type === "up" ? "#10b981" : "#ef4444") : "#64748b",
                          }}>
                          {type === "up" ? "👍" : "👎"}
                        </button>
                      ))}
                      {m.feedback && <span style={{ color: "#334155", fontSize: 9, alignSelf: "center", marginLeft: 3 }}>✓</span>}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isTyping && (
              <div style={{ display: "flex", alignItems: "center", gap: 5, padding: "6px 0" }}>
                <span style={{ fontSize: 12 }}>👶</span>
                {[0, 1, 2].map((i) => (
                  <div key={i} style={{ width: 5, height: 5, borderRadius: "50%", background: "#7c3aed", animation: "pulse 1s infinite " + i * 0.15 + "s" }} />
                ))}
              </div>
            )}
            <div ref={msgEndRef} />
          </div>

          {messages.length <= 2 && (
            <div style={{ padding: "0 14px 5px", display: "flex", flexWrap: "wrap", gap: 4 }}>
              {["Mon bébé pleure la nuit", "Quand commencer les purées ?", "Calmer les coliques", "Régurgitations normales ?"].map((q) => (
                <button key={q} onClick={() => sendMessage(q)}
                  style={{ padding: "5px 10px", borderRadius: 16, border: "1px solid #1e2d45", background: "#141c2e", color: "#94a3b8", fontSize: 10, cursor: "pointer" }}>
                  {q}
                </button>
              ))}
            </div>
          )}

          <div style={{ padding: 10, background: "#0f1729", borderTop: "1px solid #1e2d45", display: "flex", gap: 7, flexShrink: 0 }}>
            <input ref={inputRef} value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) sendMessage(); }}
              placeholder="Posez votre question..."
              disabled={isTyping}
              style={{ flex: 1, padding: "10px 13px", borderRadius: 9, border: "1px solid #1e2d45", background: "#0a0f1e", color: "#e2e8f0", fontSize: 13, outline: "none", opacity: isTyping ? 0.5 : 1 }}
            />
            <button onClick={() => sendMessage()} disabled={isTyping || !input.trim()}
              style={{
                padding: "10px 16px", borderRadius: 9, border: "none",
                background: input.trim() && !isTyping ? "linear-gradient(135deg, #7c3aed, #3b82f6)" : "#1e2d45",
                color: "#fff", fontSize: 14, fontWeight: 600,
                cursor: input.trim() && !isTyping ? "pointer" : "default",
              }}>
              ↑
            </button>
          </div>
        </>
      )}
    </div>
  );
}
