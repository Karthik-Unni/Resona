import React, { useState, useEffect } from 'react';
import {
  StyleSheet, View, Text, ScrollView, TouchableOpacity,
  Alert, ActivityIndicator, StatusBar, TextInput
} from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';

// ─── CONFIG ───────────────────────────────────────────────────────────────────
// Replace with your computer's local IP (shown when you run api.py)
const API_BASE = 'http://10.246.40.235:5000/api';

// ─── COLORS (Stitch Design System) ───────────────────────────────────────────
const C = {
  bg: '#F8F9FA',
  surface: '#FFFFFF',
  border: '#E5E7EB',
  borderMed: '#D1D5DB',
  primary: '#0F4C81',
  primaryDark: '#0A3357',
  textPrimary: '#1E2229',
  textSecondary: '#475569',
  textMuted: '#6B7280',
  textMono: '#374151',
  green: '#15803D',
  greenBg: '#DCFCE7',
  greenBorder: '#86EFAC',
  red: '#B91C1C',
  redBg: '#FEE2E2',
  amber: '#B45309',
  amberBg: '#FEF3C7',
  blue: '#1D4ED8',
  blueBg: '#DBEAFE',
  headerBg: '#FFFFFF',
};

// ─── STATUS STYLE HELPER ──────────────────────────────────────────────────────
function getStatusStyle(status) {
  if (!status) return { bg: C.greenBg, color: C.green, dot: C.green };
  const s = status.toUpperCase();
  if (s.includes('ANOMALY')) return { bg: C.redBg, color: C.red, dot: C.red };
  if (s.includes('UNKNOWN')) return { bg: C.amberBg, color: C.amber, dot: C.amber };
  if (s.includes('NEW')) return { bg: C.blueBg, color: C.blue, dot: C.blue };
  return { bg: C.greenBg, color: C.green, dot: C.green };
}

// ─── MINI SPECTROGRAM ─────────────────────────────────────────────────────────
function MiniSpectrogram({ status }) {
  const bars = 40;
  const barData = Array.from({ length: bars }, (_, i) => {
    const base = 0.3 + Math.sin(i * 0.4) * 0.2 + Math.random() * 0.1;
    const anomalyBoost = status?.includes('ANOMALY') ? Math.random() * 0.5 : 0;
    const unknownBoost = status?.includes('UNKNOWN') ? (Math.random() > 0.8 ? 0.7 : 0) : 0;
    return Math.min(1, base + anomalyBoost + unknownBoost);
  });

  return (
    <View style={styles.spectrogramContainer}>
      <View style={styles.spectrogramBars}>
        {barData.map((h, i) => (
          <View key={i} style={[styles.spectrogramBar, {
            height: `${Math.max(8, h * 100)}%`,
            backgroundColor: status?.includes('ANOMALY') ? '#EF4444' :
              status?.includes('UNKNOWN') ? '#F59E0B' : '#10B981',
            opacity: 0.6 + h * 0.4,
          }]} />
        ))}
      </View>
      <Text style={styles.spectrogramLabel}>64 Mel · 16kHz · STFT</Text>
    </View>
  );
}

// ─── MAIN APP ────────────────────────────────────────────────────────────────
function MainApp() {
  const [screen, setScreen] = useState('overview'); // 'overview' | 'review' | 'demo'
  const [state, setState] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [reviews, setReviews] = useState([]);
  const [evalMetrics, setEvalMetrics] = useState({});
  const [latestInference, setLatestInference] = useState(null);

  const [loading, setLoading] = useState(false);
  const [notes, setNotes] = useState('');
  const [selectedCondition, setSelectedCondition] = useState(null);
  const [submitted, setSubmitted] = useState(false);
  const [activeReview, setActiveReview] = useState(null);

  useEffect(() => { 
    fetchAll(); 
    const timer = setInterval(fetchAll, 2000);
    return () => clearInterval(timer);
  }, []);

  async function fetchAll() {
    try {
      const [resStatus, resAlerts, resReviews, resEval, resLatest] = await Promise.all([
        fetch(`${API_BASE}/status`).catch(()=>null),
        fetch(`${API_BASE}/alerts`).catch(()=>null),
        fetch(`${API_BASE}/reviews/pending`).catch(()=>null),
        fetch(`${API_BASE}/evaluation`).catch(()=>null),
        fetch(`${API_BASE}/latest`).catch(()=>null)
      ]);
      
      if(resStatus && resStatus.ok) setState(await resStatus.json());
      if(resAlerts && resAlerts.ok) setAlerts(await resAlerts.json());
      
      if(resReviews && resReviews.ok) {
        const revs = await resReviews.json();
        setReviews(revs);
        setActiveReview(prev => {
          if (!prev && revs.length > 0) return revs[0];
          if (prev && !revs.find(r => r.id === prev.id)) return revs.length > 0 ? revs[0] : null;
          return prev;
        });
      }
      
      if(resEval && resEval.ok) {
        const d = await resEval.json();
        if(d.available) setEvalMetrics(d.results || {});
      }

      if(resLatest && resLatest.ok) {
        const d = await resLatest.json();
        if(d.available) setLatestInference(d.result);
      }
    } catch (e) { /* silent fail */ }
  }

  async function runTestCase(tc) {
    setLoading(true);
    setScreen('demo');
    try {
      const res = await fetch(`${API_BASE}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ test_case: tc }),
      });
      const data = await res.json();
      if (!data.success) {
        Alert.alert('Error', data.error || 'Failed to simulate');
      }
    } catch (e) {
      Alert.alert('Error', 'Backend offline. Start api.py on your computer.');
    } finally { setLoading(false); }
  }

  async function submitVerification() {
    if (!selectedCondition) {
      Alert.alert('Required', 'Please select the machine condition.');
      return;
    }
    if (!activeReview) return;

    const labelMap = { normal: 'NORMAL', anomalous: 'ANOMALOUS', new_condition: 'NEW CONDITION' };
    setLoading(true);
    try {
      await fetch(`${API_BASE}/human_feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: labelMap[selectedCondition], review_id: activeReview.id, notes }),
      });
      setSubmitted(true);
      setTimeout(() => setSubmitted(false), 4000);
      setSelectedCondition(null);
      setNotes('');
      // Refresh to clear review
      fetchAll();
    } catch (e) {
      Alert.alert('Error', 'Could not reach backend.');
    } finally { setLoading(false); }
  }

  // Derived states for UI
  const systemStatusStr = state?.active_alerts_count > 0 ? 'ATTENTION REQUIRED' : (state?.system || 'OFFLINE');
  const systemStatusColor = state?.active_alerts_count > 0 ? C.red : (state?.system ? C.green : C.textMuted);
  const bgStatusColor = state?.active_alerts_count > 0 ? C.redBg : (state?.system ? C.greenBg : C.border);
  
  const resonaMetrics = evalMetrics['RESONA_Replay'] || {};
  const naiveMetrics = evalMetrics['Naive_FT'] || {};

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="dark-content" backgroundColor={C.headerBg} />

      {/* ── Header ── */}
      <View style={styles.header}>
        <View>
          <Text style={styles.brand}>RESONA</Text>
          <Text style={styles.brandSub}>ACOUSTIC HMI · MOBILE</Text>
        </View>
        <View style={styles.headerRight}>
          <View style={[styles.statusPill, { backgroundColor: bgStatusColor, borderColor: systemStatusColor }]}>
            <View style={[styles.statusDot, { backgroundColor: systemStatusColor }]} />
            <Text style={[styles.statusPillText, { color: systemStatusColor }]}>{systemStatusStr}</Text>
          </View>
        </View>
      </View>

      {/* ── Tab Bar ── */}
      <View style={styles.tabBar}>
        {[['overview','Overview'], ['review','Review'], ['demo','Demo']].map(([id, label]) => (
          <TouchableOpacity key={id} style={[styles.tab, screen === id && styles.tabActive]} onPress={() => setScreen(id)}>
            <Text style={[styles.tabText, screen === id && styles.tabTextActive]}>{label}</Text>
            {id === 'review' && reviews.length > 0 && (
              <View style={styles.badgeCount}><Text style={styles.badgeCountText}>{reviews.length}</Text></View>
            )}
          </TouchableOpacity>
        ))}
      </View>

      {loading && (
        <View style={styles.loadingBar}>
          <ActivityIndicator size="small" color={C.primary} />
          <Text style={styles.loadingText}>Processing...</Text>
        </View>
      )}

      <ScrollView style={styles.scroll} showsVerticalScrollIndicator={false}>

        {/* ══════════ OVERVIEW ══════════ */}
        {screen === 'overview' && (
          <View style={styles.page}>
            <Text style={styles.pageTitle}>Plant Overview</Text>
            <Text style={styles.pageSubtitle}>MIMII Dataset · 0 dB SNR · Continual Learning</Text>

            {/* KPI Cards */}
            <View style={styles.kpiGrid}>
              <View style={styles.kpiCard}>
                <Text style={styles.kpiLabel}>RESONA Forgetting</Text>
                <Text style={[styles.kpiValue, { color: C.green }]}>{resonaMetrics.avg_forgetting || '—'}</Text>
                <Text style={styles.kpiSub}>vs Naive: {naiveMetrics.avg_forgetting || '—'}</Text>
              </View>
              <View style={styles.kpiCard}>
                <Text style={styles.kpiLabel}>Final Accuracy</Text>
                <Text style={[styles.kpiValue, { color: C.primary }]}>{resonaMetrics.avg_final_accuracy || '—'}</Text>
                <Text style={styles.kpiSub}>3 tasks · Replay Buffer</Text>
              </View>
            </View>

            {/* Active Issues */}
            <Text style={styles.sectionTitle}>Active Issues</Text>
            <View style={styles.card}>
              {alerts.length === 0 ? (
                <Text style={{ fontSize: 13, color: C.textMuted, paddingVertical: 10, textAlign: 'center' }}>No active anomalies</Text>
              ) : (
                alerts.map((a, i) => (
                  <View key={a.id}>
                    <View style={styles.issueRow}>
                      <View style={styles.issueLeft}>
                        <View style={[styles.issueDot, { backgroundColor: a.decision === 'ANOMALOUS' ? C.red : C.amber }]} />
                        <View>
                          <Text style={styles.issueName}>{a.machine_name || `Machine ${a.machine_id}`} · {a.zone}</Text>
                          <Text style={styles.issueDesc}>
                            {a.decision === 'ANOMALOUS' ? 'Known acoustic anomaly detected' : 'Unknown OOD pattern detected'}
                          </Text>
                        </View>
                      </View>
                      {a.decision === 'UNKNOWN' && (
                        <TouchableOpacity style={styles.issueBtnAmber} onPress={() => { setScreen('review'); }}>
                          <Text style={[styles.issueBtnText, { color: C.amber }]}>Review</Text>
                        </TouchableOpacity>
                      )}
                    </View>
                    {i < alerts.length - 1 && <View style={[styles.divider]} />}
                  </View>
                ))
              )}
            </View>

            {/* CL Pipeline Summary */}
            <Text style={styles.sectionTitle}>Continual Learning Pipeline</Text>
            <View style={styles.card}>
              <View style={styles.pipelineRow}>
                {['Human\nVerify', 'Review\nAgent', 'Replay\nBuffer', 'Model\nUpdate'].map((step, i, arr) => (
                  <React.Fragment key={step}>
                    <View style={styles.pipelineStep}>
                      <View style={styles.pipelineNode} />
                      <Text style={styles.pipelineLabel}>{step}</Text>
                    </View>
                    {i < arr.length - 1 && <Text style={styles.pipelineArrow}>→</Text>}
                  </React.Fragment>
                ))}
              </View>
              <Text style={styles.pipelineDesc}>Operator feedback → Labeled embeddings → Replay cache → Nightly retrain</Text>
            </View>
          </View>
        )}

        {/* ══════════ REVIEW ══════════ */}
        {screen === 'review' && (
          <View style={styles.page}>
            <View style={styles.reviewHeader}>
              <View style={[styles.reviewBadge, { backgroundColor: activeReview ? C.amberBg : C.border }]}>
                <Text style={[styles.reviewBadgeText, { color: activeReview ? C.amber : C.textSecondary }]}>
                  {activeReview ? `REVIEW ID: ${activeReview.id.substring(0,8).toUpperCase()}` : 'NO PENDING REVIEWS'}
                </Text>
              </View>
              <Text style={styles.pageTitle}>Human Review Queue</Text>
              <Text style={styles.pageSubtitle}>Conditions requiring operator verification for human-in-the-loop validation</Text>
            </View>

            {!activeReview ? (
               <View style={styles.card}>
                 <Text style={{ fontSize: 13, color: C.textMuted, paddingVertical: 20, textAlign: 'center' }}>You're all caught up! No reviews pending.</Text>
               </View>
            ) : (
              <>
                {/* AI Advisory */}
                <View style={[styles.advisory, { backgroundColor: '#EFF6FF', borderLeftColor: C.primary }]}>
                  <Text style={[styles.advisoryTitle, { color: C.primary }]}>AI Perception Advisory</Text>
                  <Text style={styles.advisoryText}>{activeReview.machine_name || `Machine ${activeReview.machine_id}`} encountered an unfamiliar acoustic pattern (OOD score: {activeReview.ood_score?.toFixed(1) || 'N/A'}). Human verification needed to update the Continual Learning replay buffer.</Text>
                </View>

                {/* Machine Info */}
                <View style={styles.card}>
                  <View style={styles.machineHeader}>
                    <View>
                      <Text style={styles.machineName}>{activeReview.machine_name || `Machine ${activeReview.machine_id}`}</Text>
                      <Text style={styles.machineId}>Real-time OOD Score: {activeReview.ood_score?.toFixed(1) || '—'}</Text>
                    </View>
                    <View style={[styles.statusPill, { backgroundColor: C.amberBg, borderColor: C.amber }]}>
                      <View style={[styles.statusDot, { backgroundColor: C.amber }]} />
                      <Text style={[styles.statusPillText, { color: C.amber, fontSize: 9 }]}>UNFAMILIAR</Text>
                    </View>
                  </View>
                </View>

                {/* Condition Selection */}
                <Text style={styles.sectionTitle}>Machine Condition *</Text>
                {[
                  { val: 'normal', label: 'NORMAL OPERATION', sublabel: 'Machine operates correctly', badge: 'Nominal', badgeColor: C.green, badgeBg: C.greenBg },
                  { val: 'anomalous', label: 'ANOMALOUS — FAULT', sublabel: 'Bearing friction, cavitation, etc.', badge: 'Alert L2', badgeColor: C.red, badgeBg: C.redBg },
                  { val: 'new_condition', label: 'NEW / UNRECOGNIZED', sublabel: 'Valid new state — AI will learn', badge: 'Learn', badgeColor: C.blue, badgeBg: C.blueBg },
                ].map(opt => (
                  <TouchableOpacity
                    key={opt.val}
                    style={[styles.conditionCard, selectedCondition === opt.val && styles.conditionCardSelected]}
                    onPress={() => setSelectedCondition(opt.val)}
                    activeOpacity={0.8}
                  >
                    <View style={[styles.radio, selectedCondition === opt.val && styles.radioSelected]}>
                      {selectedCondition === opt.val && <View style={styles.radioDot} />}
                    </View>
                    <View style={{ flex: 1 }}>
                      <View style={styles.conditionTop}>
                        <Text style={[styles.conditionLabel, selectedCondition === opt.val && { color: C.primary }]}>{opt.label}</Text>
                        <View style={[styles.badge, { backgroundColor: opt.badgeBg }]}>
                          <Text style={[styles.badgeText, { color: opt.badgeColor }]}>{opt.badge}</Text>
                        </View>
                      </View>
                      <Text style={styles.conditionSub}>{opt.sublabel}</Text>
                    </View>
                  </TouchableOpacity>
                ))}

                {/* Notes */}
                <Text style={styles.sectionTitle}>Operator Notes (Optional)</Text>
                <TextInput
                  style={styles.notesInput}
                  multiline
                  numberOfLines={4}
                  placeholder="e.g., Bearing grease cycle at 10:45 AM..."
                  value={notes}
                  onChangeText={setNotes}
                  placeholderTextColor={C.textMuted}
                />

                {/* Actions */}
                <TouchableOpacity style={styles.submitBtn} onPress={submitVerification} disabled={loading}>
                  <Text style={styles.submitBtnText}>SUBMIT VERIFICATION</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.assignBtn} onPress={() => Alert.alert('Dispatched', 'Field dispatch alert sent to Zone C on-duty technician.')}>
                  <Text style={styles.assignBtnText}>Assign Field Technician</Text>
                </TouchableOpacity>

                {submitted && (
                  <View style={[styles.toast, { backgroundColor: C.greenBg }]}>
                    <Text style={[styles.toastText, { color: C.green }]}>✓ Verification recorded. Token dispatched to Continual Learning buffer.</Text>
                  </View>
                )}
              </>
            )}
          </View>
        )}

        {/* ══════════ DEMO ══════════ */}
        {screen === 'demo' && (
          <View style={styles.page}>
            <Text style={styles.pageTitle}>Live Inference Demo</Text>
            <Text style={styles.pageSubtitle}>Simulate machine conditions · Backend: {API_BASE}</Text>

            {/* Current State */}
            {latestInference ? (
              <View style={[styles.card, { borderLeftWidth: 3, borderLeftColor: getStatusStyle(latestInference.decision).dot }]}>
                <View style={styles.stateRow}>
                  <View style={[styles.statusPill, { backgroundColor: getStatusStyle(latestInference.decision).bg, borderColor: getStatusStyle(latestInference.decision).dot }]}>
                    <View style={[styles.statusDot, { backgroundColor: getStatusStyle(latestInference.decision).dot }]} />
                    <Text style={[styles.statusPillText, { color: getStatusStyle(latestInference.decision).color }]}>{latestInference.decision}</Text>
                  </View>
                  <Text style={{ fontSize: 10, color: C.textSecondary, fontFamily: 'monospace' }}>Machine: {latestInference.machine_id}</Text>
                </View>
                <View style={styles.metricRow}>
                  <View style={styles.metricItem}>
                    <Text style={styles.metricLabel}>CONFIDENCE</Text>
                    <Text style={styles.metricValue}>{latestInference.confidence ? (latestInference.confidence * 100).toFixed(1) + '%' : '—'}</Text>
                  </View>
                  <View style={styles.metricItem}>
                    <Text style={styles.metricLabel}>OOD SCORE</Text>
                    <Text style={styles.metricValue}>{latestInference.mahalanobis_distance ? latestInference.mahalanobis_distance.toFixed(1) : '—'}</Text>
                  </View>
                  <View style={styles.metricItem}>
                    <Text style={styles.metricLabel}>LATENCY</Text>
                    <Text style={styles.metricValue}>{latestInference.total_latency_ms ? Math.round(latestInference.total_latency_ms) + 'ms' : '—'}</Text>
                  </View>
                </View>
                <MiniSpectrogram status={latestInference.decision} />
              </View>
            ) : (
              <View style={styles.card}><Text style={{ fontSize: 13, color: C.textMuted, padding: 20, textAlign: 'center' }}>No live inference events yet.</Text></View>
            )}

            {/* Scenario buttons */}
            <Text style={styles.sectionTitle}>Narrative Demo Scenarios</Text>
            {[
              [1, '🟢 Normal Pump Operation', 'Pump 00 · Healthy baseline'],
              [2, '🔴 Known Anomaly (Bearing)', 'Pump 02 · CNN confidence 88%'],
              [3, '🟢 Normal w/ 0dB Noise', 'Noise-robust · 85% confidence'],
              [4, '🔴 Noisy Anomaly Signal', 'High noise · 79% confidence'],
              [5, '🟡 Unknown — OOD Flagged', 'Mahalanobis > Threshold · Human loop'],
              [6, '🟡 Trigger Human Verification', 'Monitoring → Human Review Agent'],
              [7, '🔵 Post-Learning Recognition', 'Learning Agent updated model'],
              [8, '🟣 Forgetting Evaluation', 'Verify historical task accuracy'],
            ].map(([tc, label, sub]) => (
              <TouchableOpacity key={tc} style={styles.demoBtn} onPress={() => runTestCase(tc)} activeOpacity={0.8}>
                <Text style={styles.demoBtnLabel}>{label}</Text>
                <Text style={styles.demoBtnSub}>{sub}</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

      </ScrollView>
    </SafeAreaView>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <MainApp />
    </SafeAreaProvider>
  );
}

// ─── STYLES ──────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: C.bg },
  scroll: { flex: 1 },
  page: { padding: 16, paddingBottom: 40 },
  pageTitle: { fontSize: 22, fontWeight: '300', color: C.textPrimary, letterSpacing: -0.5, marginBottom: 2 },
  pageSubtitle: { fontSize: 12, color: C.textMuted, fontFamily: 'monospace', marginBottom: 20 },
  sectionTitle: { fontSize: 11, fontWeight: '700', color: C.textSecondary, textTransform: 'uppercase', letterSpacing: 1, marginTop: 16, marginBottom: 8 },
  card: { backgroundColor: C.surface, borderRadius: 12, borderWidth: 1, borderColor: C.border, padding: 14, marginBottom: 10, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 2, elevation: 1 },
  divider: { height: 1, backgroundColor: C.border, marginVertical: 10 },

  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', backgroundColor: C.headerBg, borderBottomWidth: 1, borderBottomColor: C.border, paddingHorizontal: 16, paddingVertical: 10 },
  brand: { fontSize: 16, fontWeight: '700', color: C.textPrimary, letterSpacing: 1 },
  brandSub: { fontSize: 9, color: C.textMuted, fontFamily: 'monospace', letterSpacing: 0.5 },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 8 },

  tabBar: { flexDirection: 'row', backgroundColor: C.surface, borderBottomWidth: 1, borderBottomColor: C.border },
  tab: { flex: 1, paddingVertical: 10, alignItems: 'center', borderBottomWidth: 2, borderBottomColor: 'transparent', flexDirection: 'row', justifyContent: 'center', gap: 6 },
  tabActive: { borderBottomColor: C.textPrimary },
  tabText: { fontSize: 12, color: C.textMuted, fontWeight: '500' },
  tabTextActive: { color: C.textPrimary, fontWeight: '700' },
  badgeCount: { backgroundColor: C.red, borderRadius: 10, paddingHorizontal: 5, paddingVertical: 1 },
  badgeCountText: { color: '#FFF', fontSize: 9, fontWeight: 'bold' },

  loadingBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', padding: 8, backgroundColor: '#EFF6FF', gap: 8 },
  loadingText: { fontSize: 12, color: C.primary, fontFamily: 'monospace' },

  statusPill: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 999, borderWidth: 1, gap: 5 },
  statusDot: { width: 6, height: 6, borderRadius: 3 },
  statusPillText: { fontSize: 10, fontWeight: '700', fontFamily: 'monospace', textTransform: 'uppercase' },

  kpiGrid: { flexDirection: 'row', gap: 10, marginBottom: 10 },
  kpiCard: { flex: 1, backgroundColor: C.surface, borderRadius: 12, borderWidth: 1, borderColor: C.border, padding: 14, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 2 },
  kpiLabel: { fontSize: 10, color: C.textMuted, fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6 },
  kpiValue: { fontSize: 26, fontWeight: '300', marginBottom: 2 },
  kpiSub: { fontSize: 10, color: C.textMuted },

  issueRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  issueLeft: { flexDirection: 'row', alignItems: 'center', gap: 10, flex: 1 },
  issueDot: { width: 8, height: 8, borderRadius: 4, flexShrink: 0 },
  issueName: { fontSize: 13, fontWeight: '600', color: C.textPrimary },
  issueDesc: { fontSize: 11, color: C.textSecondary, marginTop: 1 },
  issueBtnRed: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 6, backgroundColor: C.redBg, borderWidth: 1, borderColor: '#FCA5A5' },
  issueBtnAmber: { paddingHorizontal: 10, paddingVertical: 5, borderRadius: 6, backgroundColor: C.amberBg, borderWidth: 1, borderColor: '#FDE68A' },
  issueBtnText: { fontSize: 11, fontFamily: 'monospace', fontWeight: '700', color: C.textPrimary },

  pipelineRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 },
  pipelineStep: { alignItems: 'center', gap: 4 },
  pipelineNode: { width: 8, height: 8, borderRadius: 4, backgroundColor: C.primary },
  pipelineLabel: { fontSize: 9, color: C.textSecondary, textAlign: 'center', fontFamily: 'monospace' },
  pipelineArrow: { color: C.textMuted, fontSize: 14, paddingBottom: 12 },
  pipelineDesc: { fontSize: 10, color: C.textMuted, fontFamily: 'monospace', lineHeight: 14 },

  reviewHeader: { marginBottom: 14 },
  reviewBadge: { alignSelf: 'flex-start', paddingHorizontal: 8, paddingVertical: 3, borderRadius: 4, marginBottom: 6 },
  reviewBadgeText: { fontSize: 10, fontFamily: 'monospace', fontWeight: '700' },

  advisory: { borderLeftWidth: 3, borderRadius: 8, padding: 12, marginBottom: 12 },
  advisoryTitle: { fontSize: 13, fontWeight: '700', marginBottom: 4 },
  advisoryText: { fontSize: 12, color: C.textSecondary, lineHeight: 18 },

  machineHeader: { flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10 },
  machineName: { fontSize: 14, fontWeight: '600', color: C.textPrimary },
  machineId: { fontSize: 11, color: C.textMuted, marginTop: 2 },

  conditionCard: { backgroundColor: C.surface, borderRadius: 10, borderWidth: 1, borderColor: C.border, padding: 12, marginBottom: 8, flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  conditionCardSelected: { borderColor: C.primary, backgroundColor: '#EFF6FF' },
  radio: { width: 18, height: 18, borderRadius: 9, borderWidth: 2, borderColor: C.borderMed, alignItems: 'center', justifyContent: 'center', marginTop: 1 },
  radioSelected: { borderColor: C.primary },
  radioDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: C.primary },
  conditionTop: { flexDirection: 'row', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 2 },
  conditionLabel: { fontSize: 12, fontWeight: '700', fontFamily: 'monospace', color: C.textPrimary },
  conditionSub: { fontSize: 11, color: C.textMuted },
  badge: { paddingHorizontal: 6, paddingVertical: 2, borderRadius: 4 },
  badgeText: { fontSize: 9, fontFamily: 'monospace', fontWeight: '700', textTransform: 'uppercase' },

  notesInput: { backgroundColor: C.surface, borderRadius: 10, borderWidth: 1, borderColor: C.border, padding: 12, fontSize: 13, color: C.textPrimary, textAlignVertical: 'top', minHeight: 80, marginBottom: 12 },

  submitBtn: { backgroundColor: C.primary, borderRadius: 10, paddingVertical: 14, alignItems: 'center', marginBottom: 8, shadowColor: C.primary, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 3 },
  submitBtnText: { color: '#FFFFFF', fontSize: 13, fontWeight: '700', fontFamily: 'monospace', letterSpacing: 0.5 },
  assignBtn: { backgroundColor: C.surface, borderRadius: 10, paddingVertical: 12, alignItems: 'center', borderWidth: 1, borderColor: C.border, marginBottom: 12 },
  assignBtnText: { color: C.textSecondary, fontSize: 13, fontWeight: '600' },

  toast: { borderRadius: 10, padding: 12, marginTop: 4 },
  toastText: { fontSize: 12, fontWeight: '600', textAlign: 'center' },

  spectrogramContainer: { backgroundColor: '#111827', borderRadius: 8, padding: 10, marginTop: 4 },
  spectrogramBars: { flexDirection: 'row', alignItems: 'flex-end', height: 80, gap: 2, justifyContent: 'space-between' },
  spectrogramBar: { flex: 1, borderRadius: 2, minHeight: 4 },
  spectrogramLabel: { color: '#6B7280', fontSize: 9, fontFamily: 'monospace', marginTop: 6, textAlign: 'center' },

  stateRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 },
  metricRow: { flexDirection: 'row', gap: 8, marginBottom: 10 },
  metricItem: { flex: 1, backgroundColor: C.bg, borderRadius: 8, padding: 8 },
  metricLabel: { fontSize: 9, color: C.textMuted, fontFamily: 'monospace', textTransform: 'uppercase', marginBottom: 3 },
  metricValue: { fontSize: 13, fontWeight: '700', color: C.textPrimary, fontFamily: 'monospace' },

  demoBtn: { backgroundColor: C.surface, borderRadius: 10, borderWidth: 1, borderColor: C.border, padding: 12, marginBottom: 8 },
  demoBtnLabel: { fontSize: 13, fontWeight: '600', color: C.textPrimary, marginBottom: 2 },
  demoBtnSub: { fontSize: 10, color: C.textMuted, fontFamily: 'monospace' },
});
