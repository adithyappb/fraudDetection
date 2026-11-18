import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import Worklist from './components/Worklist'
import Viewer from './components/Viewer'
import ReviewPanel from './components/ReviewPanel'
import AuditLog from './components/AuditLog'
import ModelCard from './components/ModelCard'

const DISCLAIMER = 'FOR EDUCATION AND RESEARCH ONLY — NOT FOR CLINICAL USE. Not a medical device.'

export default function App() {
  const [tab, setTab] = useState('review')
  const [studies, setStudies] = useState([])
  const [selected, setSelected] = useState(null)
  const [study, setStudy] = useState(null)
  const [finding, setFinding] = useState(null)
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    try {
      const s = await api.studies()
      setStudies(s)
      setError(null)
      if (!selected && s.length) setSelected(s[0].study_id)
    } catch (e) {
      setError(String(e))
    }
  }, [selected])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 3000)        // new studies arrive from Orthanc in the background
    return () => clearInterval(t)
  }, [refresh])

  const loadStudy = useCallback(async () => {
    if (!selected) return
    const s = await api.study(selected)
    setStudy(s)
    setFinding((f) => (s.ai?.findings?.some((x) => x.finding === f) ? f : s.ai?.findings?.[0]?.finding ?? null))
  }, [selected])

  useEffect(() => { loadStudy() }, [loadStudy])

  const onDecided = async () => { await loadStudy(); await refresh() }

  return (
    <div className="app">
      <div className="disclaimer" role="alert">⚠ {DISCLAIMER}</div>
      <header className="topbar">
        <div className="brand">
          <span className="logo">◉</span> MedImaging AI <span className="sub">Radiologist review workstation</span>
        </div>
        <nav>
          {[['review', 'Review'], ['audit', 'Audit log'], ['model', 'Model card & limits']].map(([k, label]) => (
            <button key={k} className={tab === k ? 'tab active' : 'tab'} onClick={() => setTab(k)}>{label}</button>
          ))}
        </nav>
      </header>
      {error && <div className="error">API unreachable: {error}</div>}
      {tab === 'review' && (
        <main className="review">
          <Worklist studies={studies} selected={selected} onSelect={setSelected} />
          {study ? (
            <>
              <Viewer key={study.study_id} study={study} finding={finding} />
              <ReviewPanel key={study.study_id} study={study} finding={finding} onFinding={setFinding} onDecided={onDecided} />
            </>
          ) : (
            <div className="empty">No studies yet — send DICOM to Orthanc (make send-studies).</div>
          )}
        </main>
      )}
      {tab === 'audit' && <AuditLog />}
      {tab === 'model' && <ModelCard />}
    </div>
  )
}
