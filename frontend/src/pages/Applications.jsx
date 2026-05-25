import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  EditorCard,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";

const STATUS_OPTIONS = [
  "Interested",
  "Applied",
  "Interview",
  "Rejected",
  "Offer",
  "Archived",
];

const EMPTY_FORM = {
  company: "",
  role: "",
  link: "",
  status: "Interested",
  applied_date: "",
  resume_version: "tailored_resume.txt",
  cover_letter_version: "cover_letter.txt",
  notes: "",
};

export default function Applications() {
  const [records, setRecords] = useState([]);
  const [filter, setFilter] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const { loading, error, success, run } = useAsyncAction();

  const loadRecords = () =>
    run(async () => {
      const data = await api.getApplications(filter);
      setRecords(data);
      return data;
    });

  useEffect(() => {
    loadRecords();
  }, [filter]);

  const updateForm = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const createRecord = () =>
    run(async () => {
      await api.createApplication(form);
      setForm(EMPTY_FORM);
      await loadRecords();
    }, "申请记录已添加");

  const updateStatus = (id, status) =>
    run(async () => {
      await api.updateApplication(id, { status });
      await loadRecords();
    }, "状态已更新");

  return (
    <>
      <PageHeader
        title="申请记录"
        description="追踪各公司申请状态、使用的简历与求职信版本。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card">
        <h2 className="card-title">新增申请</h2>
        <div className="grid-2">
          <div className="field">
            <label>公司</label>
            <input value={form.company} onChange={(e) => updateForm("company", e.target.value)} />
          </div>
          <div className="field">
            <label>岗位</label>
            <input value={form.role} onChange={(e) => updateForm("role", e.target.value)} />
          </div>
          <div className="field">
            <label>链接</label>
            <input value={form.link} onChange={(e) => updateForm("link", e.target.value)} />
          </div>
          <div className="field">
            <label>状态</label>
            <select value={form.status} onChange={(e) => updateForm("status", e.target.value)}>
              {STATUS_OPTIONS.map((status) => (
                <option key={status} value={status}>
                  {status}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>申请日期</label>
            <input value={form.applied_date} onChange={(e) => updateForm("applied_date", e.target.value)} placeholder="YYYY-MM-DD" />
          </div>
          <div className="field">
            <label>备注</label>
            <input value={form.notes} onChange={(e) => updateForm("notes", e.target.value)} />
          </div>
        </div>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={createRecord} disabled={loading || !form.company || !form.role}>
            添加记录
          </button>
        </div>
      </section>

      <section className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h2 className="card-title" style={{ margin: 0 }}>申请列表</h2>
          <select value={filter} onChange={(e) => setFilter(e.target.value)} style={{ padding: "8px 12px", borderRadius: 10, border: "1px solid var(--border)" }}>
            <option value="">全部状态</option>
            {STATUS_OPTIONS.map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </div>

        {records.length === 0 ? (
          <p className="empty-state">暂无申请记录</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>公司</th>
                  <th>岗位</th>
                  <th>状态</th>
                  <th>申请日期</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {records.map((record) => (
                  <tr key={record.id}>
                    <td>{record.company}</td>
                    <td>{record.role}</td>
                    <td>
                      <select
                        value={record.status || "Interested"}
                        onChange={(e) => updateStatus(record.id, e.target.value)}
                        style={{ border: "none", background: "transparent", fontWeight: 600 }}
                      >
                        {STATUS_OPTIONS.map((status) => (
                          <option key={status} value={status}>
                            {status}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td>{record.applied_date || "—"}</td>
                    <td>
                      {record.link ? (
                        <a href={record.link} target="_blank" rel="noreferrer" style={{ color: "var(--accent)" }}>
                          链接
                        </a>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
