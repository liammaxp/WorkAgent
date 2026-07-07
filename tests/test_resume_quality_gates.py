import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import api_server  # noqa: E402
import main as agent_main  # noqa: E402


class ResumeBulletQualityGateTests(unittest.TestCase):
    def test_bad_project_bullets_are_blocking(self):
        bad_bullets = [
            "Built scripting to support workflow.",
            "Built Python scripting to support job description analysis workflow.",
            "Implemented system for X, supporting A, B, C.",
            "Implemented an agent-style workflow that routes A, B, C, D.",
            "Built backend route/API workflow support for configuration-driven X, Y, Z.",
            "supporting X, Y, Z...",
            "Implemented FastAPI, React, and SQLite.",
        ]
        for bullet in bad_bullets:
            with self.subTest(bullet=bullet):
                result = api_server.validate_resume_bullet_quality(bullet, "Sample Project", "project")
                self.assertFalse(result["ok"])
                self.assertEqual(result["severity"], "blocking")

    def test_good_project_bullet_passes(self):
        bullet = (
            "Implemented database-backed persistence for records, project facts, generated artifacts, "
            "and review history to improve traceability across repeated tasks."
        )
        result = api_server.validate_resume_bullet_quality(bullet, "Sample Project", "project")
        self.assertTrue(result["ok"], result.get("issues"))

    def test_project_topic_dedupe_removes_repeated_workflow_bullets(self):
        bullets = [
            "Built workflow orchestration to route staged tasks through a repeatable review process.",
            "Implemented workflow routing to coordinate staged tasks through a traceable process.",
            "Developed orchestration workflow to connect staged tasks through maintainable handoffs.",
            "Automated routing workflow to support repeatable staged task coordination.",
        ]
        deduped = api_server.dedupe_project_bullet_topics(bullets, [])
        self.assertLess(len(deduped), len(bullets))


class TechnicalSkillsQualityGateTests(unittest.TestCase):
    def test_inventory_removes_pollution_and_retains_real_user_backed_skills(self):
        inventory = [
            {"skill": "Agent", "sources": ["project_evidence"], "category": "AI & Automation"},
            {"skill": "job description parsing", "sources": ["project_evidence"], "category": "AI & Automation"},
            {"skill": "Python", "sources": ["current_resume"], "category": "Languages"},
            {"skill": "React", "sources": ["project_evidence"], "category": "Backend & Databases"},
            {"skill": "AndroidX", "sources": ["project_evidence"], "category": "Frontend & Mobile"},
            {"skill": "OpenAI API", "sources": ["project_evidence"], "category": "Backend & Databases"},
            {"skill": "Gradle", "sources": ["project_evidence"], "category": "Frontend & Mobile"},
            {"skill": "Android Studio", "sources": ["project_evidence"], "category": "Testing, Build & Debugging"},
            {"skill": "Terraform", "sources": ["jd_keywords"], "category": "Tools"},
        ]
        cleaned = api_server.clean_skill_inventory_pollution(inventory)
        names = {entry["skill"] for entry in cleaned}
        self.assertNotIn("Agent", names)
        self.assertNotIn("job description parsing", names)
        self.assertNotIn("Terraform", names)
        self.assertIn("Python", names)
        self.assertIn("React", names)
        self.assertIn("AndroidX", names)
        self.assertIn("OpenAI API", names)
        self.assertIn("Gradle", names)
        self.assertIn("Android Studio", names)
        categories = {entry["skill"]: entry["category"] for entry in cleaned}
        self.assertEqual(categories["React"], "Frontend & Mobile")
        self.assertEqual(categories["OpenAI API"], "AI & Automation")
        self.assertEqual(categories["Gradle"], "Testing, Build & Debugging")
        self.assertEqual(categories["Android Studio"], "Tools")

    def test_rendered_skills_section_rejects_pollution_phrases(self):
        section = r"""
\section{Technical Skills}
\begin{itemize}[leftmargin=0.15in, label={}]
\small{
  \item{
    \textbf{Backend \& Databases:} API, job description parsing, OpenAI API
  }
}
\end{itemize}
"""
        result = api_server.validate_technical_skills_section(section, [])
        self.assertFalse(result["valid"])
        self.assertIn("API", result["pollution"])
        self.assertIn("job description parsing", result["pollution"])
        self.assertNotIn("OpenAI API", result["pollution"])

    def test_main_latex_validator_does_not_treat_pollution_as_fatal(self):
        latex = r"""
\documentclass{article}
\begin{document}
\section{Technical Skills}
\begin{itemize}[leftmargin=0.15in, label={}]
\small{
  \item{
    \textbf{Tools:} workflow coordination, GitHub
  }
}
\end{itemize}
\end{document}
"""
        issues = agent_main.technical_skills_section_issues(latex)
        self.assertFalse(any("workflow coordination" in issue for issue in issues))


class SummaryQualityTests(unittest.TestCase):
    def test_summary_rejects_boilerplate_and_excess_length(self):
        long_summary = " ".join(["candidate"] * 76) + " eager to contribute to the team."
        latex = "\\section{Summary}\n" + long_summary
        result = api_server.validate_summary_quality(latex)
        self.assertFalse(result["valid"])
        self.assertGreaterEqual(result["word_count"], 76)


class ResumeQualityGateRegressionTests(unittest.TestCase):
    def test_legal_experience_heading_does_not_block(self):
        latex = r"""
\documentclass{article}
\begin{document}
\section{Experience}
\resumeSubHeadingListStart
\resumeSubheading
{Company}{Feb 2020 -- Jan 2024}
{Programming Lead}{}
\resumeItemListStart
  \resumeItem{Implemented application features using a supported stack.}
\resumeItemListEnd
\resumeSubHeadingListEnd
\end{document}
"""
        gate = api_server.resume_quality_gate(latex, repair=True)
        self.assertTrue(gate["ok"], gate["blocking_issues"])

    def test_project_heading_with_nested_href_does_not_block(self):
        latex = r"""
\documentclass{article}
\begin{document}
\section{Projects}
\resumeSubHeadingListStart
\resumeProjectHeading
{\textbf{Project Name} $|$ \emph{Python, FastAPI, React}}
{\href{https://github.com/test/repo}{GitHub}}
\resumeItemListStart
  \resumeItem{Implemented database-backed persistence for project records to improve traceability across repeated tasks.}
\resumeItemListEnd
\resumeSubHeadingListEnd
\end{document}
"""
        gate = api_server.resume_quality_gate(latex, repair=True)
        self.assertTrue(gate["ok"], gate["blocking_issues"])

    def test_skills_pollution_is_cleaned_before_blocking(self):
        latex = r"""
\documentclass{article}
\begin{document}
\section{Technical Skills}
\begin{itemize}[leftmargin=0.15in, label={}]
\small{
  \item{
    \textbf{Languages:} Agent, Retrieval, workflow coordination, Python, FastAPI, React
  }
}
\end{itemize}
\end{document}
"""
        gate = api_server.resume_quality_gate(latex, repair=True)
        self.assertTrue(gate["ok"], gate["blocking_issues"])
        self.assertTrue(gate["cleanup"]["technical_skills"]["changed"])
        cleaned = gate["content"]
        self.assertNotIn("Agent", cleaned)
        self.assertNotIn("Retrieval", cleaned)
        self.assertNotIn("workflow coordination", cleaned)
        self.assertIn("Python", cleaned)
        self.assertIn("FastAPI", cleaned)
        self.assertIn("React", cleaned)

    def test_weak_bullet_warns_without_blocking(self):
        latex = r"""
\documentclass{article}
\begin{document}
\section{Projects}
\resumeSubHeadingListStart
\resumeProjectHeading{\textbf{Project}}{}
\resumeItemListStart
  \resumeItem{Built scripting to support workflow.}
\resumeItemListEnd
\resumeSubHeadingListEnd
\end{document}
"""
        gate = api_server.resume_quality_gate(latex, repair=True)
        self.assertTrue(gate["ok"], gate["blocking_issues"])
        self.assertTrue(any(issue["source"] == "bullets" for issue in gate["warnings"]))

    def test_true_structure_error_blocks_with_source_and_code(self):
        latex = r"""
\documentclass{article}
\begin{document}
\section{Experience}
\resumeSubheading
\resumeItemListStart
  \resumeItem{Broken entry.}
\resumeItemListEnd
\end{document}
"""
        gate = api_server.resume_quality_gate(latex, repair=False)
        self.assertFalse(gate["ok"])
        codes = {issue["code"] for issue in gate["blocking_issues"]}
        sources = {issue["source"] for issue in gate["blocking_issues"]}
        self.assertIn("structure", sources)
        self.assertIn("MISSING_RESUME_SUBHEADING_ARGS", codes)


if __name__ == "__main__":
    unittest.main()
