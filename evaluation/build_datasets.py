"""Build the golden evaluation datasets in evaluation/datasets/.

Contexts are copied verbatim from data/sample_kb, so re-run this after editing the KB:
    python evaluation/build_datasets.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "sample_kb"
OUT = ROOT / "evaluation" / "datasets"
OUT.mkdir(parents=True, exist_ok=True)

HANDBOOK, RUNBOOK = "company_hr_handbook.md", "hr_operations_runbook.md"


def sections(fname):
    text = (KB / fname).read_text(encoding="utf-8").replace("\r\n", "\n")
    return {m.group(1).strip(): m.group(2).strip() for m in re.finditer(r"^## ([^\n]+)\n(.+?)(?=\n## |\Z)", text, re.S | re.M)}


S = {**{k: (HANDBOOK, v) for k, v in sections(HANDBOOK).items()}, **{k: (RUNBOOK, v) for k, v in sections(RUNBOOK).items()}}
KEY = {
    "annual": "Annual Leave", "sick": "Sick Leave", "remote": "Remote Work Policy",
    "hours": "Working Hours and Attendance", "payroll": "Payroll", "benefits": "Benefits",
    "conduct": "Code of Conduct", "onboarding": "New Employee Onboarding",
    "leave_support": "Leave Request Support", "priority": "Payroll Issue Priority",
    "records": "Employee Record Changes", "offboarding": "Offboarding",
}


def ctx(key):
    src, text = S[KEY[key]]
    return src, [text]


def golden(input, expected_output=None, section=None, **meta):
    g = {"input": input}
    if expected_output:
        g["expected_output"] = expected_output
    if section:
        src, context = ctx(section)
        g["context"] = context
        meta = {"expected_source": src, "section": KEY[section], **meta}
    g["additional_metadata"] = meta
    return g


def save(name, goldens):
    (OUT / f"{name}.json").write_text(json.dumps(goldens, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{name}: {len(goldens)}")


# 1. Router: kb vs direct --------------------------------------------------------
kb_q = [
    "How many sick days do I get?",
    "what's the remote work policy",
    "Can I carry forward unused leave?",
    "When is payday?",
    "A colleague keeps making offensive comments to me. What should I do?",
    "How do I update my bank account details?",
    "What is the maternity leave policy?",
    "Is there a travel expense reimbursement policy?",
    "Hi! How many annual leave days do I have?",
    "Thanks. Also, who do I report a payroll error to?",
    "my payslip is wrong",
    "What is the notice period if I resign?",
]
direct_q = [
    "Hello",
    "Good morning!",
    "Thanks for the help!",
    "Bye, have a nice day",
    "How are you doing today?",
    "Thank you, that answered my question.",
    "Hey there 👋",
    "Great, cheers!",
]
save("router", [golden(q, expected_route="kb") for q in kb_q] + [golden(q, expected_route="direct") for q in direct_q])

# 2. Retrieval: paraphrased queries -> expected chunk ------------------------------
save("retrieval", [
    golden("vacation days allowance", section="annual"),
    golden("What happens to leave I don't use by the end of the year?", section="annual"),
    golden("Do I need a doctor's note if I'm off sick for three days?", section="sick"),
    golden("hybrid work rules", section="remote"),
    golden("What time does the workday start?", section="hours"),
    golden("What happens if someone is repeatedly late to work?", section="hours"),
    golden("payday date", section="payroll"),
    golden("Is there health insurance?", section="benefits"),
    golden("employee assistance program", section="benefits"),
    golden("Can I be punished for raising a complaint?", section="conduct"),
    golden("first-day checklist for new joiners", section="onboarding"),
    golden("HR portal is down, how do I log a leave request?", section="leave_support"),
    golden("What counts as a P1 payroll incident?", section="priority"),
    golden("Wrong tax deduction on my payslip, what priority is that?", section="priority"),
    golden("I moved house, how do I change my address?", section="records"),
    golden("update emergency contact", section="records"),
    golden("Do I have to return my laptop when I leave the company?", section="offboarding"),
    golden("When is system access removed for departing employees?", section="offboarding"),
])

# 3. KB evidence grader: is the KB sufficient? ------------------------------------
good = [
    ("How many days of paid annual leave do full-time employees get?", "annual"),
    ("How many paid sick days are employees allowed per year?", "sick"),
    ("How many days per week can I work remotely?", "remote"),
    ("What are the standard office hours?", "hours"),
    ("On what day is the monthly salary paid?", "payroll"),
    ("What benefits do eligible full-time employees receive?", "benefits"),
    ("Who can I report bullying to?", "conduct"),
    ("What priority is a single employee who has not received their salary?", "priority"),
    ("How should HR handle offboarding for a planned departure?", "offboarding"),
    ("What documents must HR confirm before a new employee starts?", "onboarding"),
]
weak = [
    "How many weeks of paternity leave does the company offer?",
    "What is the notice period for resignation?",
    "What is the daily travel per diem for business trips?",
    "How often are performance reviews conducted?",
    "Does the company reimburse gym memberships?",
    "How many days of bereavement leave can I take?",
    "How much is the employee referral bonus?",
    "What is the overtime pay rate?",
]
save("kb_grader",
     [golden(q, section=s, expected_grade="good") for q, s in good]
     + [golden(q, expected_grade="weak", reason="Topic not covered by the company knowledge base") for q in weak])

# 4. KB answer generation ---------------------------------------------------------
save("kb_generation", [
    golden("How many days of paid annual leave do full-time employees get?",
           "Full-time employees receive 20 days of paid annual leave per calendar year.", "annual"),
    golden("How far in advance should I submit a leave request?",
           "Leave requests should normally be submitted through the HR portal at least 5 working days before the planned leave date, and manager approval is required.", "annual"),
    golden("Can I carry over unused annual leave to next year?",
           "Yes, up to 5 days of unused leave can be carried forward into the next calendar year, and it must be used by March 31 unless HR grants an exception.", "annual"),
    golden("How many paid sick days do I get per year?",
           "Employees may take up to 10 paid sick days per calendar year.", "sick"),
    golden("When might HR ask me for a medical certificate?",
           "HR may request appropriate medical documentation for absences longer than 2 consecutive working days.", "sick"),
    golden("Who should I tell when I'm sick, and when?",
           "Notify your manager as early as possible on the first day of absence.", "sick"),
    golden("How many days a week can I work from home?",
           "Eligible employees may work remotely up to 2 days per week with manager approval.", "remote"),
    golden("Can I look after my children while working remotely?",
           "No. Remote work is not a substitute for dependent-care arrangements during working hours.", "remote"),
    golden("What are the standard working hours?",
           "Standard working hours are 9:00 AM to 6:00 PM, Monday through Friday, with a 1-hour meal break.", "hours"),
    golden("Can my team work flexible hours?",
           "Yes, teams may use approved flexible schedules as long as business coverage is maintained.", "hours"),
    golden("When is salary paid?",
           "Monthly salary is normally paid on the final business day of each month.", "payroll"),
    golden("My payslip looks wrong. What should I do, and by when?",
           "Report the discrepancy to Payroll Support within 5 business days of receiving the payslip.", "payroll"),
    golden("What benefits do full-time employees receive?",
           "Eligible full-time employees receive the company medical plan, life insurance coverage, and access to the employee assistance program.", "benefits"),
    golden("Can I change my benefits enrollment in the middle of the year?",
           "Only with a qualifying life event and supporting documentation; otherwise changes happen during the annual enrollment window.", "benefits"),
    golden("How do I report harassment at work?",
           "Report it to a manager, HR, or the confidential ethics channel. Retaliation against employees who raise concerns in good faith is prohibited.", "conduct"),
    golden("What should HR do if an employee can't submit leave through the portal?",
           "First verify the employee has an active profile and sufficient leave balance. If the portal is unavailable, record the request by email, ask the manager to confirm approval, and update the HR system once service is restored.", "leave_support"),
    golden("What priority is an individual employee not receiving their salary?",
           "It is a P2 payroll issue.", "priority"),
    golden("How do I update my bank details?",
           "Submit the change through the HR portal with any required supporting documents. Bank-detail changes are verified before the next payroll cutoff.", "records"),
    golden("What must HR confirm before a new employee's first day?",
           "The signed offer, identity and payroll documentation, manager assignment, start date, and required policy acknowledgements.", "onboarding"),
    golden("What steps are involved in offboarding an employee?",
           "Confirm the final working day, manager handover, return of company assets, benefit termination timing, final payroll inputs, and required exit documentation, and coordinate access removal with IT and the manager.", "offboarding"),
])

# 5. Web fallback: public HR facts not in the KB ---------------------------------
WEB_NOTE = "The answer must state that this is external public information to be validated with HR."
web = [
    ("What is the statutory maternity leave duration in India?",
     "Under the Maternity Benefit Act, women are entitled to 26 weeks of paid maternity leave for their first two children (12 weeks from the third child)."),
    ("How many weeks of leave does the US FMLA provide?",
     "The FMLA provides eligible employees up to 12 workweeks of unpaid, job-protected leave in a 12-month period."),
    ("Who is eligible for FMLA leave in the US?",
     "Employees who have worked for a covered employer for at least 12 months, for at least 1,250 hours in the past 12 months, at a location with 50 or more employees within 75 miles."),
    ("What is the US federal minimum wage?", "The US federal minimum wage is $7.25 per hour."),
    ("What is the overtime rate under the US Fair Labor Standards Act?",
     "Non-exempt employees must be paid at least 1.5 times their regular rate for hours worked over 40 in a workweek."),
    ("How long can an employee keep health coverage under COBRA?",
     "COBRA generally allows continuation of group health coverage for up to 18 months after job loss or reduced hours, longer in some circumstances."),
    ("What does Title VII of the Civil Rights Act prohibit?",
     "Title VII prohibits employment discrimination based on race, color, religion, sex, and national origin."),
    ("Which US employers must comply with the Americans with Disabilities Act employment rules?",
     "Employers with 15 or more employees must comply and provide reasonable accommodations to qualified individuals with disabilities."),
    ("What age group does the US Age Discrimination in Employment Act protect?",
     "The ADEA protects employees and job applicants who are 40 years of age or older."),
    ("What does the US Equal Pay Act require?",
     "The Equal Pay Act requires that men and women in the same workplace receive equal pay for equal work."),
    ("When must a US employer complete Form I-9 for a new hire?",
     "Form I-9 verifies identity and employment authorization; the employer must complete Section 2 within 3 business days of the employee's first day of work."),
    ("How much statutory annual leave are UK workers entitled to?",
     "Most UK workers are entitled to 5.6 weeks of paid holiday a year, which is 28 days for someone working 5 days a week."),
    ("How long is statutory maternity leave in the UK?", "Statutory maternity leave in the UK is up to 52 weeks."),
    ("How much statutory paternity leave can UK employees take?", "Eligible employees can take either 1 or 2 weeks of statutory paternity leave."),
    ("What is the maximum average working week under the EU Working Time Directive?",
     "The EU Working Time Directive limits average weekly working time to 48 hours, including overtime."),
    ("What percentage of salary does an employee contribute to EPF in India?",
     "Employees typically contribute 12% of their basic wages plus dearness allowance to the Employees' Provident Fund."),
]
save("web_fallback", [golden(q, a, expected_route="kb", expected_kb_grade="weak", expected_source_used="web_search",
                             criteria=WEB_NOTE, facts_as_of="2026-11") for q, a in web])

# 6. Query rewriting --------------------------------------------------------------
rewrites = [
    ("leave??", "annual", "How many days of paid annual leave do employees get and how do I request it?", ["annual leave", "request"]),
    ("wfh", "remote", "What is the company remote work (work from home) policy?", ["remote work"]),
    ("when do we get paid", "payroll", "On what date is the monthly salary paid?", ["salary", "paid"]),
    ("sick note thing", "sick", "When is medical documentation required for sick leave?", ["sick leave", "medical documentation"]),
    ("carry over days", "annual", "Can unused annual leave be carried forward to the next year?", ["unused leave", "carried forward"]),
    ("someone's harassing me", "conduct", "How do I report workplace harassment?", ["harassment", "report"]),
    ("bank change", "records", "How do I update my bank details in the HR system?", ["bank details", "update"]),
    ("new joiner stuff", "onboarding", "What is the onboarding process for new employees?", ["onboarding", "new employee"]),
    ("quitting, what now", "offboarding", "What is the offboarding process when an employee leaves?", ["offboarding", "final working day"]),
    ("payslip wrong", "payroll", "How do I report a payroll discrepancy on my payslip?", ["payslip", "discrepancy"]),
    ("change insurance", "benefits", "Can I change my benefits enrollment outside the annual window?", ["benefits", "enrollment"]),
    ("office timings?", "hours", "What are the standard working hours?", ["working hours"]),
    ("portal broken, need leave", "leave_support", "How do I submit a leave request when the HR portal is unavailable?", ["leave request", "HR portal"]),
    ("salary not came", "priority", "What should I do if I have not received my salary?", ["salary", "not received"]),
    ("always late, will I get in trouble", "hours", "What is the policy on repeated lateness or unapproved absence?", ["lateness", "attendance"]),
]
save("query_rewrite", [golden(q, ideal, section=s, expected_keywords=kw,
                              criteria="Preserves the user's intent, uses clear HR terminology, does not answer the question.")
                       for q, s, ideal, kw in rewrites])

# 7. End-to-end: question -> final path + answer ---------------------------------
INSUFFICIENT = "The assistant should say it could not find reliable evidence and direct the employee to contact HR."
e2e = [
    golden("How many paid sick days do employees get?", "Up to 10 paid sick days per calendar year.", "sick", expected_source_used="knowledge_base"),
    golden("Can I work remotely three days a week?", "No. Eligible employees may work remotely up to 2 days per week with manager approval.", "remote", expected_source_used="knowledge_base"),
    golden("By when must unused carried-forward leave be used?", "By March 31, unless HR grants an exception.", "annual", expected_source_used="knowledge_base"),
    golden("Who do I report a payroll discrepancy to?", "Payroll Support, within 5 business days of receiving the payslip.", "payroll", expected_source_used="knowledge_base"),
    golden("What priority is a major payroll processing failure?", "P1.", "priority", expected_source_used="knowledge_base"),
    golden("Is retaliation for reporting harassment allowed?", "No. Retaliation against employees who raise concerns in good faith is prohibited.", "conduct", expected_source_used="knowledge_base"),
    golden("What is included in the employee benefits?", "The company medical plan, life insurance coverage, and access to the employee assistance program.", "benefits", expected_source_used="knowledge_base"),
    golden("What should happen on a new employee's first day?", "Provide access to the HR portal, benefits information, mandatory training, and the employee handbook.", "onboarding", expected_source_used="knowledge_base"),
    golden("Hi!", "A brief, friendly greeting offering HR help.", expected_source_used="direct"),
    golden("Thanks a lot!", "A brief acknowledgement.", expected_source_used="direct"),
    golden("Good evening", "A brief, friendly greeting.", expected_source_used="direct"),
    golden("Have a great weekend!", "A brief, friendly reply.", expected_source_used="direct"),
    golden("What does the US WARN Act require?", "Covered employers (generally 100+ employees) must give 60 days' advance written notice of plant closings and mass layoffs.", expected_source_used="web_search", criteria=WEB_NOTE),
    golden("What is India's POSH Act?", "The Sexual Harassment of Women at Workplace Act, 2013 requires employers with 10 or more employees to set up an Internal Committee to handle complaints.", expected_source_used="web_search", criteria=WEB_NOTE),
    golden("What is a 401(k) plan?", "An employer-sponsored US retirement savings plan that lets employees contribute part of their salary, often with employer matching and tax advantages.", expected_source_used="web_search", criteria=WEB_NOTE),
    golden("What does the US Pregnant Workers Fairness Act require?", "Employers with 15 or more employees must provide reasonable accommodations for known limitations related to pregnancy, childbirth, or related conditions.", expected_source_used="web_search", criteria=WEB_NOTE),
    golden("What is PeoplePrime Technologies' sabbatical policy?", INSUFFICIENT, expected_source_used="insufficient_evidence", criteria=INSUFFICIENT),
    golden("How much is the PeoplePrime employee referral bonus?", INSUFFICIENT, expected_source_used="insufficient_evidence", criteria=INSUFFICIENT),
    golden("How many days of bereavement leave does PeoplePrime offer?", INSUFFICIENT, expected_source_used="insufficient_evidence", criteria=INSUFFICIENT),
    golden("What is PeoplePrime's annual learning budget per employee?", INSUFFICIENT, expected_source_used="insufficient_evidence", criteria=INSUFFICIENT),
]
save("end_to_end", e2e)
