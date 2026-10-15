import json
from pathlib import Path


OUTPUT_FILE = Path("data/evaluation/generation_reference_answers.json")


reference_data = [
    {
        "query_id": "Q001",
        "question": "What are the conditions for granting anticipatory bail under Section 438 CrPC?",
        "reference_answer": "Anticipatory bail is a discretionary remedy intended to protect personal liberty from unnecessary arrest. Courts consider factors such as the nature and gravity of the accusation, the applicant's role, the possibility of fleeing, tampering with evidence, or influencing witnesses. Appropriate conditions may be imposed to ensure cooperation with the investigation."
    },
    {
        "query_id": "Q002",
        "question": "What principles govern the exercise of discretion when deciding an application for anticipatory bail?",
        "reference_answer": "The discretion to grant anticipatory bail must be exercised judicially after considering the facts and circumstances of the individual case. Relevant considerations include the seriousness of the accusation, the applicant's role, the possibility of absconding, interference with investigation, witness intimidation, and evidence tampering."
    },
    {
        "query_id": "Q003",
        "question": "When can an accused seek protection from arrest before being taken into custody?",
        "reference_answer": "A person who has reason to believe that they may be arrested for a non-bailable offence may seek anticipatory bail under the applicable statutory provision. The remedy protects personal liberty from unnecessary custodial arrest, subject to judicial discretion and the circumstances of the case."
    },
    {
        "query_id": "Q004",
        "question": "What factors should courts consider while deciding whether to grant bail?",
        "reference_answer": "Courts generally consider the nature and gravity of the accusation, the evidence available, the possible punishment, criminal antecedents, the likelihood of absconding, the possibility of influencing witnesses or tampering with evidence, and whether release would obstruct the course of justice."
    },
    {
        "query_id": "Q005",
        "question": "What is the relationship between personal liberty and the grant of bail?",
        "reference_answer": "Personal liberty is protected by Article 21 of the Constitution and is an important consideration in bail decisions. Courts must balance individual liberty with legitimate interests such as a fair investigation, administration of justice, protection of witnesses, and the interests of society."
    },
    {
        "query_id": "Q006",
        "question": "What limitations can be imposed while granting anticipatory bail?",
        "reference_answer": "Courts may impose reasonable conditions to ensure that the accused cooperates with the investigation and does not misuse the protection. Conditions may relate to attendance, cooperation with investigation, non-interference with witnesses or evidence, and other requirements appropriate to the case."
    },
    {
        "query_id": "Q007",
        "question": "Can anticipatory bail continue for an indefinite period?",
        "reference_answer": "The duration of anticipatory bail depends on the order of the court and the circumstances of the case. It is a discretionary statutory protection, and the court may determine appropriate terms and conditions while granting it."
    },
    {
        "query_id": "Q008",
        "question": "What circumstances may justify cancellation or denial of bail?",
        "reference_answer": "Bail may be denied or cancelled where circumstances indicate that the accused may interfere with investigation, tamper with evidence, intimidate or influence witnesses, flee from justice, obstruct proceedings, or misuse the liberty granted by the court."
    },
    {
        "query_id": "Q009",
        "question": "What safeguards must be followed during the arrest of an accused?",
        "reference_answer": "Arrest should not be made automatically merely because an allegation of an offence exists. Police officers must consider the statutory requirements governing arrest and have proper justification for exercising the power. Judicial authorities must also independently apply their mind before authorizing detention."
    },
    {
        "query_id": "Q010",
        "question": "What are the legal requirements governing arrest without a warrant?",
        "reference_answer": "The power to arrest without a warrant must be exercised according to the statutory conditions governing arrest. The existence of the power does not justify every arrest. Police officers must satisfy the applicable legal requirements and have sufficient justification for exercising the power."
    },
    {
        "query_id": "Q011",
        "question": "What principles govern police custody and judicial remand of an accused?",
        "reference_answer": "Police custody and judicial remand involve deprivation of personal liberty and therefore require compliance with legal safeguards. The arrest must be legally justified, and the Magistrate must independently apply judicial mind before authorizing detention."
    },
    {
        "query_id": "Q012",
        "question": "What rights does an arrested person have during criminal proceedings?",
        "reference_answer": "An arrested person is protected by constitutional and statutory safeguards relating to personal liberty and lawful detention. Arrest and detention must comply with law, and a person cannot be subjected to arbitrary or unnecessary deprivation of liberty."
    },
    {
        "query_id": "Q013",
        "question": "Under what circumstances can an accused challenge the legality of an arrest?",
        "reference_answer": "An arrest may be challenged where statutory requirements were not satisfied, where the police failed to properly consider the necessity of arrest, or where the arrest was arbitrary or without sufficient justification."
    },
    {
        "query_id": "Q014",
        "question": "What principles govern quashing of criminal proceedings under Section 482 CrPC?",
        "reference_answer": "The High Court may exercise its inherent jurisdiction to prevent abuse of the process of the court or otherwise secure the ends of justice. The power must be exercised carefully and sparingly, particularly where the allegations are being assessed at the initial stage of criminal proceedings."
    },
    {
        "query_id": "Q015",
        "question": "When can the High Court exercise its inherent powers to quash criminal proceedings?",
        "reference_answer": "The High Court may exercise its inherent powers where intervention is necessary to prevent abuse of the process of law or secure the ends of justice. This extraordinary jurisdiction must be exercised cautiously and is not normally a substitute for a criminal trial."
    },
    {
        "query_id": "Q016",
        "question": "What are the limitations on the High Court's power to quash criminal proceedings?",
        "reference_answer": "Although the power under Section 482 CrPC is wide, it must be exercised sparingly and with caution. The High Court should generally avoid detailed appreciation of evidence at the initial stage and should not ordinarily conduct a mini-trial."
    },
    {
        "query_id": "Q017",
        "question": "Can criminal proceedings be quashed when the allegations do not disclose an offence?",
        "reference_answer": "Yes. If the allegations, even when taken at face value, do not prima facie establish the ingredients of the alleged offence, the High Court may exercise its inherent jurisdiction to prevent abuse of process and secure the ends of justice."
    },
    {
        "query_id": "Q018",
        "question": "Should a High Court conduct a detailed evidentiary inquiry while deciding a quashing petition?",
        "reference_answer": "Generally, no. At the stage of considering a quashing petition, the High Court ordinarily examines whether the allegations prima facie disclose an offence rather than conducting a detailed appreciation of disputed evidence. Detailed examination of evidence is generally a matter for trial."
    },
    {
        "query_id": "Q019",
        "question": "What principles govern the evidentiary value of statements recorded during criminal investigation?",
        "reference_answer": "The evidentiary value of an investigative statement depends on the statutory provision under which it was recorded and the circumstances in which it was made. Courts must apply the relevant rules governing admissibility, reliability, voluntariness, and use of such statements."
    },
    {
        "query_id": "Q020",
        "question": "What factors are considered when assessing the credibility of a witness?",
        "reference_answer": "Courts consider the testimony as a whole, including its consistency, intrinsic reliability, surrounding circumstances, probabilities, and material contradictions. Minor inconsistencies do not automatically destroy credibility; the significance of a contradiction depends on whether it affects the core of the case."
    },
    {
        "query_id": "Q021",
        "question": "What is the evidentiary value of a statement recorded under Section 164 CrPC?",
        "reference_answer": "The evidentiary value of a Section 164 statement depends on the nature of the statement and the legal requirements governing its recording and use. A confession must satisfy the applicable requirements relating to voluntariness and reliability and cannot automatically be treated as conclusive proof."
    },
    {
        "query_id": "Q022",
        "question": "When can inconsistencies in witness statements affect the prosecution case?",
        "reference_answer": "Inconsistencies become significant when they are material and affect the core of the prosecution case or substantially undermine the reliability of the evidence. Minor discrepancies or normal variations do not necessarily destroy a witness's credibility."
    },
    {
        "query_id": "Q023",
        "question": "What conditions govern the grant of bail in offences under the Prevention of Money Laundering Act?",
        "reference_answer": "Section 45 of the PMLA imposes statutory conditions on bail in applicable offences. Courts must consider these statutory requirements together with the ordinary principles of bail and the facts and circumstances of the individual case."
    },
    {
        "query_id": "Q024",
        "question": "What are the statutory restrictions on granting bail under the PMLA?",
        "reference_answer": "Section 45 of the PMLA contains twin conditions for bail in applicable cases. The Public Prosecutor must be given an opportunity to oppose the application, and where the application is opposed, the court must be satisfied regarding the statutory requirements concerning the accused's guilt and likelihood of committing an offence while on bail."
    },
    {
        "query_id": "Q025",
        "question": "What principles govern the burden on an accused seeking bail under a special statute?",
        "reference_answer": "Where a special statute creates additional conditions for bail, the court must apply those statutory requirements in addition to ordinary bail principles. The accused must satisfy the requirements imposed by the particular statute before bail can be granted under that framework."
    },
    {
        "query_id": "Q026",
        "question": "What conditions apply to bail in cases involving offences under the NDPS Act?",
        "reference_answer": "Section 37 of the NDPS Act imposes stringent conditions for bail in specified offences. Where the Public Prosecutor opposes bail, the court must be satisfied that there are reasonable grounds for believing that the accused is not guilty and is not likely to commit an offence while on bail."
    },
    {
        "query_id": "Q027",
        "question": "How do statutory restrictions affect the grant of bail for serious offences?",
        "reference_answer": "Where legislation imposes special restrictions on bail, courts must comply with those statutory requirements in addition to ordinary bail principles. Constitutional considerations, including personal liberty under Article 21, also remain relevant to the assessment of detention and bail."
    },
    {
        "query_id": "Q028",
        "question": "What factors should courts consider when deciding bail under a special criminal statute?",
        "reference_answer": "Courts must consider the specific statutory restrictions, the nature and gravity of the offence, the available material, the likelihood of interference with investigation or witnesses, the possibility of absconding, the period of custody, and the constitutional interest in personal liberty."
    },
    {
        "query_id": "Q029",
        "question": "How does Article 21 protect personal liberty in criminal proceedings?",
        "reference_answer": "Article 21 protects life and personal liberty by requiring deprivation of liberty to occur according to law. In criminal proceedings, this principle informs safeguards relating to arrest, detention and bail and protects individuals against arbitrary deprivation of liberty."
    },
    {
        "query_id": "Q030",
        "question": "How should courts balance personal liberty with the interests of criminal investigation?",
        "reference_answer": "Courts must balance the individual's constitutional right to personal liberty against legitimate investigative and societal interests. Arrest or detention should not be routine or punitive, while the court must also ensure that an accused does not evade investigation, tamper with evidence, influence witnesses, or obstruct justice."
    }
]


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(reference_data, f, indent=4, ensure_ascii=False)

    print("=" * 60)
    print("REFERENCE ANSWER DATASET CREATED")
    print("=" * 60)
    print(f"Total questions: {len(reference_data)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()