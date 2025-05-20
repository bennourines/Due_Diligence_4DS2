// data/dueDiligenceData.ts
export interface Section {
  id: number;
  title: string;
}

export interface Question {
  id: string;
  text: string;
}

export const SECTIONS: Section[] = [
  { id: 1, title: "Legal and Regulatory Compliance" },
  { id: 2, title: "Financial Due Diligence" },
  { id: 3, title: "Technical Due Diligence" },
  { id: 4, title: "Team and Governance" },
  { id: 5, title: "Market and Competitive Analysis" },
  { id: 6, title: "Roadmap and Development Progress" },
  { id: 7, title: "Risk and Red Flags" },
  { id: 8, title: "Environmental, Social, and Governance (ESG)" }
];

export const QUESTIONS: Record<number, Question[]> = {
  1: [
    { id: "1-1", text: "In which jurisdictions is the digital asset/cryptocurrency operating?" },
    { id: "1-2", text: "Has the project obtained all necessary licenses and registrations?" },
    { id: "1-3", text: "Are there any pending or past regulatory investigations or enforcement actions?" },
    { id: "1-4", text: "What is the legal structure of the entity behind the digital asset?" },
    { id: "1-5", text: "What is the legal classification of the digital asset (e.g., security, utility token, commodity)?" },
    { id: "1-6", text: "What are the legal rights and obligations of token holders?" }
  ],
  2: [
    { id: "2-1", text: "Are audited financial statements available?" },
    { id: "2-2", text: "What is the project's revenue model and financial performance?" },
    { id: "2-3", text: "What are the project's expenses and burn rate?" },
    { id: "2-4", text: "What is the token distribution and allocation?" },
    { id: "2-5", text: "What is the token's utility and value proposition?" },
    { id: "2-6", text: "Is there a vesting schedule for team and investor tokens to prevent market dumping?" }
  ],
  3: [
    { id: "3-1", text: "What blockchain technology is used, and what is the consensus mechanism?" },
    { id: "3-2", text: "What is the security of the blockchain network? Has it been audited for vulnerabilities?" },
    { id: "3-3", text: "Have the smart contracts been audited for security and functionality by reputable third parties?" },
    { id: "3-4", text: "Is the codebase open-source? Has it been audited by third-party security firms?" },
    { id: "3-5", text: "Are there any known vulnerabilities or past security breaches?" },
    { id: "3-6", text: "How are the digital assets custodied?" }
  ],
  4: [
    { id: "4-1", text: "Who are the founders, developers, and key team members, and what is their professional background?" },
    { id: "4-2", text: "Have the team members been involved in previous successful projects or any controversies?" },
    { id: "4-3", text: "Does the team have the necessary technical and industry expertise to deliver on their promises?" },
    { id: "4-4", text: "What is the governance structure of the project? Who holds decision-making power?" },
    { id: "4-5", text: "How are governance proposals submitted, voted on, and executed?" },
    { id: "4-6", text: "Are there any mechanisms to prevent governance attacks or manipulation?" }
  ],
  5: [
    { id: "5-1", text: "What problem does the cryptocurrency or digital asset aim to solve?" },
    { id: "5-2", text: "What is the unique selling point (USP) of the project compared to competitors?" },
    { id: "5-3", text: "What is the size of the target market, and what is the growth potential?" },
    { id: "5-4", text: "How many active users or wallets are there on the platform?" },
    { id: "5-5", text: "What is the current trading volume and liquidity of the token?" },
    { id: "5-6", text: "What are the real-world applications of the cryptocurrency or digital asset?" }
  ],
  6: [
    { id: "6-1", text: "Does the project have a clear and realistic roadmap? Are milestones being met on time?" },
    { id: "6-2", text: "What are the short-term and long-term goals of the project?" },
    { id: "6-3", text: "How active is the development team? Are there regular updates and code commits?" },
    { id: "6-4", text: "What innovative features or technologies does the project introduce?" },
    { id: "6-5", text: "How does the project stay ahead of technological advancements in the industry?" },
    { id: "6-6", text: "Are there any plans to integrate with emerging technologies (e.g., AI, IoT)?" }
  ],
  7: [
    { id: "7-1", text: "Are there any signs of a pump-and-dump scheme or market manipulation?" },
    { id: "7-2", text: "Does the project make unrealistic promises or guarantees of returns?" },
    { id: "7-3", text: "Has the project been involved in any scams, hacks, or controversies?" },
    { id: "7-4", text: "What are the regulatory risks to the portfolio?" },
    { id: "7-5", text: "How does the project adapt to potential regulatory changes?" },
    { id: "7-6", text: "What is the risk of a 'rug pull' or project abandonment by the team?" }
  ],
  8: [
    { id: "8-1", text: "What is the environmental impact of the cryptocurrency (e.g., energy consumption)?" },
    { id: "8-2", text: "Does the project rely on energy-intensive mining methods?" },
    { id: "8-3", text: "Are there initiatives to reduce energy consumption or carbon footprint?" },
    { id: "8-4", text: "Does the project promote social good or inclusivity?" },
    { id: "8-5", text: "How does the project address ethical concerns related to its operations?" },
    { id: "8-6", text: "Are there governance mechanisms in place to ensure fair decision-making?" }
  ]
};