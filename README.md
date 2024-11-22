# CyberGuard_IndiaAI_Hackathon

## Use Python>=3.9
## --> source env/bin/activate
## --> pip install -r requirements.txt
## --> python3 inference.py

The project focuses on developing an advanced Natural Language Processing (NLP) model to classify fraud types based on victim descriptions, addressing inefficiencies in cybercrime reporting systems. Initial data analysis revealed that 30% of the dataset was inaccurately labelled. Using unsupervised learning techniques, clustering was performed to identify inconsistencies, and manual annotation refined these clusters. A highly precise semi-supervised model was implemented to re-label incorrect entries effectively. Synthetic data generation, supported by manual efforts and Large Language Models (LLMs), enriched the dataset further. Finally, a BERT-based text classification model was developed to classify reports into subcategories accurately

### Goal:
To build a series of machine learning models tailored for different levels of classification granularity.
---
### Models to Build:
1. **Model 1**:  
   - **Purpose**: Classify data into one of the three `main_category` values:  
     ["Women/Child Related Crime", "Financial Fraud Crimes", "Other Cyber Crime"]
2. **Model 2**:  
   - **Purpose**: Classify data into `category` values under **"Women/Child Related Crime."**  
   - **Classes**:  
     [
    "Child Pornography CPChild Sexual Abuse Material CSAM",
    "Crime Against Women & Children",
    "Online Cyber Trafficking",
    "RapeGang Rape RGRSexually Abusive Content",
    "Sexually Explicit Act",
    "Sexually Obscene material"
  ]
3. **Model 3**:  
   - **Purpose**: Classify data into `category` values under **"Financial Fraud Crimes."**  
   - **Classes**:  
     [
    "Cryptocurrency Crime",
    "Online Financial Fraud",
    "Online Gambling  Betting"
  ]
4. **Model 4**:  
   - **Purpose**: Classify data into `category` values under **"Other Cyber Crime."**  
   - **Classes**:  
     [
    "Any Other Cyber Crime",
    "Cyber Attack/ Dependent Crimes",
    "Cyber Terrorism",
    "Hacking  Damage to computercomputer system etc",
    "Online and Social Media Related Crime",
    "Ransomware",
    "Report Unlawful Content"
  ]  
5. **Model 5**:  
   - **Purpose**: Classify data into `sub_category` values under **"Hacking/Damage to computer system, etc."**  
   - **Classes**:  
     [
        "unauthorised accessdata breach",
        "email hacking",
        "tampering with computer source documents",
        "website defacementhacking",
        "damage to computer computer systems etc"
    ] 
6. **Model 6**:  
   - **Purpose**: Classify data into `sub_category` values under **"Online Financial Fraud."**  
   - **Classes**:  
     [
        "dematdepository fraud",
        "fraud callvishing",
        "internet banking related fraud",
        "business email compromiseemail takeover",
        "upi related frauds",
        "debitcredit card fraudsim swap fraud",
        "ewallet related fraud"
    ]  
7. **Model 7**:  
   - **Purpose**: Classify data into `sub_category` values under **"Any Other Cyber Crime."**  
   - **Classes**:  
     Includes a range of miscellaneous offenses such as:  
     [
        "identity theft, spoofing, and phishing attacks ",
        "zero-day exploits",
        "attacks on servers (database mail dns) and network devices (routers)",
        "attacks on applications (e.g., e-governance, e-commerce)",
        "other",
        "attacks on critical infrastructure, scada, operational technology systems, and wireless networks",
        "fake mobile apps",
        "password attacks",
        "attacks or suspicious activities affecting cloud computing systems, servers, software, and applications",
        "disinformation or misinformation campaigns",
        "unauthorized social media access",
        "attacks or malicious suspicious activities affecting systems related to big data blockchain virtual assets and robotics",
        "attacks or incidents affecting digital payment systems",
        "child pornography/child sexual abuse material (csam)",
        "malicious code attacks (specifically mentioning virus, worm, trojan, bots, spyware, cryptominers)",
        "aadhar enabled payment system (aeps) fraud",
        "attacks or suspicious activities affecting cloud computing systems servers software and applications",
        "cyber blackmailing/threatening",
        "compromise of critical systems/information",
        "supply chain attacks",
        "sale publishing and transmitting obscene material/sexually explicit material",
        "attacks on internet of things (iot) devices and associated systems, networks, and servers",
        "cyber espionage",
        "sexual harassment",
        "web application vulnerabilities",
        "attacks on systems related to artificial intelligence (ai) and machine learning (ml)",
        "targeted scanning/probing of critical networks/systems",
        "data leaks",
        "malicious mobile app attacks"
    ]
8. **Model 8**:  
   - **Purpose**: Classify data into `sub_category` values under **"Crime Against Women & Children."**  
   - **Classes**:  
     [
        "sexual harassment",
        "computer generated csam/csem",
        "cyber blackmailing & threatening"
    ]
9. **Model 9**:  
   - **Purpose**: Classify data into `sub_category` values under **"Cyber Attack/Dependent Crimes."**  
   - **Classes**:  
     [
        "hacking/defacement",
        "sql injection",
        "tampering with computer source documents",
        "malware attack",
        "denial of service (dos)/distributed denial of service (ddos) attacks",
        "ransomware attack",
        "data breach/theft",
        "defacement/hacking"
    ]
10. **Model 10**:  
    - **Purpose**: Classify data into `sub_category` values under **"Online and Social Media Related Crime."**  
    - **Classes**:  
      [
        "intimidating email",
        "cyber bullying  stalking  sexting",
        "impersonating email",
        "profile hacking identity theft",
        "online job fraud",
        "email phishing",
        "provocative speech for unlawful acts",
        "fakeimpersonating profile",
        "online matrimonial fraud",
        "cheating by impersonation"
    ]

